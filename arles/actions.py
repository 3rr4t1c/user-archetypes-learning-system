"""Reading actions: timestamps, identifiers, and the raw Action record.

This module is deliberately small and boring. It knows how to turn one row of an
export into a parsed action, and nothing else -- no metrics, no learning, no I/O
strategy. Everything it does was, at some point, done wrong and silently:

  * parse_timestamp raises rather than defaulting. Its predecessor tried three
    strptime formats and fell back to datetime.now() when all three failed. Every
    timestamp in the dataset carries a "+00:00" offset, which all three reject, so
    every row took the fallback: a multi-day window collapsed into the seconds it
    took to read the file, and every time-dependent metric was computed on noise.
  * author_of_uri recovers a reshared post's author from the AT-URI rather than
    requiring the post itself to be present. On a sampled export the post is present
    for 0.09% of reposts, so the lookup-based approach failed 99.91% of the time --
    and returned zeros instead of an error.

The lesson in both: when a value cannot be determined, say so. A plausible default
propagates into results and cannot be seen afterwards.

For the canonical schema ArLeS actually consumes, see arles.schema; for platform
adapters, arles.mappers.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional





class MalformedActionError(ValueError):
    """Raised when an action row cannot be parsed.

    Never swallow this silently: a row that cannot be parsed must be counted and
    reported, not replaced by a plausible-looking default. See parse_timestamp.
    """


_TIMESTAMP_RE = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<frac>\d+))?"
    r"(?:\s*(?P<tz>Z|z|[+-]\d{2}:?\d{2}))?$"
)

_PLC_URI_RE = re.compile(r"^at://did:plc:([^/]+)/")


def author_of_uri(uri: Optional[str]) -> Optional[str]:
    """Return the author DID of an AT-URI, or None if it is not a did:plc URI.

    An AT-Protocol record lives in its author's repository, so the DID embedded in the
    URI *is* the author:

        at://did:plc:vlpy6zuqqum5tumv7b6dw5fp/app.bsky.feed.post/3l2dszqkmqt25
                     ^^^^^^^^^^^^^^^^^^^^^^^^ the author

    This matters because the dataset is a sample: the post being reshared is present in
    the file for only 0.09% of reposts (470 of 509,844 in bluesky_sampled_clean_small.csv),
    so resolving a reshare's author by looking the original post up in the stream fails
    almost always. Parsing the URI resolves 100% of them without needing the original.

    The returned id has no "did:plc:" prefix, matching the author_user_id column exactly
    (verified on 400k rows: for post/repost/reply/quote the parsed DID equals
    author_user_id 100% of the time).

    did:web identities (e.g. at://did:web:genco.me/...) are rare and return None.
    """
    if not uri:
        return None
    m = _PLC_URI_RE.match(uri)
    return m.group(1) if m else None


def parse_timestamp(value: Any) -> datetime:
    """Parse an action timestamp into a timezone-aware UTC datetime.

    The dataset stores ISO-8601 with an explicit UTC offset, in three variants that
    all occur in practice:

        2024-08-23 00:00:00+00:00              (no fractional part)
        2024-08-23 00:03:48.226000+00:00       (microseconds, the common case)
        2024-10-03 01:00:07.649171700+00:00    (nanoseconds, written by pandas)

    datetime.fromisoformat handles only the first two before Python 3.11, so the
    fractional part is normalised to 6 digits before parsing.

    Why this is strict: the previous implementation tried "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S" and "%Y-%m-%d" with strptime and fell back to
    datetime.now() when all three failed. Every one of them rejects the "+00:00"
    offset, so *every* row took the fallback and the whole action stream was
    replaced by the wall-clock time of the run. A multi-day window collapsed into
    the seconds it took to read the file, which zeroed the TASH-index outright and
    corrupted every other time-dependent metric. A timestamp that cannot be parsed
    is now an error, never a guess.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        m = _TIMESTAMP_RE.match(value.strip())
        if m is None:
            raise MalformedActionError(f"unparseable timestamp: {value!r}")

        head = m.group("head")
        frac = m.group("frac") or ""
        tz = m.group("tz") or "+00:00"

        # Truncate (or pad) the fractional part to microsecond resolution.
        micros = (frac + "000000")[:6]
        if tz in ("Z", "z"):
            tz = "+00:00"
        elif ":" not in tz:  # "+0000" -> "+00:00"
            tz = tz[:3] + ":" + tz[3:]

        try:
            dt = datetime.fromisoformat(f"{head}.{micros}{tz}")
        except ValueError as exc:  # pragma: no cover - regex should prevent this
            raise MalformedActionError(f"unparseable timestamp: {value!r}") from exc
    else:
        raise MalformedActionError(f"unparseable timestamp: {value!r}")

    # Normalise to UTC so that .timestamp() is comparable across rows.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class Action:
    """Represents a single social media action."""

    action_id: str
    created_at: datetime
    author_user_id: str
    target_user_id: Optional[str]
    original_action_id: Optional[str]
    activity_type: str  # post, repost, reply, quote, follow, block, ...
    text: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create Action from dictionary, handling datetime parsing.

        Raises MalformedActionError if the row cannot be parsed.
        """
        created_at = parse_timestamp(data["created_at"])
        if not data.get("action_id") or not data.get("author_user_id"):
            raise MalformedActionError("missing action_id or author_user_id")

        return cls(
            action_id=data["action_id"],
            created_at=created_at,
            author_user_id=data["author_user_id"],
            target_user_id=data.get("target_user_id"),
            original_action_id=data.get("original_action_id"),
            activity_type=data["activity_type"],
            text=data.get("text"),
        )
