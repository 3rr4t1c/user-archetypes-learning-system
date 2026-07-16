"""Map a raw Bluesky/AT-Protocol export onto the canonical ArLeS schema.

This module is the *only* place that knows what an AT-URI is. Everything downstream
sees canonical actions, so porting ArLeS to another platform means writing a sibling
of this file and touching nothing else.

Three platform quirks are resolved here rather than leaking into the metrics.

1. The reshared post's author is in the URI, not in the file.
   An AT-Protocol record lives in its author's repository, so:

       at://did:plc:vlpy6zuqqum5tumv7b6dw5fp/app.bsky.feed.post/3l2dszqkmqt25
                    ^^^^^^^^^^^^^^^^^^^^^^^^ the author

   This matters because the export is sampled: the reshared post is itself present
   for only 0.09% of reposts (470 of 509,844 in bluesky_sampled_clean_small.csv).
   Resolving the author by looking the post up in the stream therefore fails 99.91%
   of the time; parsing the URI succeeds for 100%.

2. `author_user_id` is not the actor for follows and blocks.
   Measured over 600k rows, the DID in the record's own URI -- which is by definition
   the actor -- matches `author_user_id` 100% of the time for post/repost/reply/quote,
   but only 12.6% for follows and 18.1% for blocks; for those it matches
   `target_user_id` instead. Follows and blocks are ~51% of the stream, so taking
   `author_user_id` at face value credits half the archive's activity to the person on
   the receiving end. The URI is authoritative and is what we use.

3. Timestamps are client-supplied and come in several shapes.
   Handled by arles.actions.parse_timestamp. See also arles.streaming: the archive is
   ordered by ingestion, not by created_at, which can deviate by up to ~18.6 h.

Usage
-----
    python -m arles.mappers.bluesky /Volumes/Uniform/bluesky_full canonical.csv
    python -m arles.mappers.bluesky raw.csv canonical.csv --start 2024-10-12 --days 5

The output drops the `text` column, which is the bulk of the bytes, so the canonical
extract is substantially smaller and faster to re-read than the raw export.
"""

import argparse
import csv
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

from ..actions import MalformedActionError, author_of_uri, parse_timestamp
from ..schema import CANONICAL_COLUMNS, CanonicalAction, describe_coverage
from ..streaming import build_index, discover_files, iter_window

#: Bluesky collection NSID -> canonical activity_type.
#:
#: The raw export already carries a usable activity_type column; this is the fallback
#: for deriving it from the record's URI when that column is absent or unrecognised.
_COLLECTION_TO_TYPE = {
    "app.bsky.feed.post": "post",
    "app.bsky.feed.repost": "repost",
    "app.bsky.graph.follow": "follow",
    "app.bsky.graph.block": "block",
}

_COLLECTION_RE = re.compile(r"^at://[^/]+/([^/]+)/")

#: Raw activity_type values -> canonical. Bluesky's export distinguishes reply and
#: quote in this column even though the URI collection for both is app.bsky.feed.post.
_TYPE_ALIASES = {
    "post": "post",
    "repost": "repost",
    "reply": "reply",
    "quote": "quote",
    "follow": "follow",
    "block": "block",
}


def collection_of_uri(uri: str) -> Optional[str]:
    """Return the NSID collection segment of an AT-URI, e.g. 'app.bsky.feed.repost'."""
    if not uri:
        return None
    m = _COLLECTION_RE.match(uri)
    return m.group(1) if m else None


def map_row(row: Dict[str, str]) -> CanonicalAction:
    """Map one raw Bluesky row onto a CanonicalAction.

    Raises MalformedActionError if the row cannot be mapped. Callers count and report
    those rather than silently dropping them.
    """
    action_id = (row.get("action_id") or "").strip()
    if not action_id:
        raise MalformedActionError("missing action_id")

    # The URI's DID is the actor by construction: the record lives in that repo. This
    # is authoritative and, for follows/blocks, disagrees with author_user_id (see the
    # module docstring). Fall back only for did:web identities, which the URI parser
    # does not resolve.
    actor_id = author_of_uri(action_id) or (row.get("author_user_id") or "").strip()
    if not actor_id:
        raise MalformedActionError(f"cannot determine actor for {action_id!r}")

    raw_type = (row.get("activity_type") or "").strip().lower()
    activity_type = _TYPE_ALIASES.get(raw_type)
    if activity_type is None:
        collection = collection_of_uri(action_id)
        activity_type = _COLLECTION_TO_TYPE.get(collection or "", "other")

    parent_id = (row.get("original_action_id") or "").strip() or None
    # The whole point: recover the reshared post's author from its URI instead of
    # requiring the post itself to be present in a sampled export.
    parent_actor_id = author_of_uri(parent_id) if parent_id else None

    if activity_type in ("follow", "block") and not parent_id:
        # For graph edges the counterparty lives in target_user_id, not in a URI.
        parent_actor_id = (row.get("target_user_id") or "").strip() or None

    return CanonicalAction(
        action_id=action_id,
        actor_id=actor_id,
        activity_type=activity_type,
        created_at=parse_timestamp(row["created_at"]),
        parent_id=parent_id,
        parent_actor_id=parent_actor_id,
    )


def convert(
    source: str,
    destination: str,
    start=None,
    days: Optional[float] = None,
    progress: bool = True,
) -> Dict[str, int]:
    """Stream a raw export (file or directory) into a canonical CSV.

    Bounded memory: rows are written as they are read. Returns tally counts.
    """
    files = discover_files(source)
    spans = build_index(files, cache_path=None, verbose=progress) if progress else None
    if spans is None:
        spans = build_index(files, cache_path=None, verbose=False)

    end = start + timedelta(days=days) if (start and days) else None

    counts: Dict[str, int] = {}

    def bump(key, n=1):
        counts[key] = counts.get(key, 0) + n

    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(CANONICAL_COLUMNS))
        writer.writeheader()

        for row in iter_window(files, start, end, spans=spans, progress=progress):
            bump("read")
            try:
                action = map_row(row)
            except MalformedActionError:
                bump("skipped")
                continue
            bump(action.activity_type)
            if action.activity_type == "repost":
                if action.parent_actor_id:
                    bump("repost_attributed")
                if action.is_self_reshare:
                    bump("self_reshare")
            writer.writerow(action.to_row())
            bump("written")

    return counts


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("source", help="raw Bluesky CSV, or a directory of them")
    ap.add_argument("destination", help="canonical CSV to write")
    ap.add_argument("--start", default=None, help="window start (YYYY-MM-DD)")
    ap.add_argument("--days", type=float, default=None, help="window length in days")
    args = ap.parse_args()

    start = None
    if args.start:
        s = args.start.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            s = f"{s} 00:00:00+00:00"
        start = parse_timestamp(s)

    counts = convert(args.source, args.destination, start=start, days=args.days)

    print()
    print(f"read     : {counts.get('read', 0):,}")
    print(f"written  : {counts.get('written', 0):,}")
    print(f"skipped  : {counts.get('skipped', 0):,}")
    print()
    for t in ("post", "repost", "reply", "quote", "follow", "block", "other"):
        if counts.get(t):
            print(f"  {t:<8} {counts[t]:,}")
    print()
    print(describe_coverage(counts))


if __name__ == "__main__":
    main()
