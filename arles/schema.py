"""The canonical action schema that ArLeS consumes.

ArLeS learns archetypes from a stream of *actions*, not from platform APIs. Anything
with posts and reshares -- Bluesky, Mastodon, Twitter/X exports, a forum, a simulation
-- can be analysed by mapping it onto the six columns below. Platform-specific quirks
belong in a mapper (see arles.mappers), never in the metrics.

The canonical columns
---------------------
    action_id         unique id of this action
    actor_id          who performed it
    activity_type     one of ACTIVITY_TYPES
    created_at        ISO-8601 with a UTC offset
    parent_id         the content acted upon (reshared/replied/quoted); empty for posts
    parent_actor_id   who authored that content; empty for posts

Why `parent_actor_id` is its own column, and not derived
-------------------------------------------------------
Attributing a reshare to the original author is what the entire super-spreader axis
rests on. Deriving it by looking the parent post up in the stream fails whenever the
data is sampled: on the Bluesky sample, the original post is present for 0.09% of
reposts (470 of 509,844), which silently zeroed reshare yield, virality, the h-index
and the TASH-index.

Bluesky happens to embed the author in its AT-URI, so its mapper recovers this for
100% of reposts. Other platforms may not, and that is exactly why the requirement is
explicit here rather than hidden inside a metric: if your data cannot supply
`parent_actor_id`, the super-spreader axis is not estimable and you should know that
before you read a figure, not after.

What is deliberately absent
---------------------------
There is no `text` column. ArLeS is content-agnostic by design: it measures how users
act, not what they say. This is not only a modelling choice -- on the Bluesky archive
the text is the bulk of the bytes, so a canonical extract is far smaller and faster to
re-read than the raw export.

There is no follower graph, and no popularity metadata. Everything is computed from
the action stream itself.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .arles import MalformedActionError, parse_timestamp

#: The canonical column names, in order.
CANONICAL_COLUMNS = (
    "action_id",
    "actor_id",
    "activity_type",
    "created_at",
    "parent_id",
    "parent_actor_id",
)

#: Columns a mapper must always populate.
REQUIRED_COLUMNS = ("action_id", "actor_id", "activity_type", "created_at")

#: Controlled vocabulary for activity_type.
#:
#: post   -- original content authored by actor_id
#: repost -- verbatim reshare of parent_id, authored by parent_actor_id
#: reply  -- response to parent_id
#: quote  -- reshare of parent_id with added commentary
#: follow -- actor_id begins following parent_actor_id
#: block  -- actor_id blocks parent_actor_id
#: other  -- anything else; counted as activity, ignored by every metric
ACTIVITY_TYPES = frozenset(
    {"post", "repost", "reply", "quote", "follow", "block", "other"}
)

#: The activity types the archetype metrics actually read.
#:
#: Only `post` and `repost` carry diffusion. Replies and quotes are excluded, following
#: Verdolotti et al., which counts "only original posts and reshares, excluding
#: self-reshares, replies, and quotes, as these do not reflect content endorsement or
#: diffusion". Follows and blocks still count towards a user's activity volume (and so
#: towards their confidence score) but feed no metric.
DIFFUSION_TYPES = frozenset({"post", "repost"})


@dataclass(frozen=True)
class CanonicalAction:
    """One action, already mapped onto the canonical schema."""

    action_id: str
    actor_id: str
    activity_type: str
    created_at: datetime
    parent_id: Optional[str] = None
    parent_actor_id: Optional[str] = None

    @property
    def content_id(self) -> str:
        """The piece of content this action concerns.

        For a reshare that is the parent; for an original post it is the post itself.
        This is the key co-action metrics group on: two users acting on the same
        content_id within a short window are acting together.
        """
        return self.parent_id or self.action_id

    @property
    def is_self_reshare(self) -> bool:
        """A user resharing their own content. Excluded from diffusion accounting."""
        return (
            self.activity_type == "repost"
            and self.parent_actor_id is not None
            and self.parent_actor_id == self.actor_id
        )

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "CanonicalAction":
        """Build from a canonical row, validating rather than guessing.

        Raises MalformedActionError on anything unparseable. Never substitutes a
        default: the original implementation defaulted an unparseable timestamp to
        datetime.now(), which replaced the whole action stream with the wall clock and
        zeroed every time-dependent metric. Silence is the failure mode to avoid.
        """
        for column in REQUIRED_COLUMNS:
            if not row.get(column):
                raise MalformedActionError(f"missing required column {column!r}")

        activity_type = str(row["activity_type"]).strip().lower()
        if activity_type not in ACTIVITY_TYPES:
            raise MalformedActionError(
                f"unknown activity_type {activity_type!r}; expected one of "
                f"{sorted(ACTIVITY_TYPES)}. Map platform-specific names in your mapper."
            )

        parent_id = row.get("parent_id") or None
        parent_actor_id = row.get("parent_actor_id") or None

        if activity_type == "repost" and not parent_id:
            raise MalformedActionError("repost without parent_id")

        return cls(
            action_id=str(row["action_id"]),
            actor_id=str(row["actor_id"]),
            activity_type=activity_type,
            created_at=parse_timestamp(row["created_at"]),
            parent_id=parent_id,
            parent_actor_id=parent_actor_id,
        )

    def to_row(self) -> Dict[str, str]:
        """Serialise back to a canonical CSV row."""
        return {
            "action_id": self.action_id,
            "actor_id": self.actor_id,
            "activity_type": self.activity_type,
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent_id or "",
            "parent_actor_id": self.parent_actor_id or "",
        }


def describe_coverage(counts: Dict[str, int]) -> str:
    """Human-readable note on what a dataset can and cannot support.

    Written for the moment a user points ArLeS at their own export and wants to know
    which of the three axes their data can actually sustain.
    """
    lines = []
    reposts = counts.get("repost", 0)
    attributed = counts.get("repost_attributed", 0)

    if reposts == 0:
        lines.append(
            "No reposts: neither the super-spreader nor the amplifier axis is "
            "estimable. ArLeS needs reshares to measure diffusion."
        )
    else:
        share = 100 * attributed / reposts
        lines.append(f"Reposts with a known parent_actor_id: {share:.1f}%")
        if share < 50:
            lines.append(
                "  Below 50%: the super-spreader axis is unreliable. Most reshares "
                "cannot be credited to an author, so reshare yield, the h-index and "
                "the TASH-index are computed from a small, non-random subset. Supply "
                "parent_actor_id from your platform's identifiers if you can."
            )
    if counts.get("post", 0) == 0:
        lines.append("No posts: the super-spreader axis has nothing to measure.")
    return "\n".join(lines)
