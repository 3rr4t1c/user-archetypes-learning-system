"""Tests for the canonical schema and the Bluesky mapper.

The mapper carries three platform quirks that each caused a real, silent failure:

  * the reshared post's author lives in the AT-URI, not in the file (the post itself
    is present for 0.09% of reposts in the sampled export);
  * `author_user_id` is the *recipient* for follows and blocks, which are ~51% of the
    archive, so trusting it credits half the stream to the wrong person;
  * timestamps are client-supplied and arrive in several shapes.

Each is pinned below.
"""

from datetime import datetime, timezone

import pytest

from arles.arles import MalformedActionError
from arles.mappers.bluesky import collection_of_uri, map_row
from arles.schema import (
    ACTIVITY_TYPES,
    DIFFUSION_TYPES,
    CanonicalAction,
    describe_coverage,
)

POST_URI = "at://did:plc:n5tgqghvsmedkapljlpsquj7/app.bsky.feed.post/3l6gmxbqhwbj2"
REPOST_URI = "at://did:plc:t5taxt7lhgyu25oq5nevakqd/app.bsky.feed.repost/3l2dtwcpeen2t"
ORIG_URI = "at://did:plc:vlpy6zuqqum5tumv7b6dw5fp/app.bsky.feed.post/3l2dszqkmqt25"
FOLLOW_URI = "at://did:plc:xv5qwqopovrus4iyhf757nxu/app.bsky.graph.follow/3l2dua5w3r32s"


# --------------------------------------------------------------------------- schema


def test_canonical_action_roundtrips():
    a = CanonicalAction.from_dict({
        "action_id": "a1", "actor_id": "u1", "activity_type": "repost",
        "created_at": "2024-10-12T00:00:00.000Z",
        "parent_id": "p1", "parent_actor_id": "u2",
    })
    assert a.created_at == datetime(2024, 10, 12, tzinfo=timezone.utc)
    row = a.to_row()
    assert CanonicalAction.from_dict(row) == a


def test_content_id_is_the_parent_for_reshares_and_self_for_posts():
    repost = CanonicalAction("a1", "u1", "repost", datetime.now(timezone.utc), "p1", "u2")
    post = CanonicalAction("a2", "u1", "post", datetime.now(timezone.utc))
    assert repost.content_id == "p1"   # co-action groups on the reshared content
    assert post.content_id == "a2"


def test_self_reshare_detected():
    own = CanonicalAction("a1", "u1", "repost", datetime.now(timezone.utc), "p1", "u1")
    other = CanonicalAction("a2", "u1", "repost", datetime.now(timezone.utc), "p2", "u2")
    assert own.is_self_reshare
    assert not other.is_self_reshare


@pytest.mark.parametrize("missing", ["action_id", "actor_id", "activity_type", "created_at"])
def test_required_columns_are_required(missing):
    row = {"action_id": "a1", "actor_id": "u1", "activity_type": "post",
           "created_at": "2024-10-12T00:00:00.000Z"}
    row[missing] = ""
    with pytest.raises(MalformedActionError, match=missing):
        CanonicalAction.from_dict(row)


def test_unknown_activity_type_is_rejected_with_a_useful_message():
    with pytest.raises(MalformedActionError, match="mapper"):
        CanonicalAction.from_dict({
            "action_id": "a1", "actor_id": "u1", "activity_type": "boost",
            "created_at": "2024-10-12T00:00:00.000Z",
        })


def test_repost_without_parent_is_rejected():
    with pytest.raises(MalformedActionError, match="parent_id"):
        CanonicalAction.from_dict({
            "action_id": "a1", "actor_id": "u1", "activity_type": "repost",
            "created_at": "2024-10-12T00:00:00.000Z",
        })


def test_diffusion_types_exclude_replies_and_quotes():
    """Following Verdolotti et al.: replies and quotes are not endorsement/diffusion."""
    assert DIFFUSION_TYPES == {"post", "repost"}
    assert DIFFUSION_TYPES < ACTIVITY_TYPES


def test_schema_has_no_text_column():
    """ArLeS is content-agnostic by construction, not by convention."""
    from arles.schema import CANONICAL_COLUMNS

    assert "text" not in CANONICAL_COLUMNS


# --------------------------------------------------------------------------- mapper


def test_mapper_recovers_the_reshared_posts_author_from_the_uri():
    """The 0.09% -> 100% fix, at the boundary where it belongs."""
    a = map_row({
        "action_id": REPOST_URI, "activity_type": "repost",
        "created_at": "2024-10-12T00:00:00.000Z",
        "author_user_id": "t5taxt7lhgyu25oq5nevakqd",
        "original_action_id": ORIG_URI, "text": "ignored",
    })
    assert a.activity_type == "repost"
    assert a.actor_id == "t5taxt7lhgyu25oq5nevakqd"
    assert a.parent_actor_id == "vlpy6zuqqum5tumv7b6dw5fp"
    assert a.parent_id == ORIG_URI


def test_mapper_uses_the_uri_not_author_user_id_for_follows():
    """The regression for the ~51%-of-the-stream bug.

    A follow record lives in the follower's repo, so the URI's DID is the actor. The
    export puts the *recipient* in author_user_id and the actor in target_user_id.
    Trusting author_user_id credits the follow to the person who was followed.
    """
    a = map_row({
        "action_id": FOLLOW_URI, "activity_type": "follow",
        "created_at": "2024-10-12T00:00:00.000Z",
        "author_user_id": "wgkkm6dizceyedtwohk7naz3",     # the recipient
        "target_user_id": "xv5qwqopovrus4iyhf757nxu",     # the actual actor
        "original_action_id": "",
    })
    assert a.actor_id == "xv5qwqopovrus4iyhf757nxu"       # from the URI
    assert a.actor_id != "wgkkm6dizceyedtwohk7naz3"


def test_mapper_agrees_with_author_user_id_on_content_actions():
    """Where author_user_id *is* known-good, the URI must reproduce it exactly.

    Measured at 100% over 400k rows for post/repost/reply/quote; that agreement is
    what licenses trusting the URI where there is nothing to compare against.
    """
    a = map_row({
        "action_id": POST_URI, "activity_type": "post",
        "created_at": "2024-08-23 00:00:00+00:00",
        "author_user_id": "n5tgqghvsmedkapljlpsquj7",
        "original_action_id": "",
    })
    assert a.actor_id == "n5tgqghvsmedkapljlpsquj7"
    assert a.parent_id is None and a.parent_actor_id is None


def test_mapper_falls_back_to_author_user_id_for_did_web():
    """did:web identities are rare and not parsed by the URI helper."""
    a = map_row({
        "action_id": "at://did:web:genco.me/app.bsky.feed.post/3l2pwj5vwtk2u",
        "activity_type": "post", "created_at": "2024-08-23 00:00:00+00:00",
        "author_user_id": "genco.me", "original_action_id": "",
    })
    assert a.actor_id == "genco.me"


def test_mapper_handles_every_timestamp_shape_in_the_archive():
    for ts in ("2024-08-23 00:00:00+00:00",
               "2024-08-23 00:03:48.226000+00:00",
               "2024-10-03 01:00:07.649171700+00:00",
               "2024-08-23T00:00:00.000Z"):
        a = map_row({"action_id": POST_URI, "activity_type": "post", "created_at": ts,
                     "author_user_id": "n5tgqghvsmedkapljlpsquj7",
                     "original_action_id": ""})
        assert a.created_at.year == 2024


def test_mapper_derives_type_from_the_uri_when_the_column_is_unusable():
    a = map_row({
        "action_id": REPOST_URI, "activity_type": "",
        "created_at": "2024-10-12T00:00:00.000Z",
        "author_user_id": "t5taxt7lhgyu25oq5nevakqd",
        "original_action_id": ORIG_URI,
    })
    assert a.activity_type == "repost"


def test_unknown_platform_type_becomes_other_not_an_error():
    a = map_row({
        "action_id": "at://did:plc:abc/app.bsky.labeler.service/1",
        "activity_type": "weird", "created_at": "2024-10-12T00:00:00.000Z",
        "author_user_id": "abc", "original_action_id": "",
    })
    assert a.activity_type == "other"


def test_mapper_raises_on_an_unmappable_row():
    with pytest.raises(MalformedActionError):
        map_row({"action_id": "", "activity_type": "post",
                 "created_at": "2024-10-12T00:00:00.000Z", "author_user_id": ""})


def test_collection_of_uri():
    assert collection_of_uri(REPOST_URI) == "app.bsky.feed.repost"
    assert collection_of_uri(POST_URI) == "app.bsky.feed.post"
    assert collection_of_uri("nonsense") is None
    assert collection_of_uri("") is None


# ------------------------------------------------------------------- coverage note


def test_coverage_warns_when_reshares_cannot_be_attributed():
    note = describe_coverage({"post": 10, "repost": 1000, "repost_attributed": 1})
    assert "super-spreader axis is unreliable" in note


def test_coverage_is_quiet_when_attribution_is_good():
    note = describe_coverage({"post": 10, "repost": 1000, "repost_attributed": 1000})
    assert "unreliable" not in note
    assert "100.0%" in note
