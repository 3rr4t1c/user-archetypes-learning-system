"""Tests for action parsing.

These exist because of a specific bug: every timestamp in the dataset carries a
"+00:00" UTC offset, which datetime.strptime("%Y-%m-%d %H:%M:%S") rejects. The old
parser tried three such formats and fell back to datetime.now() when all of them
failed -- which was always. The entire action stream silently became "now", and
every time-dependent metric (TASH-index above all) was computed on a few seconds of
wall-clock instead of days of data.

The lesson encoded here: a timestamp that cannot be parsed must raise, never default.
"""

from datetime import datetime, timedelta, timezone

import pytest

from arles.actions import Action, MalformedActionError, author_of_uri, parse_timestamp


# Every fractional-second width that actually occurs in the dataset. Counted over
# 300k rows of bluesky_sampled_clean_small.csv: 0 digits (1,756), 6 digits (298,224),
# 9 digits (20). The 9-digit variant is written by pandas' datetime64[ns] and is
# rejected by datetime.fromisoformat before Python 3.11.
REAL_WORLD_TIMESTAMPS = [
    ("2024-08-23 00:00:00+00:00", datetime(2024, 8, 23, 0, 0, 0, tzinfo=timezone.utc)),
    (
        "2024-08-23 00:03:48.226000+00:00",
        datetime(2024, 8, 23, 0, 3, 48, 226000, tzinfo=timezone.utc),
    ),
    (
        "2024-10-03 01:00:07.649171700+00:00",
        datetime(2024, 10, 3, 1, 0, 7, 649171, tzinfo=timezone.utc),
    ),
]


@pytest.mark.parametrize("raw,expected", REAL_WORLD_TIMESTAMPS)
def test_parses_every_format_present_in_the_dataset(raw, expected):
    assert parse_timestamp(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "2024-08-23T00:00:00Z",  # trailing Z, rejected by fromisoformat < 3.11
        "2024-08-23 00:00:00+0000",  # offset without a colon
        "2024-08-23T00:00:00+00:00",  # T separator
    ],
)
def test_parses_iso8601_variants(raw):
    assert parse_timestamp(raw) == datetime(2024, 8, 23, tzinfo=timezone.utc)


def test_naive_timestamp_is_assumed_utc():
    assert parse_timestamp("2024-08-23 00:00:00") == datetime(
        2024, 8, 23, tzinfo=timezone.utc
    )


def test_offsets_are_normalised_to_utc():
    # 02:00 at +02:00 is midnight UTC.
    assert parse_timestamp("2024-08-23 02:00:00+02:00") == datetime(
        2024, 8, 23, tzinfo=timezone.utc
    )


@pytest.mark.parametrize(
    "raw", ["", "not-a-date", "2024-13-45 99:99:99+00:00", None, 12345]
)
def test_unparseable_timestamps_raise_and_never_default(raw):
    """The regression test for the original bug.

    The failure mode was not an exception -- it was a plausible-looking datetime.now()
    that propagated silently into every metric. Anything unparseable must raise.
    """
    with pytest.raises(MalformedActionError):
        parse_timestamp(raw)


def test_from_dict_does_not_fall_back_to_now():
    """Directly pins the original bug: a real dataset timestamp must not become now()."""
    action = Action.from_dict(
        {
            "action_id": "at://did:plc:abc/app.bsky.feed.post/3l2dszqkmqt25",
            "created_at": "2024-08-23 00:03:48.226000+00:00",
            "author_user_id": "abc",
            "activity_type": "post",
        }
    )
    assert action.created_at.year == 2024
    assert abs(action.created_at - datetime.now(timezone.utc)) > timedelta(days=365)


def test_from_dict_requires_identifiers():
    with pytest.raises(MalformedActionError):
        Action.from_dict(
            {"action_id": "", "created_at": "2024-08-23 00:00:00+00:00",
             "author_user_id": "abc", "activity_type": "post"}
        )


def test_author_of_uri_extracts_the_did():
    assert (
        author_of_uri("at://did:plc:vlpy6zuqqum5tumv7b6dw5fp/app.bsky.feed.post/3l2dszqkmqt25")
        == "vlpy6zuqqum5tumv7b6dw5fp"
    )
    assert (
        author_of_uri("at://did:plc:t5taxt7lhgyu25oq5nevakqd/app.bsky.feed.repost/3l2dtwcpeen2t")
        == "t5taxt7lhgyu25oq5nevakqd"
    )


def test_author_of_uri_matches_the_author_user_id_column_format():
    """The parsed DID must be directly comparable to author_user_id (no did:plc: prefix).

    This is what licenses using it to attribute a reshare to the original author.
    """
    action = Action.from_dict(
        {
            "action_id": "at://did:plc:n5tgqghvsmedkapljlpsquj7/app.bsky.feed.post/3l6gmxbqhwbj2",
            "created_at": "2024-08-23 00:00:00+00:00",
            "author_user_id": "n5tgqghvsmedkapljlpsquj7",
            "activity_type": "post",
        }
    )
    assert author_of_uri(action.action_id) == action.author_user_id


@pytest.mark.parametrize(
    "uri", [None, "", "at://did:web:genco.me/app.bsky.feed.post/3l2pwj5vwtk2u", "garbage"]
)
def test_author_of_uri_returns_none_when_not_resolvable(uri):
    assert author_of_uri(uri) is None


def test_timestamps_are_ordered_and_span_real_time():
    """A window of actions must span real time, not the duration of the parse.

    Before the fix, first_seen and last_seen collapsed to the wall-clock instants at
    which the rows happened to be processed: a 7-day window measured ~21 seconds wide.
    """
    rows = [
        ("2024-10-03 00:00:00+00:00", "u1"),
        ("2024-10-05 12:00:00.500000+00:00", "u2"),
        ("2024-10-08 00:00:00.649171700+00:00", "u3"),
    ]
    parsed = [parse_timestamp(ts) for ts, _ in rows]
    assert parsed == sorted(parsed)
    span_days = (parsed[-1] - parsed[0]).total_seconds() / 86400
    assert span_days == pytest.approx(5.0, abs=0.01)
