"""Tests for the confidence score.

Confidence answers "how much evidence supports this user's vector", so its volume term
counts the repost events the user is involved in -- as resharer or as the author being
reshared -- and not their total activity. A user with 500 replies and one repost has
almost nothing behind their archetype, and confidence must say so. That is the
content-agnostic form of the target variable in Verdolotti et al.

Two bugs are pinned here:

1. The recency term used datetime.now() as its reference. Once timestamps parse
   correctly (before that fix they were all now(), which hid this), a 2024 window
   scored recency ~5e-10 for every user and a confidence >= 0.5 gate rejected 100% of
   them. The score also silently depended on the date of the run.

2. Both time terms were normalised by a hard-coded 30 days against a 5-day analysis
   window, squashing recency into [0.85, 1] and lifespan into [0, 0.167]. Neither could
   discriminate between users, so the documented 30% and 20% weights did nothing and
   confidence was really just the volume term.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from arles.features import FeatureExtractor, WindowIndex, confidence_scores
from arles.schema import CanonicalAction

START = datetime(2024, 10, 12, tzinfo=timezone.utc)
END = START + timedelta(days=5)


def repost(actor, parent, parent_actor, minutes):
    return CanonicalAction(
        action_id=f"r-{actor}-{parent}-{minutes}",
        actor_id=actor,
        activity_type="repost",
        created_at=START + timedelta(minutes=minutes),
        parent_id=parent,
        parent_actor_id=parent_actor,
    )


def build(actions, start=START, end=END):
    index = WindowIndex.build(actions)
    fx = FeatureExtractor(index, start, end)
    for a in actions:
        fx.add(a)
    return fx


def conf_of(fx, user):
    ids, c = fx.confidence()
    return c[ids.index(user)]


# ------------------------------------------------------------------ wiring


def test_extractor_exposes_confidence_aligned_with_the_feature_matrix():
    """It must actually be reachable: the first version of this function was written
    and never called, so the pipeline shipped with no confidence output at all."""
    fx = build([repost(f"u{i}", "p1", "alice", i) for i in range(5)])
    ids_x, X = fx.finish()
    ids_c, c = fx.confidence()
    assert ids_x == ids_c
    assert c.shape[0] == X.shape[0]
    assert np.all((c >= 0.0) & (c <= 1.0))


def test_volume_counts_both_sides_of_a_reshare():
    """The author being reshared has evidence too -- it is what the vector is built
    from. Verdolotti et al.'s target counts both sides for the same reason."""
    fx = build([repost(f"u{i}", "p1", "alice", i) for i in range(20)])
    # alice performs no reposts at all, but is reshared 20 times.
    assert conf_of(fx, "alice") > conf_of(fx, "u0")


# ------------------------------------------------------------------ bug 1


def test_confidence_does_not_depend_on_the_wall_clock():
    """Historical data must not be penalised for being historical.

    Two windows with identical internal structure must score identically however far
    apart they sit from today. Under the old code the 2024 window scored ~5e-10 on
    recency and a recent one ~1.0, purely because of when the script ran.
    """
    pattern = [0, 24 * 60, 48 * 60, 72 * 60, 96 * 60, 120 * 60]
    old = build([repost("u1", f"p{i}", "alice", m) for i, m in enumerate(pattern)])
    old_ids, old_conf = old.confidence()

    base = datetime.now(timezone.utc) - timedelta(days=5)
    recent_actions = [
        CanonicalAction(f"r{i}", "u1", "repost", base + timedelta(minutes=m),
                        f"p{i}", "alice")
        for i, m in enumerate(pattern)
    ]
    recent = build(recent_actions, start=base, end=base + timedelta(days=5))
    recent_ids, recent_conf = recent.confidence()

    assert old_conf[old_ids.index("u1")] == pytest.approx(
        recent_conf[recent_ids.index("u1")], abs=1e-6
    )


# ------------------------------------------------------------------ bug 2


def test_recency_discriminates_within_the_window():
    """With a hard-coded 30-day normaliser a user last seen on day 0 and one last seen
    on day 5 scored exp(-5/30)=0.85 vs 1.0 -- a 0.15 spread on a 0.3-weighted term."""
    actions = [
        repost("early", "p1", "a1", 0), repost("early", "p2", "a2", 1),
        repost("late", "p3", "a3", 7198), repost("late", "p4", "a4", 7199),
    ]
    fx = build(actions)
    assert conf_of(fx, "late") - conf_of(fx, "early") > 0.2


def test_lifespan_uses_the_full_zero_to_one_range():
    """Against a 30-day normaliser a user spanning the whole 5-day window scored
    5/30 = 0.167, indistinguishable from one spanning a day (0.033)."""
    actions = [repost("burst", f"b{i}", f"a{i}", i * 0.1) for i in range(4)]
    actions += [repost("spread", f"s{i}", f"a{i}", i * 2400) for i in range(4)]
    fx = build(actions)
    assert conf_of(fx, "spread") > conf_of(fx, "burst")


# ------------------------------------------------------------------ shape


def test_volume_still_dominates_as_documented():
    actions = [repost("heavy", f"p{i}", f"a{i}", i * 60) for i in range(40)]
    actions += [repost("light", "px", "ax", 7000)]
    fx = build(actions)
    assert conf_of(fx, "heavy") > conf_of(fx, "light")


def test_confidence_stays_in_the_unit_interval():
    actions = [repost(f"u{i%5}", f"p{i}", f"a{i%3}", i * 30) for i in range(200)]
    _, c = build(actions).confidence()
    assert np.all(c >= 0.0) and np.all(c <= 1.0)


def test_empty_window_returns_empty():
    ids, c = build([]).confidence()
    assert ids == [] and c.shape == (0,)


def test_single_instant_window_does_not_divide_by_zero():
    actions = [repost("u1", "p1", "a", 0), repost("u2", "p1", "a", 0)]
    fx = build(actions, start=START, end=START)
    _, c = fx.confidence()
    assert np.all(np.isfinite(c))


def test_confidence_scores_is_pure_and_vectorised():
    conf = confidence_scores(
        involvement=np.array([0.0, 1.0, 20.0, 1000.0]),
        first_seen=np.full(4, START.timestamp()),
        last_seen=np.full(4, END.timestamp()),
        window_start=START,
        window_end=END,
    )
    assert conf.shape == (4,)
    assert np.all(np.diff(conf) >= 0)  # monotone in volume
    assert np.all((conf >= 0) & (conf <= 1))
