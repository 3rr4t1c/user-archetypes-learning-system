"""Tests for the confidence score.

Two bugs are pinned here:

1. The recency term used datetime.now() as its reference. Once timestamps parse
   correctly (before that fix they were all now(), which hid this), a 2024 window
   scored recency ~ 5e-10 for every user and the confidence >= 0.5 gate rejected
   100% of them. The score also silently depended on the date of the run.

2. Both time terms were normalised by a hard-coded 30 days against a 5-day analysis
   window, squashing recency into [0.85, 1] and lifespan into [0, 0.167]. Neither
   could discriminate between users, so the documented 30% and 20% weights did
   nothing and confidence was really just the volume term.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from arles.arles import Action, ArchetypeLearner

WINDOW_START = datetime(2024, 10, 3, tzinfo=timezone.utc)


def _action(user, when, action_id="a", activity="post"):
    return Action(
        action_id=f"at://did:plc:{user}/app.bsky.feed.post/{action_id}",
        created_at=when,
        author_user_id=user,
        target_user_id=None,
        original_action_id=None,
        activity_type=activity,
        text=None,
    )


def _learner_over_window(users_actions):
    """users_actions: {user: [offsets in hours from WINDOW_START]}"""
    learner = ArchetypeLearner()
    for user, offsets in users_actions.items():
        for i, h in enumerate(offsets):
            learner.process_action(
                _action(user, WINDOW_START + timedelta(hours=h), action_id=f"p{i}")
            )
    return learner


def test_confidence_does_not_depend_on_the_wall_clock():
    """The regression test for bug 1.

    Historical data must not be penalised for being historical: two windows with an
    identical internal structure must score identically, however far apart they sit
    from today. Under the old code the 2024 window scored ~5e-10 on recency and the
    recent one ~1.0, purely because of when the script happened to run.
    """
    pattern = [0, 24, 48, 72, 96, 120]

    old = _learner_over_window({"u1": pattern})
    old_conf = old._compute_confidence_scores(old.next_user_idx)

    recent = ArchetypeLearner()
    base = datetime.now(timezone.utc) - timedelta(days=5)
    for i, h in enumerate(pattern):
        recent.process_action(
            _action("u1", base + timedelta(hours=h), action_id=f"p{i}")
        )
    recent_conf = recent._compute_confidence_scores(recent.next_user_idx)

    assert old_conf[0] == pytest.approx(recent_conf[0], abs=1e-6)

    # And the score must be high on its merits: this user is active throughout the
    # window and right up to its end, so recency and lifespan are both maxed. Volume
    # is the only term short of 1.0 (6 actions against the 2 * min_actions = 20
    # saturation point), which caps the total at ~0.82.
    assert old_conf[0] > 0.8


def test_recency_reference_is_the_window_end_not_now():
    learner = _learner_over_window({"u1": [0, 60, 120]})
    n = learner.next_user_idx
    implicit = learner._compute_confidence_scores(n)
    explicit = learner._compute_confidence_scores(
        n, reference_time=(WINDOW_START + timedelta(hours=120)).timestamp()
    )
    assert implicit == pytest.approx(explicit)


def test_recency_discriminates_within_the_window():
    """The regression test for bug 2, recency half.

    With a hard-coded 30-day normaliser, a user last seen on day 0 and one last seen
    on day 5 scored exp(-5/30)=0.85 vs 1.0 -- a 0.15 spread on a 0.3-weighted term.
    Normalising by the window makes the difference real.
    """
    learner = _learner_over_window(
        {
            "early": [0, 1],  # stops at the very start
            "late": [118, 120],  # active at the window end
        }
    )
    n = learner.next_user_idx
    conf = learner._compute_confidence_scores(n, window_days=5.0)
    early = conf[learner.user_id_to_idx["early"]]
    late = conf[learner.user_id_to_idx["late"]]
    # Same volume, so the whole gap comes from recency.
    assert late - early > 0.2


def test_lifespan_uses_the_full_zero_to_one_range():
    """The regression test for bug 2, lifespan half.

    Against a 30-day normaliser, a user spanning the entire 5-day window scored
    5/30 = 0.167 -- indistinguishable from a user spanning one day (0.033).
    """
    learner = _learner_over_window(
        {
            "burst": [0, 0.1, 0.2, 0.3],  # all activity in 18 minutes
            "spread": [0, 40, 80, 120],  # spread across the window
        }
    )
    n = learner.next_user_idx
    conf = learner._compute_confidence_scores(n, window_days=5.0)
    # Identical volume; 'spread' must win on lifespan.
    assert conf[learner.user_id_to_idx["spread"]] > conf[learner.user_id_to_idx["burst"]]


def test_confidence_stays_in_unit_interval():
    learner = _learner_over_window({f"u{i}": list(range(0, 121, 12)) for i in range(5)})
    conf = learner._compute_confidence_scores(learner.next_user_idx)
    assert np.all(conf >= 0.0) and np.all(conf <= 1.0)


def test_volume_still_dominates_as_documented():
    """Volume carries 50%: a prolific user must outrank a one-action user."""
    learner = _learner_over_window({"heavy": list(range(0, 121, 6)), "light": [120]})
    n = learner.next_user_idx
    conf = learner._compute_confidence_scores(n, window_days=5.0)
    assert conf[learner.user_id_to_idx["heavy"]] > conf[learner.user_id_to_idx["light"]]


def test_zero_users_returns_empty():
    assert ArchetypeLearner()._compute_confidence_scores(0).shape == (0,)


def test_single_instant_window_does_not_divide_by_zero():
    learner = _learner_over_window({"u1": [0], "u2": [0]})
    conf = learner._compute_confidence_scores(learner.next_user_idx)
    assert np.all(np.isfinite(conf))
