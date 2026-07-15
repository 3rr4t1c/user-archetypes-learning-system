"""Tests for the twelve archetype features.

Each feature is checked against a hand-built scenario where the right answer is
obvious, because "the number came out plausible" is exactly how this codebase's
previous metrics hid the fact that three of them were identically zero.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from arles.features import (
    AMPLIFIER_FEATURES,
    COORDINATED_FEATURES,
    FEATURE_NAMES,
    SUPERSPREADER_FEATURES,
    FeatureExtractor,
    WindowIndex,
    h_index,
)
from arles.schema import CanonicalAction

START = datetime(2024, 10, 12, tzinfo=timezone.utc)
END = START + timedelta(days=5)


def repost(actor, parent, parent_actor, minutes, action_id=None):
    return CanonicalAction(
        action_id=action_id or f"r-{actor}-{parent}-{minutes}",
        actor_id=actor,
        activity_type="repost",
        created_at=START + timedelta(minutes=minutes),
        parent_id=parent,
        parent_actor_id=parent_actor,
    )


def post(actor, minutes, action_id):
    return CanonicalAction(
        action_id=action_id, actor_id=actor, activity_type="post",
        created_at=START + timedelta(minutes=minutes),
    )


def extract(actions, **kwargs):
    index = WindowIndex.build(actions)
    fx = FeatureExtractor(index, START, END, **kwargs)
    for a in actions:
        fx.add(a)
    ids, X = fx.finish()
    return ids, X, {name: X[:, i] for i, name in enumerate(FEATURE_NAMES)}


def col(ids, values, user):
    return values[ids.index(user)]


# --------------------------------------------------------------------- structure


def test_twelve_features_four_per_archetype():
    assert len(SUPERSPREADER_FEATURES) == 4
    assert len(AMPLIFIER_FEATURES) == 4
    assert len(COORDINATED_FEATURES) == 4
    assert len(FEATURE_NAMES) == 12
    assert len(set(FEATURE_NAMES)) == 12


def test_only_reposts_are_consumed():
    """Posts and replies must not change any feature.

    This is the claim that lets ArLeS run on any platform exporting reshares alone.
    """
    reposts = [repost("u1", "p1", "author", m) for m in range(5)]
    noise = [post("u1", 1, "px"), post("author", 2, "p1")]
    _, X_plain, _ = extract(reposts)
    _, X_noisy, _ = extract(reposts + noise)
    assert np.allclose(X_plain, X_noisy)


def test_self_reshares_are_excluded():
    """Per Verdolotti et al., a user resharing themselves is not diffusion."""
    actions = [repost("author", "p1", "author", 1), repost("u2", "p1", "author", 2)]
    ids, X, f = extract(actions)
    assert col(ids, f["influence_score"], "author") == 1.0  # only u2's counted


def test_unattributed_reposts_are_counted_not_silently_dropped():
    index = WindowIndex.build([
        CanonicalAction("r1", "u1", "repost", START, "p1", None),
        repost("u2", "p2", "a", 1),
    ])
    assert index.n_unattributed == 1
    assert index.n_reposts == 1


# ---------------------------------------------------------------- super-spreader


def test_influence_score_counts_reshares_received():
    actions = [repost(f"u{i}", "p1", "alice", i) for i in range(7)]
    actions += [repost("u9", "p2", "bob", 10)]
    ids, X, f = extract(actions)
    assert col(ids, f["influence_score"], "alice") == 7
    assert col(ids, f["influence_score"], "bob") == 1


def test_h_index_needs_multiple_posts_with_multiple_reshares():
    # alice: 3 posts with 3, 3, 3 reshares -> h = 3
    actions = []
    for p in ("p1", "p2", "p3"):
        for i in range(3):
            actions.append(repost(f"u{p}{i}", p, "alice", len(actions)))
    # bob: 1 post with 50 reshares -> h = 1, despite far more influence
    for i in range(50):
        actions.append(repost(f"v{i}", "q1", "bob", len(actions)))
    ids, X, f = extract(actions)
    assert col(ids, f["h_index"], "alice") == 3
    assert col(ids, f["h_index"], "bob") == 1
    assert col(ids, f["influence_score"], "bob") == 50
    # The two say different things -- which is why both are in the bucket.


@pytest.mark.parametrize("counts,expected", [
    ([], 0), ([1], 1), ([5, 4, 3, 2, 1], 3), ([10, 8, 5, 4, 3], 4), ([1000], 1),
])
def test_h_index_helper(counts, expected):
    assert h_index(counts) == expected


def test_tash_decays_when_an_author_goes_quiet():
    """The regression for a real bug: only authors active in a slot were updated,

    so a single early burst persisted undecayed to the end of the window. The
    definition is TASH_t = alpha*TASH_{t-1} + (1-alpha)*H_t with H_t = 0 when idle.
    """
    slot = timedelta(hours=6)
    # alice: bursts in slot 0 only. bob: same burst, but keeps going every slot.
    actions = []
    for p in ("p1", "p2"):
        for i in range(3):
            actions.append(repost(f"a{p}{i}", p, "alice", 10 + i))
            actions.append(repost(f"b{p}{i}", f"q{p}", "bob", 10 + i))
    for s in range(1, 8):
        for p in ("q1", "q2"):
            for i in range(3):
                actions.append(repost(f"c{s}{p}{i}", p, "bob", s * 360 + 10 + i))
    ids, X, f = extract(actions, slot=slot)
    assert col(ids, f["tash_index"], "alice") < col(ids, f["tash_index"], "bob")
    # alice's burst must have decayed towards zero, not been frozen at its peak.
    assert col(ids, f["tash_index"], "alice") < 0.5


def test_tai_and_tash_are_zero_without_reshares():
    ids, X, f = extract([repost("u1", "p1", "alice", 1)])
    assert col(ids, f["tai_score"], "u1") == 0.0   # u1 authored nothing
    assert col(ids, f["tash_index"], "u1") == 0.0


# --------------------------------------------------------------------- amplifier


def test_repost_count_is_a_count_not_a_saturating_ema():
    """The old metric was 1-0.9^k: 0.995 at 50 reposts, 0.99997 at 100 -- flat."""
    actions = [repost("heavy", f"p{i}", "a", i) for i in range(200)]
    actions += [repost("light", "px", "a", 500)]
    ids, X, f = extract(actions)
    assert col(ids, f["repost_count"], "heavy") == 200
    assert col(ids, f["repost_count"], "light") == 1


def test_repost_rate_is_per_day_over_the_window():
    actions = [repost("u1", f"p{i}", "a", i * 60) for i in range(10)]
    ids, X, f = extract(actions)
    assert col(ids, f["repost_rate"], "u1") == pytest.approx(10 / 5.0)  # 5-day window


def test_ear_index_rewards_resharing_early_into_a_large_cascade():
    """EaR = (1/|P_u|) * sum(N_p - r + 1), exactly as in Verdolotti et al."""
    # One post, 10 reshares. early is rank 1, late is rank 10.
    actions = [repost("early", "p1", "alice", 0)]
    actions += [repost(f"mid{i}", "p1", "alice", i + 1) for i in range(8)]
    actions.append(repost("late", "p1", "alice", 20))
    ids, X, f = extract(actions)
    # N_p = 10: early scores 10-1+1 = 10, late scores 10-10+1 = 1.
    assert col(ids, f["ear_index"], "early") == pytest.approx(10.0)
    assert col(ids, f["ear_index"], "late") == pytest.approx(1.0)


def test_ear_index_needs_the_final_cascade_size_not_the_running_one():
    """The reason pass 1 exists.

    Rank 1 of a cascade that ends at 10 must outscore rank 1 of a cascade that ends
    at 2. A single streaming pass cannot know either N_p at rank 1.
    """
    big = [repost("first_big", "p1", "alice", 0)]
    big += [repost(f"o{i}", "p1", "alice", i + 1) for i in range(9)]
    small = [repost("first_small", "p2", "bob", 0), repost("o_s", "p2", "bob", 1)]
    ids, X, f = extract(big + small)
    assert col(ids, f["ear_index"], "first_big") == pytest.approx(10.0)
    assert col(ids, f["ear_index"], "first_small") == pytest.approx(2.0)


def test_amplification_breadth_counts_distinct_authors():
    # focused: 20 reposts, all from one author. broad: 5 reposts, 5 authors.
    actions = [repost("focused", f"p{i}", "alice", i) for i in range(20)]
    actions += [repost("broad", f"q{i}", f"author{i}", 100 + i) for i in range(5)]
    ids, X, f = extract(actions)
    assert col(ids, f["amplification_breadth"], "focused") == 1
    assert col(ids, f["amplification_breadth"], "broad") == 5


# ------------------------------------------------------------------- coordinated


def test_co_action_detects_users_hitting_the_same_content_together():
    # swarm: three users reshare p1 within seconds. loner: reshares p2 alone.
    actions = [
        repost("s1", "p1", "alice", 0),
        repost("s2", "p1", "alice", 0.1),
        repost("s3", "p1", "alice", 0.2),
        repost("loner", "p2", "bob", 60),
    ]
    ids, X, f = extract(actions)
    assert col(ids, f["co_action_rate"], "s3") == 1.0
    assert col(ids, f["co_action_size"], "s3") == 2.0  # s1 and s2
    assert col(ids, f["co_action_rate"], "loner") == 0.0
    assert col(ids, f["co_action_size"], "loner") == 0.0


def test_co_action_ignores_peers_outside_the_window():
    actions = [repost("a", "p1", "x", 0), repost("b", "p1", "x", 60)]  # 60 min apart
    ids, X, f = extract(actions)
    assert col(ids, f["co_action_rate"], "b") == 0.0


def test_co_action_latency_is_higher_for_tighter_synchrony():
    tight = [repost("t1", "p1", "x", 0), repost("t2", "p1", "x", 0.05)]   # 3s
    loose = [repost("l1", "p2", "y", 0), repost("l2", "p2", "y", 4.0)]    # 240s
    ids, X, f = extract(tight + loose)
    assert col(ids, f["co_action_latency"], "t2") > col(ids, f["co_action_latency"], "l2")


def test_niche_co_action_ignores_viral_pile_ons():
    """The feature that carries the archetype.

    Resharing a viral post puts you next to hundreds of strangers -- popularity, not
    coordination. Only swarming on content that stayed obscure counts.
    """
    # viral post: 60 resharers (above the default threshold of 50), all within Δt
    viral = [repost(f"v{i}", "viral", "alice", i * 0.01) for i in range(60)]
    # niche post: 3 resharers, tightly grouped
    niche = [repost(f"n{i}", "niche", "bob", 200 + i * 0.01) for i in range(3)]
    ids, X, f = extract(viral + niche)
    # a viral resharer has many co-actors but no niche co-action
    assert col(ids, f["co_action_size"], "v59") > 50
    assert col(ids, f["niche_co_action"], "v59") == 0.0
    # a niche swarmer has few co-actors but they all count
    assert col(ids, f["niche_co_action"], "n2") == 2.0


def test_features_survive_out_of_order_input():
    """The archive is ingestion-ordered; created_at deviates by up to ~18.6 h.

    finish() sorts, so shuffling the input must not change a single feature.
    """
    actions = [repost(f"u{i%5}", f"p{i%3}", "alice", i * 0.02) for i in range(60)]
    ids_a, X_a, _ = extract(actions)
    shuffled = actions[30:] + actions[:30]  # a gross reordering
    ids_b, X_b, _ = extract(shuffled)
    reorder = [ids_b.index(u) for u in ids_a]
    assert np.allclose(X_a, X_b[reorder])


def test_empty_window_returns_empty():
    ids, X, _ = extract([])
    assert ids == []
    assert X.shape == (0, 12)


def test_no_nans_or_infs_ever_reach_the_matrix():
    actions = [repost("u1", "p1", "alice", 0)]
    _, X, _ = extract(actions)
    assert np.all(np.isfinite(X))


def test_buffer_roundtrips_through_the_cache(tmp_path):
    """Caching must be exact: a cached window has to score identically to a fresh read.

    This is what makes re-scoring a new grid a minutes-long job instead of the 4.5-hour
    archive re-read the first sweep cost.
    """
    actions = [repost(f"u{i%7}", f"p{i%4}", f"a{i%3}", i * 0.5) for i in range(120)]
    index = WindowIndex.build(actions)

    fx = FeatureExtractor(index, START, END)
    for a in actions:
        fx.add(a)
    ids_a, X_a = fx.finish()

    path = str(tmp_path / "win.npz")
    fx.save_buffer(path)

    fx2 = FeatureExtractor.load_buffer(path, index)
    ids_b, X_b = fx2.finish()

    assert ids_a == ids_b
    assert np.allclose(X_a, X_b)
    assert fx2.window_start == fx.window_start
    assert fx2.window_end == fx.window_end
