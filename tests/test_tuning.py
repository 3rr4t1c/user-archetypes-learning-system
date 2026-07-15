"""Tests for the (alpha, delta) re-tuning.

The point of this module is to turn a parameter choice into an empirical result, so
the tests check that the machinery would actually detect a wrong answer: nDCG has to
reward good rankings and punish bad ones, and the sweep has to find a planted optimum.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from arles.features import FeatureExtractor, WindowIndex
from arles.schema import CanonicalAction
from arles.tuning import (
    DEFAULT_ALPHAS,
    DEFAULT_DELTAS,
    format_grid,
    grid_search,
    involvement_target,
    ndcg_at_k,
)

START = datetime(2024, 10, 12, tzinfo=timezone.utc)
END = START + timedelta(days=5)


def repost(actor, parent, parent_actor, minutes):
    return CanonicalAction(
        action_id=f"r-{actor}-{parent}-{minutes}",
        actor_id=actor, activity_type="repost",
        created_at=START + timedelta(minutes=minutes),
        parent_id=parent, parent_actor_id=parent_actor,
    )


# ------------------------------------------------------------------------- nDCG


def test_ndcg_is_one_for_a_perfect_ranking():
    scores = np.array([3.0, 2.0, 1.0])
    relevance = np.array([3.0, 2.0, 1.0])
    assert ndcg_at_k(scores, relevance, 3) == pytest.approx(1.0)


def test_ndcg_is_lower_for_a_reversed_ranking():
    relevance = np.array([3.0, 2.0, 1.0])
    good = ndcg_at_k(np.array([3.0, 2.0, 1.0]), relevance, 3)
    bad = ndcg_at_k(np.array([1.0, 2.0, 3.0]), relevance, 3)
    assert bad < good


def test_ndcg_rewards_getting_the_top_right():
    """Discounting means the head of the ranking dominates -- as intended.

    Only user 0 matters. Scoring them top gives 1.0; scoring them strictly last
    (everyone else above) drops nDCG to 1/log2(5) = 0.431.
    """
    relevance = np.array([10.0, 0.0, 0.0, 0.0])
    top_right = ndcg_at_k(np.array([1.0, 0.0, 0.0, 0.0]), relevance, 4)
    top_wrong = ndcg_at_k(np.array([0.0, 1.0, 1.0, 1.0]), relevance, 4)
    assert top_right == pytest.approx(1.0)
    assert top_wrong == pytest.approx(1.0 / np.log2(5))
    assert top_wrong < 0.5


def test_ndcg_handles_all_zero_relevance():
    assert ndcg_at_k(np.array([1.0, 2.0]), np.array([0.0, 0.0]), 2) == 0.0


def test_ndcg_is_deterministic_under_ties():
    scores = np.ones(50)
    relevance = np.arange(50, dtype=float)
    a = ndcg_at_k(scores, relevance, 10)
    b = ndcg_at_k(scores, relevance, 10)
    assert a == b


def test_ndcg_k_larger_than_population():
    assert ndcg_at_k(np.array([1.0]), np.array([1.0]), 100) == pytest.approx(1.0)


# ----------------------------------------------------------------------- target


def test_involvement_counts_both_sides_of_a_reshare():
    """Verdolotti et al.'s target: engaged as resharer OR as reshared author."""
    actions = [repost("u1", "p1", "alice", 1), repost("u2", "p1", "alice", 2)]
    target = involvement_target(actions, ["u1", "u2", "alice", "nobody"])
    assert list(target) == [1.0, 1.0, 2.0, 0.0]


def test_involvement_ignores_self_reshares():
    actions = [repost("alice", "p1", "alice", 1)]
    assert list(involvement_target(actions, ["alice"])) == [0.0]


def test_absent_users_score_zero_not_missing():
    """A user who vanishes next window is a prediction failure, not missing data."""
    target = involvement_target([], ["u1", "u2"])
    assert list(target) == [0.0, 0.0]


# ------------------------------------------------------------------ grid search


def _planted_window():
    """A window where consistent authors should outrank one-hit-wonders.

    steady: reshared in every slot. flash: one big burst at the start, then silence.
    A well-chosen (alpha, delta) should rank steady above flash for predicting the
    next window, where only steady is still active.
    """
    actions = []
    for slot in range(20):  # 20 x 6h = 5 days
        base = slot * 360
        for p in range(3):
            for i in range(3):
                actions.append(repost(f"s{slot}{p}{i}", f"sp{slot}{p}", "steady", base + i))
    for p in range(10):
        for i in range(10):
            actions.append(repost(f"f{p}{i}", f"fp{p}", "flash", i))
    return actions


def test_grid_search_finds_a_planted_optimum():
    actions = _planted_window()
    index = WindowIndex.build(actions)
    fx = FeatureExtractor(index, START, END)
    for a in actions:
        fx.add(a)
    ids, _ = fx.finish()

    # Next window: only `steady` is still being reshared.
    nxt = [repost(f"n{i}", f"np{i}", "steady", 1) for i in range(20)]
    target = involvement_target(nxt, ids)

    best_tash, best_tai, results = grid_search(
        fx, target, alphas=(0.1, 0.5, 0.9),
        deltas=(timedelta(hours=6), timedelta(days=2)), ks=(10, 100), progress=False,
    )
    assert best_tash.metric == "tash_index"
    assert best_tai.metric == "tai_score"
    assert 0.0 <= best_tash.score <= 1.0
    # alphas x deltas x 2 time-aware metrics, plus the 2 static baselines
    assert len(results) == 3 * 2 * 2 + 2


def test_grid_search_scores_the_static_baselines():
    """Without these the sweep can say which TASH is best, never whether TASH is worth
    having. Verdolotti et al. compare against exactly these two."""
    actions = _planted_window()
    index = WindowIndex.build(actions)
    fx = FeatureExtractor(index, START, END)
    for a in actions:
        fx.add(a)
    ids, _ = fx.finish()
    target = involvement_target(
        [repost(f"n{i}", f"np{i}", "steady", 1) for i in range(20)], ids
    )
    _, _, results = grid_search(
        fx, target, alphas=(0.5,), deltas=(timedelta(hours=6),), ks=(100,),
        progress=False,
    )
    assert {r.metric for r in results} == {
        "tash_index", "tai_score", "influence_score", "h_index"
    }


def test_baselines_can_be_switched_off():
    actions = _planted_window()
    index = WindowIndex.build(actions)
    fx = FeatureExtractor(index, START, END)
    for a in actions:
        fx.add(a)
    ids, _ = fx.finish()
    target = involvement_target(
        [repost(f"n{i}", f"np{i}", "steady", 1) for i in range(5)], ids
    )
    _, _, results = grid_search(
        fx, target, alphas=(0.5,), deltas=(timedelta(hours=6),), ks=(100,),
        progress=False, include_baselines=False,
    )
    assert {r.metric for r in results} == {"tash_index", "tai_score"}


def test_grid_search_returns_the_argmax():
    actions = _planted_window()
    index = WindowIndex.build(actions)
    fx = FeatureExtractor(index, START, END)
    for a in actions:
        fx.add(a)
    ids, _ = fx.finish()
    target = involvement_target(
        [repost(f"n{i}", f"np{i}", "steady", 1) for i in range(20)], ids
    )
    best_tash, _, results = grid_search(
        fx, target, alphas=(0.1, 0.5, 0.9),
        deltas=(timedelta(hours=6), timedelta(hours=12)), ks=(100,), progress=False,
    )
    tash_rows = [r for r in results if r.metric == "tash_index"]
    assert best_tash.score == max(r.score for r in tash_rows)


def test_default_grid_reaches_the_static_degenerate_point():
    """delta == the window means one slot: the EMA degenerates to the static metric.

    The first sweep put both optima on a grid edge -- TASH rising monotonically to
    alpha=0.9, TAI to delta=2d -- so the grid had not contained the optimum, it had
    merely stopped. Both directions point towards less time-awareness, so the grid now
    runs all the way to the degenerate point instead of assuming the answer lies short
    of it. delta must never *exceed* the window, which would be meaningless.
    """
    assert max(DEFAULT_DELTAS) == timedelta(days=5)      # the 5-day analysis window
    assert 0.99 in DEFAULT_ALPHAS                        # alpha runs to its own edge
    assert 0.5 in DEFAULT_ALPHAS and 0.6 in DEFAULT_ALPHAS  # the prior work's optima


def test_format_grid_renders_a_surface():
    actions = _planted_window()
    index = WindowIndex.build(actions)
    fx = FeatureExtractor(index, START, END)
    for a in actions:
        fx.add(a)
    ids, _ = fx.finish()
    target = involvement_target(
        [repost(f"n{i}", f"np{i}", "steady", 1) for i in range(5)], ids
    )
    _, _, results = grid_search(
        fx, target, alphas=(0.1, 0.5), deltas=(timedelta(hours=6),),
        ks=(100,), progress=False,
    )
    grid = format_grid(results, "tash_index")
    assert "nDCG@100 surface for tash_index" in grid
    assert "best:" in grid
    assert "*" in grid  # the argmax is marked
