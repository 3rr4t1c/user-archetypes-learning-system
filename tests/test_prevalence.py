"""Tests for the head statistics.

The module exists because of one measured fact: on the full archive
`median_superspreader` is 0.0000 in every window and the mean stays inside
[0.021, 0.037]. A statistic that cannot move cannot support a claim, and the
super-spreader line in the published figure is that statistic.

The tests below pin the properties that make the replacements trustworthy -- above all
that a pooled bar can move and a per-window bar cannot, which is the whole argument.
"""

import numpy as np
import pytest

from arles.features import FEATURE_NAMES
from arles.prevalence import (
    AXIS_ANCHOR,
    DEFAULT_QUANTILE,
    fold_change,
    gini,
    head_stats,
    pooled_threshold,
    sweep_quantiles,
    top_share,
)


def population(n, n_heads, seed=0, head=0.9):
    """A window shaped like a real one: a mass of ~zeros and a tiny head."""
    rng = np.random.default_rng(seed)
    s = np.abs(rng.normal(0, 0.02, n))
    s[:n_heads] = head + rng.uniform(0, 0.1, n_heads)
    return np.clip(s, 0, 1)


# ------------------------------------------------------- the reason this module exists


def test_the_mean_cannot_see_a_change_the_prevalence_can():
    """The user's example, made concrete.

    10 super-spreaders in 600k accounts become 15. That is +50% of the phenomenon. The
    mean moves by ~1e-5 -- less than the noise from the population changing at all --
    while prevalence reports the 1.5x it is.
    """
    pre = population(600_000, 10, seed=1)
    post = population(600_000, 15, seed=2)

    assert abs(post.mean() - pre.mean()) < 1e-4, "the mean is blind here, by construction"

    bar = 0.5  # anywhere between the bulk and the head; see the sweep tests below
    h_pre = head_stats(pre, bar, "superspreader")
    h_post = head_stats(post, bar, "superspreader")
    assert (h_pre.count, h_post.count) == (10, 15)
    assert fold_change(h_pre.rate_per_100k, h_post.rate_per_100k) == pytest.approx(1.5)


def test_a_per_axis_quantile_erases_the_scale_difference_between_archetypes():
    """The bug this module shipped with, pinned with the numbers that exposed it.

    The three archetypes are rare at three different scales: amplifiers outnumber
    super-spreaders by an order of magnitude, coordinated accounts sit in between. Take
    the top q of *each axis separately* and every axis has (1-q) of the population above
    its bar -- by construction, whatever the data says. Measured on E1's real pre-event
    window at q=0.999: 194 super-spreaders, 198 amplifiers, 304 coordinated. Three
    numbers that look like a measurement and are arithmetic.

    A common bar leaves the counts free to differ. Same window at theta=0.5:
    538 / 3,137 / 616.
    """
    rng = np.random.default_rng(11)
    n = 200_000
    # Three axes of genuinely different rarity, as the archetypes are. Beta rather than
    # a clipped normal: clipping piles mass on 1.0, and the ties make a quantile bar
    # catch far more than (1-q) of the population, which would mask the very effect
    # under test.
    rare = rng.beta(0.6, 14, n)     # super-spreader-like: a handful reach high
    common = rng.beta(2.0, 2.5, n)  # amplifier-like: ordinary behaviour, done well
    middle = rng.beta(1.0, 6, n)    # coordinated-like: in between

    # Per-axis quantile: identical counts, and the scale difference is gone.
    per_axis = [head_stats(c, pooled_threshold([c], 0.999), "x").count
                for c in (rare, common, middle)]
    assert max(per_axis) - min(per_axis) <= 1, (
        "a per-axis quantile reports three archetypes of wildly different prevalence as "
        "equally common -- this is the statistic, not the data"
    )

    # One common bar: the ordering survives into the measurement. 57 / 111,618 / 9,119.
    at = [head_stats(c, 0.4, "x").count for c in (rare, common, middle)]
    assert at[1] > 10 * at[2] > 10 * at[0], (
        "amplifier >> coordinated >> super-spreader, as the archetypes are defined"
    )
    assert all(c > 0 for c in at), "and all three are still measurable at this bar"


def test_the_common_bar_is_the_default_and_the_quantile_is_not():
    from arles.prevalence import COMMON_BAR, SWEEP_BARS
    assert COMMON_BAR == 0.5
    assert SWEEP_BARS == (0.3, 0.4, 0.5, 0.6, 0.7)
    assert all(0.0 < b < 1.0 for b in SWEEP_BARS)


def test_fold_change_is_how_archetypes_of_different_scale_get_compared():
    """15 super-spreaders and 15,000 amplifiers can be the same 1.5x. The raw counts are
    not comparable across archetypes and never will be; the fold change against each
    archetype's own baseline is."""
    assert fold_change(10.0, 15.0) == pytest.approx(fold_change(10_000.0, 15_000.0))


def test_the_bar_must_be_finer_than_the_head_it_is_meant_to_isolate():
    """The trap in DEFAULT_QUANTILE, pinned so nobody trips over it silently.

    A quantile bar selects a *fixed share* of the population, so on a 600k-user window
    the top 0.1% is 600 accounts. That is a perfectly good operational definition of
    "amplifier", and a bad one of "super-spreader", which the archetype's own definition
    says is a handful. Ten accounts in 600k is the top 0.0017%, i.e. q = 0.999983.

    The consequence: `count` is a statement about q at least as much as about the data,
    so no single q can be reported as if it were a measurement. What survives is the
    *shape* across q -- hence `sweep_quantiles`.
    """
    pre = population(600_000, 10, seed=1)
    post = population(600_000, 15, seed=2)

    loose = pooled_threshold([pre, post], 0.999)
    assert head_stats(pre, loose, "superspreader").count > 500, (
        "the top 0.1% of 600k is ~600 accounts -- it reaches far past the 10 real heads "
        "and fills up with ordinary ones"
    )

    tight = pooled_threshold([pre, post], 1 - 25 / 1_200_000)
    assert head_stats(pre, tight, "superspreader").count == pytest.approx(10, abs=2)


def test_a_per_window_bar_is_flat_by_construction():
    """Why the bar must be pooled and frozen.

    Each window's own top 0.1% is 0.1% of that window whatever happened -- the figure
    would draw a flat line through any data at all, including data with a real change.
    """
    pre = population(200_000, 10, seed=1)
    post = population(200_000, 400, seed=2)  # a 40x change in the phenomenon

    per_window = [
        (w >= np.quantile(w, 0.999)).mean() * 1e5 for w in (pre, post)
    ]
    assert per_window[0] == pytest.approx(per_window[1], rel=0.02), "flat, as warned"

    bar = pooled_threshold([pre, post], 0.999)
    pooled = [head_stats(w, bar, "superspreader").rate_per_100k for w in (pre, post)]
    assert pooled[1] > pooled[0] * 5, "the pooled bar sees it"


def test_count_and_rate_can_disagree_and_both_are_reported():
    """E1 tripled the population in one window. A head count that triples means the
    rate did not move; showing only the count would call that growth."""
    small = population(200_000, 10, seed=1)
    big = population(600_000, 30, seed=2)  # 3x the users, 3x the heads: no change in rate

    bar = 0.5
    a, b = head_stats(small, bar, "superspreader"), head_stats(big, bar, "superspreader")

    assert b.count == 3 * a.count
    assert b.rate_per_100k == pytest.approx(a.rate_per_100k, rel=0.01)


# ------------------------------------------------------------------ pooled threshold


def test_threshold_is_one_number_for_every_window():
    cols = [population(10_000, 5, seed=s) for s in range(4)]
    bar = pooled_threshold(cols, 0.999)
    assert np.isfinite(bar)
    # Each window is scored against the same bar, so the counts are comparable.
    counts = [head_stats(c, bar, "superspreader").count for c in cols]
    assert sum(counts) > 0


def test_threshold_is_the_quantile_of_the_concatenation_not_of_the_means():
    a = np.zeros(1000)
    b = np.ones(1000)
    # Pooled: half zeros, half ones -> the 0.999 quantile is 1.0, not 0.5.
    assert pooled_threshold([a, b], 0.999) == pytest.approx(1.0)


def test_bigger_windows_carry_more_weight_in_the_bar():
    """A property worth knowing rather than a bug: pooling concatenates users, so the
    bar is the top q of *user-windows*, not the average of per-window bars."""
    small = np.full(10, 1.0)
    big = np.zeros(10_000)
    assert pooled_threshold([small, big], 0.99) == pytest.approx(0.0)


def test_empty_input_is_nan_not_a_crash():
    assert np.isnan(pooled_threshold([]))
    assert np.isnan(pooled_threshold([np.zeros(0)]))


def test_quantile_must_be_a_quantile():
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError, match="q must be"):
            pooled_threshold([np.zeros(10)], bad)


# ------------------------------------------------------------------------ head stats


def test_head_stats_reports_the_shape_a_mean_hides():
    s = population(100_000, 50, seed=3)
    bar = pooled_threshold([s], 0.999)
    h = head_stats(s, bar, "superspreader")

    assert h.n_users == 100_000
    assert h.count == pytest.approx(100, abs=60)  # ~0.1% by construction
    assert h.rate_per_100k == pytest.approx(h.count / h.n_users * 1e5)
    assert h.p999 >= h.p99
    assert h.pmax >= h.p999
    assert 0.0 <= h.gini <= 1.0
    assert 0.0 <= h.top1pct_share <= 1.0


def test_the_bar_is_quoted_in_a_unit_a_reader_can_check():
    """"score >= 0.87" means nothing to anyone. "a median of N reshares" does."""
    n = 1000
    s = population(n, 10, seed=4)
    X = np.zeros((n, len(FEATURE_NAMES)))
    X[:, FEATURE_NAMES.index("influence_score")] = 3.0
    X[:10, FEATURE_NAMES.index("influence_score")] = 50_000.0  # the head

    bar = pooled_threshold([s], 0.99)
    h = head_stats(s, bar, "superspreader", X=X)
    assert h.anchor_feature == "influence_score"
    assert h.anchor_median == pytest.approx(50_000.0)


def test_every_axis_has_an_anchor():
    from arles.embedding import AXES
    for axis in AXES:
        assert AXIS_ANCHOR[axis] in FEATURE_NAMES


def test_empty_window_is_nan_not_a_crash():
    h = head_stats(np.zeros(0), 0.5, "amplifier")
    assert h.n_users == 0 and h.count == 0
    assert np.isnan(h.rate_per_100k)


def test_nobody_above_the_bar_is_zero_not_nan():
    """A real answer: an event where the archetype vanished must read as zero."""
    h = head_stats(np.zeros(1000), 0.5, "coordinated")
    assert h.count == 0
    assert h.rate_per_100k == 0.0
    assert np.isnan(h.anchor_median), "no head, so nothing to quote"


# ----------------------------------------------------------------------------- gini


def test_gini_of_perfect_equality_is_zero():
    assert gini(np.full(1000, 4.2)) == pytest.approx(0.0, abs=1e-9)


def test_gini_of_one_account_holding_everything_approaches_one():
    v = np.zeros(10_000)
    v[0] = 1.0
    assert gini(v) > 0.999


def test_gini_of_all_zeros_is_zero_not_a_divide_by_zero():
    assert gini(np.zeros(100)) == 0.0


def test_gini_rises_as_the_tail_gets_heavier():
    rng = np.random.default_rng(0)
    mild = gini(rng.pareto(3.0, 10_000))
    heavy = gini(rng.pareto(1.1, 10_000))
    assert heavy > mild


def test_gini_needs_no_threshold():
    """The reason it is here: prevalence and intensity both depend on where the bar is
    put, so both can be argued with by arguing about 0.999. This cannot."""
    v = np.abs(np.random.default_rng(1).normal(0, 1, 5000))
    assert gini(v) == gini(v)  # no bar in the signature at all


def test_gini_ignores_nan():
    v = np.array([1.0, 2.0, np.nan, 3.0])
    assert np.isfinite(gini(v))


# ------------------------------------------------------------------------ top share


def test_top_one_percent_share_of_a_uniform_population_is_one_percent():
    assert top_share(np.full(1000, 1.0), 0.01) == pytest.approx(0.01)


def test_top_one_percent_share_of_a_winner_take_all_population_is_one():
    v = np.zeros(1000)
    v[0] = 5.0
    assert top_share(v, 0.01) == pytest.approx(1.0)


def test_top_share_of_everything_is_everything():
    v = np.abs(np.random.default_rng(2).normal(0, 1, 500))
    assert top_share(v, 1.0) == pytest.approx(1.0)


def test_top_share_rejects_a_nonsense_fraction():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="fraction must be"):
            top_share(np.ones(10), bad)


# ------------------------------------------------------------------------- the sweep


def test_the_sweep_shows_whether_the_finding_belongs_to_the_data_or_the_bar():
    """A real change should be visible at every bar; only its size should differ."""
    pre = population(100_000, 10, seed=1)
    post = population(100_000, 200, seed=2)
    swept = sweep_quantiles([pre, post], "superspreader", (0.99, 0.999, 0.9999))

    assert set(swept) == {0.99, 0.999, 0.9999}
    for q, rates in swept.items():
        assert rates[1] > rates[0], f"the rise should survive q={q}"


def test_the_sweep_can_report_a_finding_that_is_only_the_bars():
    """The case the sweep exists to catch, so it must be able to happen.

    Here the head shrinks but a mid-band grows. At a very high bar prevalence falls; at
    a low bar it rises. Reported at one bar, either is a 'finding'.
    """
    rng = np.random.default_rng(5)
    pre = np.concatenate([np.abs(rng.normal(0, 0.02, 99_900)), np.full(100, 0.95)])
    post = np.concatenate([np.abs(rng.normal(0, 0.02, 97_000)), np.full(3_000, 0.55),
                           np.full(10, 0.95)])
    swept = sweep_quantiles([pre, post], "superspreader", (0.99, 0.9999))
    assert swept[0.99][1] > swept[0.99][0]      # low bar: "coordination rose"
    assert swept[0.9999][1] < swept[0.9999][0]  # high bar: "coordination collapsed"


# ------------------------------------------------------------------- fold change


def test_fold_change_is_the_number_a_rare_class_is_read_in():
    assert fold_change(10.0, 15.0) == pytest.approx(1.5)


def test_fold_change_from_nothing_is_undefined_not_infinite():
    """0 -> 15 is not a fold change; it is a different sentence, and the figure must
    not draw it as an infinitely tall bar."""
    assert np.isnan(fold_change(0.0, 15.0))


def test_fold_change_propagates_nan():
    assert np.isnan(fold_change(float("nan"), 1.0))
    assert np.isnan(fold_change(1.0, float("nan")))


def test_default_quantile_is_documented_as_a_figure_parameter():
    assert DEFAULT_QUANTILE == 0.999
