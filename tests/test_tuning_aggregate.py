"""Tests for dataset-level (rather than per-period) parameter fitting.

Fitting on one period overfits it. Measured on the real archive with the same script:

    TASH optimum on the E1 pair : alpha=0.7, delta=15min   (nDCG@100 0.5115)
    TASH optimum on the E2 pair : alpha=0.5, delta=2d      (nDCG@100 0.3038)

A 192x disagreement on delta from the same dataset, and adopting either one costs the
other period real accuracy. Both sit on a surface that is flat to within ~0.01 nDCG
across a wide region, so the argmax is noise. These tests pin the machinery that
detects that: averaging across pairs, and reporting the plateau instead of a bare
argmax.
"""

from datetime import timedelta

import pytest

from arles.tuning import (
    TuningResult,
    aggregate,
    beats_on_every_pair,
    format_grid,
    plateau,
)

D15, D1H, D2D = timedelta(minutes=15), timedelta(hours=1), timedelta(days=2)


def r(metric, alpha, delta, score):
    return TuningResult(metric=metric, alpha=alpha, delta=delta, ndcg={100: score})


def test_aggregate_averages_across_pairs():
    pair_a = [r("tash_index", 0.5, D15, 0.60)]
    pair_b = [r("tash_index", 0.5, D15, 0.20)]
    merged = aggregate([pair_a, pair_b])
    assert len(merged) == 1
    assert merged[0].score == pytest.approx(0.40)
    assert merged[0].n_pairs == 2
    assert merged[0].per_pair == (0.60, 0.20)


def test_aggregate_reports_spread_across_pairs():
    """A large std is the signal that a setting is period-specific, not general."""
    stable = [[r("tash_index", 0.5, D1H, 0.40)], [r("tash_index", 0.5, D1H, 0.42)]]
    volatile = [[r("tash_index", 0.7, D15, 0.51)], [r("tash_index", 0.7, D15, 0.20)]]
    assert aggregate(stable)[0].std < 0.02
    assert aggregate(volatile)[0].std > 0.15


def test_aggregate_reproduces_the_measured_e1_e2_disagreement():
    """The real numbers: each period's argmax is the other's mediocre setting."""
    e1 = [
        r("tash_index", 0.7, D15, 0.5115),   # E1's argmax
        r("tash_index", 0.5, D2D, 0.3400),
    ]
    e2 = [
        r("tash_index", 0.7, D15, 0.2022),
        r("tash_index", 0.5, D2D, 0.3038),   # E2's argmax
    ]
    merged = aggregate([e1, e2])
    best = max(merged, key=lambda x: x.score)
    # E1's argmax averages to 0.357, E2's to 0.322 -- neither is a dataset optimum,
    # and the winner flips depending on which period you happen to fit on.
    by_setting = {(x.alpha, x.delta): x.score for x in merged}
    assert by_setting[(0.7, D15)] == pytest.approx((0.5115 + 0.2022) / 2)
    assert by_setting[(0.5, D2D)] == pytest.approx((0.3400 + 0.3038) / 2)
    # and the per-pair spread exposes it
    volatile = next(x for x in merged if x.alpha == 0.7)
    assert volatile.std > 0.15


def test_aggregate_keeps_metrics_separate():
    """TASH and TAI are fitted independently: their optima need not coincide."""
    pair = [r("tash_index", 0.5, D15, 0.5), r("tai_score", 0.5, D15, 0.3)]
    merged = aggregate([pair, pair])
    assert {x.metric for x in merged} == {"tash_index", "tai_score"}
    assert len(merged) == 2


def test_aggregate_of_nothing_is_empty():
    assert aggregate([]) == []


# ------------------------------------------------------------------- plateau


def test_plateau_finds_everything_within_tolerance():
    rows = [
        r("tash_index", 0.1, D15, 0.380),
        r("tash_index", 0.2, D15, 0.375),
        r("tash_index", 0.3, D15, 0.372),
        r("tash_index", 0.4, D15, 0.100),
    ]
    near, best = plateau(rows, "tash_index", tolerance=0.01)
    assert best.score == 0.380
    assert len(near) == 3          # the 0.100 is excluded
    assert near[0] is best         # sorted best-first


def test_plateau_of_a_genuine_peak_is_just_the_peak():
    rows = [
        r("tash_index", 0.1, D15, 0.90),
        r("tash_index", 0.2, D15, 0.10),
        r("tash_index", 0.3, D15, 0.11),
    ]
    near, best = plateau(rows, "tash_index", tolerance=0.01)
    assert len(near) == 1
    assert best.alpha == 0.1


def test_flat_surface_is_called_out_in_the_rendered_grid():
    """When many settings tie, the grid must say so rather than imply precision."""
    rows = []
    for a in (0.1, 0.2, 0.3, 0.4, 0.5):
        for d in (D15, D1H, D2D):
            rows.append(r("tash_index", a, d, 0.37 + 0.001 * a))
    grid = format_grid(rows, "tash_index", tolerance=0.01)
    assert "FLAT" in grid
    assert "not a bare argmax" in grid
    assert "+" in grid


def test_peaked_surface_is_not_called_flat():
    rows = [r("tash_index", a, D15, 0.9 if a == 0.5 else 0.1)
            for a in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
    grid = format_grid(rows, "tash_index", tolerance=0.01)
    assert "FLAT" not in grid
    assert "plateau: 1/8" in grid


def test_grid_reports_pair_count_and_spread_when_aggregated():
    merged = aggregate([
        [r("tash_index", 0.5, D15, 0.60)],
        [r("tash_index", 0.5, D15, 0.20)],
    ])
    grid = format_grid(merged, "tash_index")
    assert "mean of 2 window pairs" in grid
    assert "+/-" in grid


# ------------------------------------------------- the NaN-key bug (real, shipped)


def _static(metric, score):
    """A static baseline row, exactly as baseline_results builds it: alpha = NaN."""
    return TuningResult(metric=metric, alpha=float("nan"), delta=timedelta(0),
                        ndcg={100: score})


def test_static_baselines_are_aggregated_not_split_per_pair():
    """The regression for a bug that shipped and corrupted a 5.5-hour run's output.

    aggregate() keyed on (metric, alpha, delta) and the baselines carry alpha=NaN.
    NaN != NaN, so every pair's baseline landed in its own bucket: a 6-pair sweep
    printed influence_score six times instead of averaging it once, and the paired
    comparison then zipped a 6-tuple against a 1-tuple, truncated to one element, and
    reported meaningless "0/6 pairs" verdicts.
    """
    pairs = [[_static("influence_score", s)] for s in
             (0.9196, 0.9141, 0.8904, 0.8867, 0.6288, 0.4860)]
    merged = aggregate(pairs)
    assert len(merged) == 1, "NaN alpha split the baseline into one bucket per pair"
    assert merged[0].n_pairs == 6
    assert merged[0].score == pytest.approx(0.7876, abs=1e-4)  # the real archive value


def test_aggregated_baseline_keeps_a_numeric_alpha():
    """The sentinel used for keying must not leak into the result object."""
    merged = aggregate([[_static("h_index", 0.8)], [_static("h_index", 0.6)]])
    assert merged[0].alpha != merged[0].alpha  # still NaN, not the string "static"


def test_baselines_and_grid_rows_coexist_after_aggregation():
    pairs = []
    for s in (0.80, 0.60):
        pairs.append([
            _static("influence_score", s),
            TuningResult("tai_score", 0.4, D2D, {100: s + 0.01}),
        ])
    merged = aggregate(pairs)
    assert {r.metric for r in merged} == {"influence_score", "tai_score"}
    assert all(r.n_pairs == 2 for r in merged)


def test_paired_comparison_works_against_an_aggregated_baseline():
    """Once aggregated, best-vs-baseline is a real 6-vs-6 paired comparison."""
    inf = aggregate([[_static("influence_score", s)] for s in
                     (0.92, 0.91, 0.89, 0.89, 0.63, 0.49)])[0]
    tai = aggregate([[TuningResult("tai_score", 0.4, D2D, {100: s})] for s in
                     (0.494, 0.902, 0.701, 0.920, 0.882, 0.885)])[0]
    # The real archive numbers: TAI wins on only some pairs, so it does not
    # consistently beat the static baseline.
    assert not beats_on_every_pair(tai, inf)
    assert not beats_on_every_pair(inf, tai)


# ------------------------------------------- paired comparison with magnitudes


def _res(metric, per_pair, alpha=0.5, delta=D1H):
    import numpy as np
    return TuningResult(metric=metric, alpha=alpha, delta=delta,
                        ndcg={100: float(np.mean(per_pair))},
                        ndcg_std={100: float(np.std(per_pair))},
                        n_pairs=len(per_pair), per_pair=tuple(per_pair))


def test_paired_test_uses_magnitude_not_just_sign():
    """Why the sign test was replaced.

    A setting that wins by +0.02 on five pairs and loses by -0.001 on one is plainly
    better, but scores 5/6 and the old unanimity rule called it 'not significant'.
    """
    from arles.tuning import compare_paired

    a = _res("tash_index", [0.52, 0.52, 0.52, 0.52, 0.52, 0.499])
    b = _res("h_index",    [0.50, 0.50, 0.50, 0.50, 0.50, 0.500])
    cmp = compare_paired(a, b)
    assert cmp.wins == 5                      # the old rule would reject on this alone
    assert cmp.mean_diff > 0
    assert not beats_on_every_pair(a, b)      # sign test: fails
    assert cmp.p_value < 0.10                 # magnitude test: nearly there at n=6


def test_paired_test_reports_effect_size_and_direction():
    from arles.tuning import compare_paired

    a = _res("tai_score", [0.80, 0.81, 0.79, 0.82, 0.80, 0.81])
    b = _res("influence_score", [0.79, 0.80, 0.78, 0.81, 0.79, 0.80])
    cmp = compare_paired(a, b)
    assert cmp.mean_diff == pytest.approx(0.01, abs=1e-6)
    assert cmp.wins == 6
    assert "better" in cmp.describe()
    assert cmp.significant                    # consistent +0.01 clears p=0.031


def test_paired_test_calls_noise_insignificant():
    from arles.tuning import compare_paired

    a = _res("tash_index", [0.52, 0.47, 0.51, 0.48, 0.53, 0.49])
    b = _res("h_index",    [0.50, 0.50, 0.50, 0.50, 0.50, 0.50])
    cmp = compare_paired(a, b)
    assert not cmp.significant


def test_paired_test_needs_matching_pair_counts():
    """The shape bug: zipping a 6-tuple against a 1-tuple silently truncated to 1."""
    from arles.tuning import compare_paired

    a = _res("tash_index", [0.5] * 6)
    b = _res("h_index", [0.4])
    assert compare_paired(a, b) is None
