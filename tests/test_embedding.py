"""Tests for the 12 -> 3 archetype embedding.

The central claim: a pooled fit makes scores comparable across windows, and a
per-window fit does not. Measured on the full archive, E1's seven windows swing the
maximum reshares received per author 6,842 -> 275,198 -- a 40x change in the min-max
denominator -- so the same behaviour scores 1.0 in one window and 0.025 in another.
test_pooled_fit_survives_the_real_40x_swing reproduces that with the real numbers.
"""

import numpy as np
import pytest

from arles.embedding import AXES, BUCKETS, ArchetypeEmbedder, fit_pooled
from arles.features import FEATURE_NAMES


def make_X(n, seed=0, scale=1.0):
    """A feature matrix with the shape real data has: heavy-tailed, correlated."""
    rng = np.random.default_rng(seed)
    latent = rng.pareto(1.5, size=n) * scale  # the diffusion power we hope PC1 finds
    X = np.zeros((n, len(FEATURE_NAMES)))
    for j in range(len(FEATURE_NAMES)):
        X[:, j] = latent * rng.uniform(0.5, 1.5, n) + rng.exponential(0.5, n)
    return X


# --------------------------------------------------------------------- structure


def test_three_axes_of_four_features_each():
    assert len(BUCKETS) == 3
    assert AXES == ("superspreader", "amplifier", "coordinated")
    assert sum(len(f) for f in BUCKETS.values()) == len(FEATURE_NAMES)
    # every feature belongs to exactly one axis
    flat = [f for feats in BUCKETS.values() for f in feats]
    assert sorted(flat) == sorted(FEATURE_NAMES)


def test_output_is_three_columns_in_the_unit_interval():
    X = make_X(500)
    Z = ArchetypeEmbedder().fit_transform(X)
    assert Z.shape == (500, 3)
    assert np.all((Z >= 0.0) & (Z <= 1.0))


def test_transform_before_fit_is_an_error_not_a_guess():
    with pytest.raises(RuntimeError, match="fit"):
        ArchetypeEmbedder().transform(make_X(10))


def test_wrong_width_is_rejected():
    with pytest.raises(ValueError, match="FEATURE_NAMES"):
        ArchetypeEmbedder().fit(np.zeros((10, 5)))


# --------------------------------------------------- the reason this module exists


def test_pooled_fit_survives_the_real_40x_swing():
    """The bug the module exists to prevent, with the archive's real numbers.

    E1's windows swing max influence_score 6,842 -> 275,198. A user with a fixed
    behaviour must not have their score collapse just because a later window contains
    a bigger outlier.
    """
    rng = np.random.default_rng(1)

    def window(max_influence, n=2000):
        X = np.abs(rng.normal(3, 2, size=(n, len(FEATURE_NAMES))))
        X[0, :] = max_influence  # the window's extreme account
        X[1, :] = 6842          # the SAME user, present in both windows
        return X

    w_pre = window(6_842)
    w_late = window(275_198)

    # Per-window fitting: our fixed user is the extreme in one window and ordinary in
    # the other, so their score moves although their behaviour did not.
    per_pre = ArchetypeEmbedder().fit_transform(w_pre)[1, 0]
    per_late = ArchetypeEmbedder().fit_transform(w_late)[1, 0]
    assert abs(per_pre - per_late) > 0.2, "expected the per-window fit to distort"

    # Pooled: one ruler for both windows.
    pooled = fit_pooled([w_pre, w_late])
    poo_pre = pooled.transform(w_pre)[1, 0]
    poo_late = pooled.transform(w_late)[1, 0]
    assert poo_pre == pytest.approx(poo_late, abs=1e-9)


def test_pooled_transform_is_identical_however_the_windows_are_split():
    """A window's scores must not depend on which other windows were in the batch,
    only on the fit -- otherwise 'pooled' just moves the problem."""
    a, b, c = make_X(300, 1), make_X(300, 2), make_X(300, 3)
    emb = fit_pooled([a, b, c])
    together = emb.transform(np.vstack([a, b, c]))
    apart = np.vstack([emb.transform(a), emb.transform(b), emb.transform(c)])
    assert np.allclose(together, apart)


def test_frozen_fit_roundtrips_through_json():
    """A figure must be regenerable on the same scale months later."""
    X = make_X(400)
    emb = fit_pooled([X])
    reloaded = ArchetypeEmbedder.from_json(emb.to_json())
    assert np.allclose(emb.transform(X), reloaded.transform(X))


# ------------------------------------------------------------------------- sign


def test_higher_features_always_mean_a_higher_score():
    """PC1's sign is arbitrary; unpinned it can invert between fits and min-max then
    silently reverses the axis."""
    X = make_X(500)
    emb = ArchetypeEmbedder().fit(X)
    for axis in AXES:
        assert emb.buckets[axis].loadings.sum() > 0

    busy = np.full((1, len(FEATURE_NAMES)), 100.0)
    idle = np.zeros((1, len(FEATURE_NAMES)))
    assert np.all(emb.transform(busy) >= emb.transform(idle))


def test_sign_is_stable_across_independent_fits():
    for seed in range(5):
        emb = ArchetypeEmbedder().fit(make_X(300, seed=seed))
        for axis in AXES:
            assert emb.buckets[axis].loadings.sum() > 0


# -------------------------------------------------------------------- log scaling


def test_log_scaling_stops_one_account_from_defining_the_axis():
    """influence_score spans 1..275,198 with a median of 3. On raw counts the variance
    IS the outlier, so PC1 describes it and min-max crushes everyone else to zero.

    Measured on this fixture: the other 1,999 accounts average 0.0000 raw and 0.0473
    logged. Log scaling buys back three orders of magnitude of usable range -- but note
    it does not fully rescue min-max, which stays outlier-sensitive by construction.
    That is why the scores are small in absolute terms, and why the pooled fit (not the
    log) is what makes them comparable across windows.
    """
    rng = np.random.default_rng(7)
    X = np.abs(rng.normal(3, 1, size=(2000, len(FEATURE_NAMES))))
    X[0, :] = 275_198  # one whale

    raw = ArchetypeEmbedder(log_scale=False).fit_transform(X)
    logged = ArchetypeEmbedder(log_scale=True).fit_transform(X)

    assert raw[0, 0] == pytest.approx(1.0) and logged[0, 0] == pytest.approx(1.0)
    # Raw: the bulk is annihilated -- the axis carries no information about them.
    assert raw[1:, 0].max() < 0.001
    # Logged: the bulk is separable again, by orders of magnitude.
    assert logged[1:, 0].mean() > 50 * max(raw[1:, 0].mean(), 1e-9)
    assert logged[1:, 0].max() > 0.05


def test_log_scaling_may_reorder_because_pc1_reweights():
    """log1p is monotone per feature, but PC1 is a linear combination of the
    transformed features -- so the projection's order can legitimately change.

    Worth pinning: an earlier version of this test asserted the ranking was preserved,
    which is false and would have masked a real reweighting.
    """
    X = make_X(300, seed=5)
    raw = ArchetypeEmbedder(log_scale=False).fit_transform(X)[:, 0]
    logged = ArchetypeEmbedder(log_scale=True).fit_transform(X)[:, 0]
    assert not np.array_equal(np.argsort(raw), np.argsort(logged))


def test_a_user_dominating_on_every_feature_ranks_top_either_way():
    """The monotonicity that does hold: beat everyone on all four features of a bucket
    and you top that axis, log-scaled or not."""
    X = make_X(300, seed=6)
    X[0, :] = X[:, :].max(axis=0) * 2.0
    for log_scale in (False, True):
        Z = ArchetypeEmbedder(log_scale=log_scale).fit_transform(X)
        assert Z[0, 0] == pytest.approx(Z[:, 0].max())


# ----------------------------------------------------------------- degenerate data


def test_constant_column_does_not_produce_nan():
    X = make_X(200)
    X[:, 0] = 5.0  # a feature that never varies
    Z = ArchetypeEmbedder().fit_transform(X)
    assert np.all(np.isfinite(Z))


def test_all_identical_users_score_mid_scale_not_nan():
    """Zero spread means no information, so 0.5 rather than a divide-by-zero."""
    X = np.full((50, len(FEATURE_NAMES)), 4.0)
    Z = ArchetypeEmbedder().fit_transform(X)
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z, 0.5)


def test_single_user_is_finite():
    Z = ArchetypeEmbedder().fit_transform(np.full((1, len(FEATURE_NAMES)), 2.0))
    assert Z.shape == (1, 3)
    assert np.all(np.isfinite(Z))


def test_nan_and_inf_in_features_do_not_propagate():
    X = make_X(100)
    X[0, 0] = np.nan
    X[1, 1] = np.inf
    Z = ArchetypeEmbedder().fit_transform(X)
    assert np.all(np.isfinite(Z))


def test_values_outside_the_fitted_range_are_clipped_not_extrapolated():
    """A user more extreme than anything in the pooled fit saturates at 1.0."""
    X = make_X(300)
    emb = ArchetypeEmbedder().fit(X)
    beyond = np.full((1, len(FEATURE_NAMES)), 1e9)
    Z = emb.transform(beyond)
    assert np.all(Z <= 1.0) and np.all(Z >= 0.0)
    assert np.allclose(Z, 1.0)


# ---------------------------------------------------------------------- reporting


def test_loadings_table_names_every_feature():
    emb = ArchetypeEmbedder().fit(make_X(200))
    table = emb.loadings_table()
    for name in FEATURE_NAMES:
        assert name in table
    for axis in AXES:
        assert axis in table


# ------------------------------------------------------- degeneracy is not a score


def test_a_bucket_with_no_spread_is_flagged_not_silently_scored_half():
    """The trap this guard exists for.

    When a bucket's features never vary -- e.g. co-action on data too sparse for two
    users to touch the same content within delta_t -- the axis returns 0.5 for every
    user. That reads as "moderate coordination". It means "no measurement". Observed
    for real on a fixture whose reposts never collided.
    """
    X = make_X(300)
    for j, name in enumerate(FEATURE_NAMES):
        if name in BUCKETS["coordinated"]:
            X[:, j] = 0.0  # nobody ever co-acts

    emb = ArchetypeEmbedder().fit(X)
    Z = emb.transform(X)

    coord = AXES.index("coordinated")
    assert np.allclose(Z[:, coord], 0.5)          # the seductive number
    assert emb.buckets["coordinated"].degenerate  # ...is flagged
    assert not emb.buckets["superspreader"].degenerate

    warnings = emb.warnings()
    assert any("coordinated" in w and "DEGENERATE" in w for w in warnings)
    assert any("not measured" in w for w in warnings)
    assert "[DEGENERATE]" in emb.loadings_table()


def test_a_single_dead_feature_is_reported_without_killing_the_axis():
    X = make_X(300)
    X[:, FEATURE_NAMES.index("niche_co_action")] = 7.0  # constant, so zero loading

    emb = ArchetypeEmbedder().fit(X)
    fit = emb.buckets["coordinated"]
    assert not fit.degenerate
    assert "niche_co_action" in fit.dead_features
    assert any("contribute nothing" in w for w in emb.warnings())
    assert "contributes nothing" in emb.loadings_table()


def test_healthy_data_produces_no_warnings():
    emb = ArchetypeEmbedder().fit(make_X(500))
    assert emb.warnings() == []
    assert "DEGENERATE" not in emb.loadings_table()
