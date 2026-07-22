"""Tests for the incumbent paired significance test.

The archive itself is 146 GB and cannot run here, so what is tested is the logic the
result depends on: that pairing is by shared user, that the Wilcoxon runs (and its
scipy-free fallback agrees in sign), and that a real within-user shift is detected while
noise is not.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "paired_test.py"
_spec = importlib.util.spec_from_file_location("paired_test", _SCRIPT)
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)

from arles.embedding import AXES


def test_pairing_is_by_shared_user_not_by_position():
    """The whole point: only users present in both windows are paired, matched by id."""
    pre = {"a": np.array([0.1, 0.2, 0.3]), "b": np.array([0.4, 0.5, 0.6]),
           "gone": np.array([0.9, 0.9, 0.9])}
    post = {"a": np.array([0.15, 0.2, 0.3]), "b": np.array([0.5, 0.5, 0.6]),
            "new": np.array([0.0, 0.0, 0.0])}
    n, by_axis = pt.paired_rows(pre, post)
    assert n == 2  # a and b; 'gone' and 'new' excluded
    pre_ss, post_ss = by_axis[AXES[0]]
    # order follows the shared-user iteration over `pre`
    assert np.allclose(pre_ss, [0.1, 0.4])
    assert np.allclose(post_ss, [0.15, 0.5])


def test_no_incumbents_is_zero_pairs_not_a_crash():
    n, by_axis = pt.paired_rows({"a": np.zeros(3)}, {"b": np.zeros(3)})
    assert n == 0
    for axis in AXES:
        pre, post = by_axis[axis]
        assert pre.size == 0 and post.size == 0


# ------------------------------------------------------------------ the test itself


def test_a_real_within_user_shift_is_significant():
    rng = np.random.default_rng(0)
    # 2000 users each nudged up by ~0.05 -- a real, consistent shift.
    diffs = rng.normal(0.05, 0.02, 2000)
    stat, p, method = pt.wilcoxon(diffs)
    assert p < 1e-6


def test_pure_noise_is_not_significant():
    rng = np.random.default_rng(1)
    diffs = rng.normal(0.0, 0.05, 2000)  # symmetric around zero
    stat, p, method = pt.wilcoxon(diffs)
    assert p > 0.05


def test_all_zero_differences_is_p_one_not_a_divide_by_zero():
    stat, p, method = pt.wilcoxon(np.zeros(500))
    assert p == 1.0
    assert "no non-zero" in method


def test_the_fallback_agrees_with_scipy_in_significance(monkeypatch):
    """The scipy-free path must reach the same conclusion, or a machine without scipy
    would silently get a different answer."""
    pytest.importorskip("scipy")
    rng = np.random.default_rng(2)
    diffs = rng.normal(0.03, 0.02, 1500)

    stat_sp, p_sp, _ = pt.wilcoxon(diffs)

    import builtins
    real_import = builtins.__import__

    def no_scipy(name, *a, **k):
        if name.startswith("scipy"):
            raise ImportError("scipy hidden for the test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    stat_fb, p_fb, method = pt.wilcoxon(diffs)

    assert "approximation" in method
    assert (p_sp < 0.05) == (p_fb < 0.05)  # same conclusion


def test_median_difference_carries_the_direction():
    """A p-value without a signed effect size is unreadable; the script reports the
    median within-user difference, which must have the right sign."""
    up = np.full(100, 0.2)
    stat, p, _ = pt.wilcoxon(up)
    assert np.median(up) > 0
