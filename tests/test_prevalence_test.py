"""Tests for the two-proportion prevalence test.

This is the test the paper reports, so its arithmetic is pinned here: the z-test itself,
that a large real difference is significant while a null one is not, and that it reads the
prevalence counts straight from a prevalence.csv.
"""

import csv
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prevalence_test.py"
_spec = importlib.util.spec_from_file_location("prevalence_test", _SCRIPT)
pvt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pvt)


def test_a_tenfold_prevalence_change_is_significant():
    # E1 coordinated, roughly: 127 vs 1284 per 100k on ~66k / ~180k users.
    z, p, r1, r2 = pvt.two_proportion(84, 66000, 2312, 180000)
    assert r2 > 5 * r1
    assert p < 1e-50
    assert z > 0


def test_no_difference_is_not_significant():
    z, p, r1, r2 = pvt.two_proportion(1000, 100000, 1000, 100000)
    assert p == pytest.approx(1.0, abs=1e-9)
    assert abs(z) < 1e-9


def test_a_small_difference_on_small_n_is_not_significant():
    """The property the paired test lacked: with a modest n, a modest difference is
    correctly non-significant, so a significant result means something."""
    z, p, r1, r2 = pvt.two_proportion(10, 1000, 13, 1000)
    assert p > 0.05


def test_the_sign_of_z_follows_the_direction():
    up = pvt.two_proportion(10, 10000, 100, 10000)
    down = pvt.two_proportion(100, 10000, 10, 10000)
    assert up[0] > 0 and down[0] < 0


def test_empty_window_is_p_one_not_a_crash():
    z, p, r1, r2 = pvt.two_proportion(0, 0, 5, 100)
    assert p == 1.0


def test_it_reads_prevalence_csv_and_pairs_pre_with_post(tmp_path):
    csv_path = tmp_path / "prevalence.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "window", "axis", "count", "n_users"])
        w.writerow(["E1", 1, "coordinated", 84, 66000])
        w.writerow(["E1", 2, "coordinated", 2312, 180000])
        w.writerow(["E1", 1, "amplifier", 1600, 66000])
        w.writerow(["E1", 2, "amplifier", 3060, 180000])
    cells = pvt.load(str(csv_path))
    rows = pvt.run(cells, post_window=2)

    assert {r["axis"] for r in rows} == {"coordinated", "amplifier"}
    coord = next(r for r in rows if r["axis"] == "coordinated")
    assert coord["rate_ratio"] > 5
    assert coord["p_value"] < 1e-50


def test_a_missing_post_window_is_skipped_not_crashed(tmp_path):
    csv_path = tmp_path / "prevalence.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "window", "axis", "count", "n_users"])
        w.writerow(["E4", 1, "coordinated", 100, 50000])  # only window 1 present
    cells = pvt.load(str(csv_path))
    rows = pvt.run(cells, post_window=2)
    assert rows == []
