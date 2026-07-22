#!/usr/bin/env python3
"""Two-proportion significance test on archetype prevalence, pre vs post event.

The right test for these scores, and the counterpart to scripts/paired_test.py, which
showed why the per-user test is wrong here: the archetype scores are floor-heavy (most
users score zero), so a test on the scores themselves is dominated by the floor and its
p-values are driven by sample size rather than effect. Prevalence -- the fraction of
accounts scoring above the bar -- is the statistic that carries the signal, so the test
belongs on the prevalence counts.

For each event and axis it compares the pre-event window against a post-event window with a
two-proportion z-test: does the share of accounts above the bar differ, and by how much.
Here significance and effect size travel together -- E1's coordinated prevalence rises
about tenfold, so its tiny p-value reflects a real, large change, not merely a large n.

Input is figures/prevalence.csv (written by make_figures); no archive pass is needed, so
this runs in a second.

Usage
-----
    python scripts/prevalence_test.py figures/prevalence.csv --out figures/prevalence_test.csv
    python scripts/prevalence_test.py figures/prevalence.csv --post-window 2
"""

import argparse
import csv
from math import erf, sqrt
from pathlib import Path


def two_proportion(c1, n1, c2, n2):
    """z and two-sided p for H0: the two prevalence rates are equal.

    Pooled-variance two-proportion z-test. Returns (z, p, rate1_per_100k, rate2_per_100k).
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0, float("nan"), float("nan")
    p1, p2 = c1 / n1, c2 / n2
    pool = (c1 + c2) / (n1 + n2)
    se = sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2))))
    return z, p, p1 * 1e5, p2 * 1e5


def load(prevalence_csv):
    """{(event, axis, window): (count, n_users)} from a prevalence.csv."""
    out = {}
    for r in csv.DictReader(open(prevalence_csv)):
        out[(r["event"], r["axis"], int(r["window"]))] = (
            int(r["count"]), int(r["n_users"])
        )
    return out


def run(cells, post_window=2):
    """One row per (event, axis): pre window 1 vs the chosen post window."""
    events, axes = [], []
    for (ev, ax, _w) in cells:
        if ev not in events:
            events.append(ev)
        if ax not in axes:
            axes.append(ax)
    rows = []
    for ev in events:
        for ax in axes:
            pre = cells.get((ev, ax, 1))
            post = cells.get((ev, ax, post_window))
            if pre is None or post is None:
                continue
            z, p, r1, r2 = two_proportion(pre[0], pre[1], post[0], post[1])
            rows.append({
                "event": ev, "axis": ax,
                "n_pre": pre[1], "n_post": post[1],
                "rate_pre_per_100k": round(r1, 1), "rate_post_per_100k": round(r2, 1),
                "rate_ratio": round(r2 / r1, 2) if r1 else "",
                "z": round(z, 2), "p_value": p,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("prevalence_csv", nargs="?", default="figures/prevalence.csv")
    ap.add_argument("--out", default="figures/prevalence_test.csv")
    ap.add_argument("--post-window", type=int, default=2,
                    help="post window to compare against window 1 (default 2, first post)")
    args = ap.parse_args()

    cells = load(args.prevalence_csv)
    rows = run(cells, args.post_window)

    print(f"pre (window 1) vs post (window {args.post_window}), two-proportion z-test")
    print(f"{'event':6}{'axis':15}{'pre/100k':>10}{'post/100k':>10}{'ratio':>7}"
          f"{'z':>8}{'p':>12}")
    for r in rows:
        ps = f"{r['p_value']:.1e}" if r["p_value"] > 0 else "<1e-300"
        print(f"{r['event']:6}{r['axis']:15}{r['rate_pre_per_100k']:>10.1f}"
              f"{r['rate_post_per_100k']:>10.1f}{str(r['rate_ratio']):>7}{r['z']:>8.1f}"
              f"{ps:>12}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
