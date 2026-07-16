#!/usr/bin/env python3
"""Re-analyse a saved tuning surface without touching the archive.

The sweep costs hours; the analysis should cost nothing. `tune_time_aware.py --save`
writes every setting's per-pair nDCG to JSON, and this reads it back, so a question
about the statistics never requires re-reading 146 GB.

What this is for, and what it is not
------------------------------------
ArLeS uses the TASH-Index and TAI-Score as *features* -- two of the four indicators
whose shared variance PC1 turns into the super-spreader axis. They are not competing
rankers, and nothing here is a benchmark: there is no baseline to beat. The paper's
question is how spreading behaviour shifts when users migrate, not which index ranks
best.

So the only job of the sweep is to choose (alpha, delta) defensibly, on this dataset,
by the procedure the metrics' own paper used -- rather than inheriting constants fitted
on a different platform, or picking a round number. What this script reports is
therefore the surface and its plateau, not a winner.

Reading the surface
-------------------
The `delta = analysis window` row is each metric's static counterpart: one slot means
the EMA never iterates, so TASH degenerates to a plain h-index and TAI to a plain
influence score. It is included so the surface shows what time-awareness is adding,
without a separate baseline computation. Verified on the archive: that row reads 0.7341
and 0.7876, matching independently computed h_index and influence_score to four
decimals.

A wide plateau means the choice does not matter much and the paper should say so,
quoting the region alongside the value. A narrow one means it does.

Usage
-----
    python scripts/analyse_tuning.py tuning_surface.json
    python scripts/analyse_tuning.py tuning_surface.json --metric tash_index
"""

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arles.tuning import (  # noqa: E402
    TuningResult,
    _fmt_delta,
    compare_paired,
    format_grid,
    plateau,
)


def load(path):
    rows = json.loads(Path(path).read_text())
    out = []
    for r in rows:
        out.append(
            TuningResult(
                metric=r["metric"],
                alpha=float(r["alpha"]),
                delta=timedelta(seconds=r["delta_seconds"]),
                ndcg={int(k): v for k, v in r["ndcg"].items()},
                ndcg_std={int(k): v for k, v in (r.get("ndcg_std") or {}).items()},
                n_pairs=r.get("n_pairs", 1),
                per_pair=tuple(r.get("per_pair") or ()),
            )
        )
    return out


def check_surface(results):
    """Refuse to analyse a surface this script cannot read correctly.

    A file written before the baselines were folded into the grid contains one row per
    pair for each static metric (alpha = NaN, never self-equal, so never aggregated).
    Analysing it silently produced an arbitrary single pair's score presented as a mean.
    Detect and say so rather than print a plausible number.
    """
    problems = []
    stale = [r for r in results if r.alpha != r.alpha]  # NaN alpha
    if stale:
        problems.append(
            f"{len(stale)} rows carry alpha=NaN. This surface was written before the "
            f"static baselines were folded into the grid as the delta=window row, and "
            f"those rows were never aggregated across pairs. Re-run the sweep (the "
            f"window cache makes it cheap) or ignore those rows."
        )
    counts = {r.n_pairs for r in results}
    if len(counts) > 1:
        problems.append(
            f"rows disagree on how many pairs they were averaged over ({sorted(counts)}), "
            f"so they are not comparable."
        )
    return problems


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("surface", help="JSON written by tune_time_aware.py --save")
    ap.add_argument("--metric", default=None, help="only this metric's grid")
    ap.add_argument("--tolerance", type=float, default=0.01)
    args = ap.parse_args()

    results = load(args.surface)
    problems = check_surface(results)
    if problems:
        print("This surface cannot be analysed as-is:\n")
        for p in problems:
            print(f"  - {p}\n")
        results = [r for r in results if r.alpha == r.alpha]
        print(f"Continuing with the {len(results)} grid rows only.\n")

    n_pairs = max((r.n_pairs for r in results), default=1)
    print(f"{len(results)} settings, {n_pairs} window pairs\n")

    metrics = [args.metric] if args.metric else ["tash_index", "tai_score"]
    for m in metrics:
        print(format_grid(results, m, tolerance=args.tolerance))
        print()

    print("=" * 72)
    print("PARAMETERS TO REPORT")
    print("=" * 72)
    for m in metrics:
        near, best = plateau(results, m)
        if best is None:
            continue
        window_rows = [r for r in results if r.metric == m and r.delta == max(
            x.delta for x in results if x.metric == m)]
        static = window_rows[0] if window_rows else None

        print(f"  {m}: alpha={best.alpha:.2f}, delta={_fmt_delta(best.delta)}, "
              f"nDCG@100={best.score:.4f}")
        print(f"      plateau: {len(near)} of "
              f"{len([r for r in results if r.metric == m])} settings indistinguishable")
        if static is not None:
            cmp = compare_paired(best, static)
            print(f"      static counterpart (delta={_fmt_delta(static.delta)}, one "
                  f"slot): {static.score:.4f}")
            if cmp is not None:
                print(f"      time-awareness contributes {cmp.mean_diff:+.4f} "
                      f"({cmp.wins}/{cmp.n} pairs, p={cmp.p_value:.3f})")
    print()
    print("  These are feature parameters, not a benchmark result. Report the value")
    print("  and the plateau; if the plateau is wide, say the choice is not critical.")


if __name__ == "__main__":
    main()
