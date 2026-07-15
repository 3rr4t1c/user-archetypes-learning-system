#!/usr/bin/env python3
"""Re-tune (alpha, delta) for the TASH-Index and TAI-Score on this dataset.

Why
---
Verdolotti et al. grid-search alpha and delta against nDCG on a target defined as
"the total number of reshares in which the user is engaged -- either as the original
author whose posts are amplified by others, or as a user who actively reshares such
content", obtaining alpha=0.5 / delta=14 days for TASH and alpha=0.6 / delta=18 days
for TAI.

Those values cannot be carried over. delta=14 days does not fit inside a 5-day
analysis window: the EMA would hold a single term, TASH would silently collapse to a
plain h-index, and the time-aware component would do nothing. And the grid was fitted
on a different platform with a different action rate.

The *procedure* transfers intact, though. Drop the misinformation restriction from
their target and what remains is well defined on any reshare stream. So this script
runs their optimisation, unchanged in method, on this data:

    rank users by the metric computed on window t
    score that ranking with nDCG against their involvement in window t+1
    grid-search (alpha, delta)

The result is an empirical parameter choice on this dataset, reported alongside the
full nDCG surface so a reader can see whether the optimum is a plateau or a spike --
which is what the paper needs to say, and what a reviewer will ask for.

Usage
-----
    python scripts/tune_time_aware.py /Volumes/Uniform/bluesky_full \\
        --start 2024-10-12 --days 5

    # narrower sweep
    python scripts/tune_time_aware.py <archive> --start 2024-08-25 --days 5 \\
        --alphas 0.3,0.5,0.7 --deltas 1h,6h,12h

Reading the result
------------------
The surface is printed in full. If it is flat, say so in the paper and the exact value
does not matter much. If it is peaked, the argmax is doing real work and belongs in
the text with its nDCG.
"""

import argparse
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arles.arles import MalformedActionError, parse_timestamp  # noqa: E402
from arles.features import FeatureExtractor, WindowIndex  # noqa: E402
from arles.mappers.bluesky import map_row  # noqa: E402
from arles.streaming import build_index, discover_files, iter_window  # noqa: E402
from arles.tuning import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_DELTAS,
    format_grid,
    grid_search,
    involvement_target,
)

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_cli_datetime(value):
    s = value.strip()
    if _DATE_ONLY_RE.match(s):
        s = f"{s} 00:00:00+00:00"
    try:
        return parse_timestamp(s)
    except MalformedActionError:
        raise argparse.ArgumentTypeError(
            f"invalid --start {value!r}: expected YYYY-MM-DD or a full ISO timestamp"
        )


def parse_delta(text):
    m = re.match(r"^(\d+(?:\.\d+)?)(min|h|d)$", text.strip())
    if not m:
        raise argparse.ArgumentTypeError(
            f"invalid delta {text!r}: expected e.g. 15min, 6h, 2d"
        )
    v, unit = float(m.group(1)), m.group(2)
    unit_map = {"min": "minutes", "h": "hours", "d": "days"}
    return timedelta(**{unit_map[unit]: v})


def load_window(files, spans, start, end, progress=True):
    """Both passes over one window. Returns (extractor, user_ids)."""
    index = WindowIndex()
    for row in iter_window(files, start, end, spans=spans, progress=progress):
        try:
            action = map_row(row)
        except MalformedActionError:
            continue
        index.add(action)

    fx = FeatureExtractor(index, start, end)
    for row in iter_window(files, start, end, spans=spans, progress=progress):
        try:
            action = map_row(row)
        except MalformedActionError:
            continue
        fx.add(action)
    return fx, index


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="a CSV file, or a directory of sequential CSVs")
    ap.add_argument("--start", type=parse_cli_datetime, required=True,
                    help="start of the fitting window (t)")
    ap.add_argument("--days", type=float, default=5.0,
                    help="window length in days (default: 5, the paper's window)")
    ap.add_argument("--alphas", default=None,
                    help="comma-separated, e.g. 0.3,0.5,0.7 (default: 0.0..0.9)")
    ap.add_argument("--deltas", default=None,
                    help="comma-separated, e.g. 15min,1h,6h,1d (default: 15min..2d)")
    ap.add_argument("--index-cache", default=None)
    args = ap.parse_args()

    alphas = (
        tuple(float(a) for a in args.alphas.split(","))
        if args.alphas else DEFAULT_ALPHAS
    )
    deltas = (
        tuple(parse_delta(d) for d in args.deltas.split(","))
        if args.deltas else DEFAULT_DELTAS
    )

    t0 = time.time()
    files = discover_files(args.path)
    cache = args.index_cache
    if cache is None and Path(args.path).is_dir():
        cache = str(Path(args.path) / ".arles_index.json")
    spans = build_index(files, cache_path=cache, verbose=True)

    step = timedelta(days=args.days)
    fit_start, fit_end = args.start, args.start + step
    tgt_start, tgt_end = fit_end, fit_end + step

    print(f"\nfitting window (t)  : {fit_start.isoformat()[:19]} -> {fit_end.isoformat()[:19]}")
    print(f"target window (t+1) : {tgt_start.isoformat()[:19]} -> {tgt_end.isoformat()[:19]}")
    print("\nreading window t ...")
    fx, index = load_window(files, spans, fit_start, fit_end)
    user_ids, _ = fx.finish()
    print(f"  {index.n_reposts:,} reposts, {len(user_ids):,} users")

    print("\nreading window t+1 (target) ...")
    target_actions = []
    for row in iter_window(files, tgt_start, tgt_end, spans=spans, progress=True):
        try:
            action = map_row(row)
        except MalformedActionError:
            continue
        if action.activity_type == "repost":
            target_actions.append(action)
    target = involvement_target(target_actions, user_ids)
    print(f"  {len(target_actions):,} reposts; "
          f"{int((target > 0).sum()):,}/{len(user_ids):,} users still active")

    print(f"\ngrid search: {len(alphas)} alphas x {len(deltas)} deltas")
    best_tash, best_tai, results = grid_search(
        fx, target, alphas=alphas, deltas=deltas, progress=True
    )

    print()
    print(format_grid(results, "tash_index"))
    print()
    print(format_grid(results, "tai_score"))

    print()
    print("=" * 72)
    print("RESULT (quote these in the paper)")
    print("=" * 72)
    for r, prior in ((best_tash, "alpha=0.5, delta=14d"), (best_tai, "alpha=0.6, delta=18d")):
        print(f"  {r.metric:<12} alpha={r.alpha:.1f}  delta={_fmt(r.delta)}")
        print(f"               nDCG@10={r.ndcg.get(10, 0):.4f}  "
              f"nDCG@100={r.ndcg.get(100, 0):.4f}  nDCG@1000={r.ndcg.get(1000, 0):.4f}")
        print(f"               (Verdolotti et al.: {prior})")
    print(f"\nelapsed: {(time.time() - t0)/60:.1f} min")


def _fmt(d):
    s = d.total_seconds()
    if s < 3600:
        return f"{int(s // 60)}min"
    if s < 86400:
        return f"{s / 3600:g}h"
    return f"{s / 86400:g}d"


if __name__ == "__main__":
    main()
