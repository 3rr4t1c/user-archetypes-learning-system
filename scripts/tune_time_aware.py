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
import json
import re
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arles.arles import MalformedActionError, parse_timestamp  # noqa: E402
from arles.features import FeatureExtractor, WindowIndex  # noqa: E402
from arles.mappers.bluesky import map_row  # noqa: E402
from arles.streaming import build_index, discover_files, iter_window  # noqa: E402
from arles.tuning import (  # noqa: E402
    DEFAULT_ALPHAS,
    DEFAULT_DELTAS,
    aggregate,
    format_baselines,
    format_grid,
    grid_search,
    involvement_target,
    plateau,
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


def _cache_key(start, end):
    return f"win_{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}"


def load_window(files, spans, start, end, progress=True, cache_dir=None):
    """Both passes over one window. Returns (extractor, index).

    Cached to `cache_dir` when given. The first sweep took 4.5 hours and nearly all of
    it was re-reading the same windows off the external drive to produce identical
    buffers. With the cache, changing the grid costs minutes.
    """
    cache_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / f"{_cache_key(start, end)}.npz"
        idx_path = Path(cache_dir) / f"{_cache_key(start, end)}.index.json"
        if cache_path.exists() and idx_path.exists():
            if progress:
                print(f"  cache hit: {cache_path.name}")
            meta = json.loads(idx_path.read_text())
            index = WindowIndex(
                content_reshares={},  # not needed once the buffer is built
                content_author={},
                n_reposts=meta["n_reposts"],
                n_unattributed=meta["n_unattributed"],
            )
            return FeatureExtractor.load_buffer(str(cache_path), index), index

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

    if cache_path:
        fx.save_buffer(str(cache_path))
        (Path(cache_dir) / f"{_cache_key(start, end)}.index.json").write_text(
            json.dumps({"n_reposts": index.n_reposts,
                        "n_unattributed": index.n_unattributed})
        )
        if progress:
            print(f"  cached: {cache_path.name}")
    return fx, index


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="a CSV file, or a directory of sequential CSVs")
    ap.add_argument("--start", type=parse_cli_datetime, default=None,
                    help="start of the first fitting window (default: archive start)")
    ap.add_argument("--days", type=float, default=5.0,
                    help="window length in days (default: 5, the paper's window)")
    ap.add_argument("--pairs", type=int, default=6,
                    help="number of (t, t+1) window pairs to fit across, spread evenly "
                         "over the archive (default: 6). Parameters belong to the "
                         "dataset, not to one period: fitted on E1 alone the TASH "
                         "optimum is delta=15min, on E2 alone it is delta=2d.")
    ap.add_argument("--alphas", default=None,
                    help="comma-separated, e.g. 0.3,0.5,0.7 (default: 0.0..0.9)")
    ap.add_argument("--deltas", default=None,
                    help="comma-separated, e.g. 15min,1h,6h,1d (default: 15min..2d)")
    ap.add_argument("--cache-dir", default=".arles_windows",
                    help="cache each window's buffer here so re-running with a "
                         "different grid skips the archive entirely (the first "
                         "sweep spent 4.5h almost entirely on re-reads). Pass '' "
                         "to disable.")
    ap.add_argument("--save", default=None,
                    help="write the full per-pair surface to JSON, so re-analysis "
                         "never needs a re-run")
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
    archive_start = args.start or spans[0].start
    archive_end = spans[-1].end

    # Spread the pairs evenly over the observation period. Each pair needs two
    # consecutive windows, so the last one must start 2*step before the end.
    usable = (archive_end - archive_start) - 2 * step
    if usable.total_seconds() <= 0:
        starts = [archive_start]
    else:
        n = max(1, args.pairs)
        stride = usable / max(n - 1, 1) if n > 1 else timedelta(0)
        starts = [archive_start + i * stride for i in range(n)]

    print(f"\nfitting {len(starts)} window pair(s) across "
          f"{archive_start.isoformat()[:10]} -> {archive_end.isoformat()[:10]}")
    for i, s in enumerate(starts, 1):
        print(f"  pair {i}: t = {s.isoformat()[:19]} -> {(s+step).isoformat()[:19]}, "
              f"t+1 = {(s+step).isoformat()[:19]} -> {(s+2*step).isoformat()[:19]}")

    per_pair: List[list] = []
    for i, fit_start in enumerate(starts, 1):
        fit_end = fit_start + step
        tgt_start, tgt_end = fit_end, fit_end + step
        print(f"\n{'=' * 72}\nPAIR {i}/{len(starts)}: {fit_start.isoformat()[:19]}\n{'=' * 72}")

        print("reading window t ...")
        fx, index = load_window(files, spans, fit_start, fit_end,
                                cache_dir=args.cache_dir or None)
        user_ids, _ = fx.finish()
        print(f"  {index.n_reposts:,} reposts, {len(user_ids):,} users")
        if not user_ids:
            print("  empty window; skipping this pair.")
            continue

        print("reading window t+1 (target) ...")
        target_actions = []
        for row in iter_window(files, tgt_start, tgt_end, spans=spans, progress=True):
            try:
                action = map_row(row)
            except MalformedActionError:
                continue
            if action.activity_type == "repost":
                target_actions.append(action)
        target = involvement_target(target_actions, user_ids)
        active = int((target > 0).sum())
        print(f"  {len(target_actions):,} reposts; {active:,}/{len(user_ids):,} "
              f"users still active")
        if active == 0:
            print("  no user survives into t+1; nDCG undefined. Skipping this pair.")
            continue

        print(f"grid search: {len(alphas)} alphas x {len(deltas)} deltas")
        _, _, results = grid_search(
            fx, target, alphas=alphas, deltas=deltas, progress=False
        )
        per_pair.append(results)
        del fx, index, target_actions

    if not per_pair:
        print("\nNo usable window pair. Nothing fitted.")
        return

    merged = aggregate(per_pair)

    print()
    print(format_grid(merged, "tash_index"))
    print()
    print(format_grid(merged, "tai_score"))
    print()
    print(format_baselines(merged))

    if args.save:
        payload = [
            {"metric": r.metric, "alpha": r.alpha,
             "delta_seconds": r.delta.total_seconds(), "ndcg": r.ndcg,
             "ndcg_std": r.ndcg_std, "n_pairs": r.n_pairs,
             "per_pair": list(r.per_pair)}
            for r in merged
        ]
        Path(args.save).write_text(json.dumps(payload, indent=2))
        print(f"\nfull surface written to {args.save}")

    print()
    print("=" * 72)
    print(f"RESULT: fitted on the dataset ({len(per_pair)} window pairs)")
    print("=" * 72)
    for metric, prior in (
        ("tash_index", "alpha=0.5, delta=14d"),
        ("tai_score", "alpha=0.6, delta=18d"),
    ):
        near, best = plateau(merged, metric)
        print(f"  {metric:<12} alpha={best.alpha:.1f}  delta={_fmt(best.delta)}")
        print(f"               nDCG@10={best.ndcg.get(10, 0):.4f}  "
              f"nDCG@100={best.ndcg.get(100, 0):.4f}  "
              f"nDCG@1000={best.ndcg.get(1000, 0):.4f}")
        if best.n_pairs > 1:
            spread = ", ".join(f"{v:.3f}" for v in best.per_pair)
            print(f"               per-pair nDCG@100: {spread}")
            print(f"               std across pairs: {best.std:.4f}")
        print(f"               plateau: {len(near)} settings within 0.01 nDCG")
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
