#!/usr/bin/env python3
"""Measure whether the reshare graph is dense enough to support the super-spreader axis.

Why this exists
---------------
The super-spreader indicators (reshare yield, virality, TASH-index) all describe how
often, how fast and how consistently a user's *own posts* get reshared. They are only
estimable if authors actually accumulate reshares within an analysis window.

The TASH-index is the binding constraint. It is an EMA over per-slot social h-indices,
and an h-index of h requires h posts each with >= h reshares *inside the slot*. If the
median author collects ~1 reshare per window, h is pinned at 1 for everyone and TASH
carries no information -- and no choice of alpha or delta repairs that, because it is a
property of the data, not of the parameters.

Sampling does not merely shrink an h-index, it destroys it: at a 1% sample, a post with
100 real reshares shows ~1, so "h posts with >= h reshares each" cannot form. Measured:
on the ~1% sample the E1 pre-event window gives mean 1.79 reshares/author and 2.05% of
authors reaching h>=2; on the full archive the same window gives mean 14.46 and 24.24%.
The axis is estimable on the full archive and not on the sample.

Reshares are attributed to the original author by parsing the DID out of the AT-URI
(arles.arles.author_of_uri), which resolves 100% of reposts, instead of requiring the
original post to be present in the file (0.09% in the sampled data).

Why --windows matters
---------------------
Figure 8 does not plot one window: it plots 7 consecutive 5-day windows per event
(E1: Aug 25 -> Sep 24, E2: Oct 12 -> Nov 11). Density is not constant across them --
E1's span brackets the Brazil ban, during which the platform grew sharply. If density
swings across the 7 windows then the h-index distribution swings too, TASH means
something different in each, and the per-window min-max rescaling compounds it. So
measure every window, not just the pre-event one, and read the SUMMARY table for how
much the ground shifts underneath the figure.

Cost
----
Bounded memory: one window's counters at a time, so N windows cost N passes rather than
N times the RAM. Only files that can overlap a window are opened, with a binary search
to its start. A 5-day window on the 146 GB archive reads ~1 file and takes ~1 minute.

Usage
-----
    # the two Figure 8 spans, 7 windows each
    python scripts/check_reshare_density.py /Volumes/Uniform/bluesky_full \\
        --start 2024-08-25 --days 5 --windows 7
    python scripts/check_reshare_density.py /Volumes/Uniform/bluesky_full \\
        --start 2024-10-12 --days 5 --windows 7

    # a single window, or a single file
    python scripts/check_reshare_density.py data/bluesky_sampled_clean_small.csv

Reading the result
------------------
    ~all authors at h=1  -> TASH is degenerate: it collapses to "did I get reshared at
                            all this slot". The super-spreader axis needs reshare counts
                            instead, or a denser sample.
    h spread over 1..10+ -> TASH is a real h-index; choose delta so a typical slot holds
                            a few reshares per active author, with delta << the window.
    density varying a lot across windows -> scores are not comparable between them.
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arles.arles import MalformedActionError, author_of_uri, parse_timestamp  # noqa: E402
from arles.streaming import (  # noqa: E402
    build_index,
    check_contiguous,
    discover_files,
    iter_window,
)

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_cli_datetime(value):
    """Parse a --start argument: a bare date (2024-10-12) or a full timestamp.

    Deliberately more permissive than arles.arles.parse_timestamp. The two have
    different contracts:

      * A *data row* with no time component means the timestamp column lost
        information, and must raise rather than be silently rounded to midnight.
        Tolerating "%Y-%m-%d" there is precisely the sloppiness that let the
        original datetime.now() bug hide.
      * A *command-line argument* is typed by a human, where "2024-10-12"
        unambiguously means midnight UTC.

    So the leniency lives here, at the CLI boundary, and never touches the parser
    applied to the data.
    """
    s = value.strip()
    if _DATE_ONLY_RE.match(s):
        s = f"{s} 00:00:00+00:00"
    try:
        return parse_timestamp(s)
    except MalformedActionError:
        raise argparse.ArgumentTypeError(
            f"invalid --start {value!r}: expected YYYY-MM-DD or a full ISO timestamp"
        )


def h_index(counts):
    """Largest h such that h posts have >= h reshares each."""
    ordered = sorted(counts, reverse=True)
    h = 0
    for i, c in enumerate(ordered, start=1):
        if c >= i:
            h = i
        else:
            break
    return h


def measure_window(files, spans, start, end, seek, max_disorder):
    """Reshare statistics for one window. Holds only this window's counters."""
    per_post = defaultdict(int)
    post_author = {}
    n_rows = n_reposts = n_attributed = n_unattributed = 0

    for row in iter_window(
        files, start, end, spans=spans, progress=True, seek=seek,
        max_disorder=max_disorder,
    ):
        n_rows += 1
        if row.get("activity_type") != "repost":
            continue
        n_reposts += 1
        orig = row.get("original_action_id")
        if not orig:
            continue
        author = author_of_uri(orig)
        if author is None:
            n_unattributed += 1
            continue
        per_post[orig] += 1
        post_author[orig] = author
        n_attributed += 1

    by_author = defaultdict(list)
    for post, cnt in per_post.items():
        by_author[post_author[post]].append(cnt)

    return {
        "start": start,
        "end": end,
        "n_rows": n_rows,
        "n_reposts": n_attributed,
        "n_unattributed": n_unattributed,
        "received": {a: sum(c) for a, c in by_author.items()},
        "h": {a: h_index(c) for a, c in by_author.items()},
    }


def report_window(res, verbose=True):
    received, hs = res["received"], res["h"]
    if not received:
        print("\nNo reposts found in the window.")
        return None

    vals = sorted(received.values())
    n_auth = len(vals)

    def pct(p):
        return vals[min(int(p * n_auth), n_auth - 1)]

    print()
    print(f"window            : {res['start'].isoformat()} -> {res['end'].isoformat()}")
    print(f"rows in window    : {res['n_rows']:,}")
    print(f"reposts in window : {res['n_reposts']:,}   "
          f"(unattributable URIs: {res['n_unattributed']:,})")
    print(f"distinct authors  : {n_auth:,}")
    print()
    print("reshares received per author")
    print(f"  mean {sum(vals)/n_auth:.2f} | median {pct(.5)} | p90 {pct(.9)} "
          f"| p99 {pct(.99)} | max {vals[-1]}")

    dist = defaultdict(int)
    for h in hs.values():
        dist[h] += 1

    if verbose:
        print()
        print("h-index per author (whole window as ONE slot = the best case for TASH;")
        print("any smaller delta can only lower it)")
        for h in sorted(dist):
            print(f"  h = {h}: {dist[h]:,} authors ({100*dist[h]/n_auth:.2f}%)")
        for k in (2, 3, 5, 10):
            n = sum(1 for h in hs.values() if h >= k)
            print(f"  h >= {k}: {n:,} ({100*n/n_auth:.2f}%)")

    return {
        "authors": n_auth,
        "reposts": res["n_reposts"],
        "mean": sum(vals) / n_auth,
        "median": pct(.5),
        "max": vals[-1],
        "share_h1": 100 * dist.get(1, 0) / n_auth,
        "share_h2": 100 * sum(1 for h in hs.values() if h >= 2) / n_auth,
        "share_h3": 100 * sum(1 for h in hs.values() if h >= 3) / n_auth,
        "start": res["start"],
    }


def verdict(share_h1, share_h3, max_received):
    # Keys on the share pinned at h=1, not on the h>=2 tail. An earlier version passed
    # anything with >1% at h>=2, which called 97.9%-at-h=1 "usable" -- far too lenient:
    # if almost every author is pinned at 1 in the best case, then at delta << window
    # the per-slot h is {0,1} and TASH is just "reshared this slot?".
    if share_h1 > 90:
        print(f"VERDICT: DEGENERATE. {share_h1:.1f}% of authors are pinned at h=1 even")
        print( "         with the whole window as one slot. At any usable delta the")
        print( "         per-slot h-index is {0,1} and TASH reduces to 'was I reshared")
        print( "         at all this slot'. No (alpha, delta) fixes this. Build the")
        print( "         super-spreader axis from reshare counts, which here span")
        print(f"         1..{max_received}, or use a denser sample.")
    elif share_h3 < 5:
        print(f"VERDICT: WEAK. {share_h1:.1f}% at h=1 and only {share_h3:.2f}% reach h>=3.")
        print( "         TASH will separate a thin tail and say little about the rest.")
    else:
        print(f"VERDICT: USABLE. h is spread ({share_h1:.1f}% at h=1, {share_h3:.2f}% at")
        print( "         h>=3). Choose delta so a typical slot holds a few reshares per")
        print( "         active author, with delta << the analysis window.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="a CSV file, or a directory of sequential CSVs")
    ap.add_argument("--days", type=float, default=5.0,
                    help="window length in days (default: 5, the paper's window)")
    ap.add_argument("--windows", type=int, default=1,
                    help="number of consecutive windows to measure (default: 1). "
                         "Figure 8 uses 7 per event.")
    ap.add_argument("--start", default=None, type=parse_cli_datetime,
                    help="window start: YYYY-MM-DD or full ISO timestamp "
                         "(default: first timestamp in the archive)")
    ap.add_argument("--no-seek", action="store_true",
                    help="scan from the start of each file instead of binary-searching "
                         "to the window. Slower; use if you suspect rows are unsorted.")
    ap.add_argument("--max-disorder-hours", type=float, default=24.0,
                    help="how far created_at may deviate from ingestion order "
                         "(default: 24; observed max on the archive is 18.6h). Bounds "
                         "how far the reader seeks and when it may stop.")
    ap.add_argument("--index-cache", default=None,
                    help="where to cache the per-file time index "
                         "(default: <path>/.arles_index.json when path is a directory)")
    args = ap.parse_args()

    t0 = time.time()

    files = discover_files(args.path)
    print(f"archive           : {args.path}")
    print(f"files             : {len(files)}")
    print(f"total size        : {sum(Path(p).stat().st_size for p in files) / 1e9:.1f} GB")

    cache = args.index_cache
    if cache is None and Path(args.path).is_dir():
        cache = str(Path(args.path) / ".arles_index.json")

    print("\nindexing (2 rows per file, not a full scan):")
    spans = build_index(files, cache_path=cache, verbose=True)
    check_contiguous(spans)
    print(f"  archive spans   : {spans[0].start.isoformat()[:19]} -> "
          f"{spans[-1].end.isoformat()[:19]}")

    first = args.start or spans[0].start
    max_disorder = timedelta(hours=args.max_disorder_hours)
    step = timedelta(days=args.days)

    summaries = []
    for w in range(args.windows):
        start = first + w * step
        end = start + step
        if args.windows > 1:
            print(f"\n{'=' * 72}")
            print(f"WINDOW {w + 1}/{args.windows}: {start.isoformat()[:19]} -> "
                  f"{end.isoformat()[:19]}")
            print("=" * 72)
        res = measure_window(files, spans, start, end, not args.no_seek, max_disorder)
        s = report_window(res, verbose=(args.windows == 1))
        if s:
            summaries.append(s)
        del res

    print(f"\nelapsed           : {(time.time() - t0)/60:.1f} min")

    if not summaries:
        return

    if len(summaries) > 1:
        print()
        print("SUMMARY across windows")
        print(f"  {'window start':<12} {'reposts':>10} {'authors':>9} {'mean':>7} "
              f"{'median':>7} {'max':>7} {'h=1 %':>7} {'h>=2 %':>7} {'h>=3 %':>7}")
        for s in summaries:
            print(f"  {s['start'].strftime('%d %b'):<12} {s['reposts']:>10,} "
                  f"{s['authors']:>9,} {s['mean']:>7.2f} {s['median']:>7} "
                  f"{s['max']:>7,} {s['share_h1']:>7.1f} {s['share_h2']:>7.2f} "
                  f"{s['share_h3']:>7.2f}")

        means = [s["mean"] for s in summaries]
        swing = max(means) / min(means) if min(means) else float("inf")
        print()
        print(f"  density swing across windows: {swing:.1f}x "
              f"(mean reshares/author {min(means):.2f} -> {max(means):.2f})")
        if swing > 1.5:
            print()
            print("  CAUTION: the reshare density is not constant across the windows")
            print("  Figure 8 compares. Each window's scores are min-max rescaled")
            print("  within that window's own population, so a score of 0.2 in a sparse")
            print("  window and 0.2 in a dense one are not the same quantity. Fit the")
            print("  scaler and PCA once on the pooled windows and freeze them, or the")
            print("  trend line partly tracks how much data each window happened to")
            print("  contain.")

    print()
    verdict(summaries[0]["share_h1"], summaries[0]["share_h3"], summaries[0]["max"])


if __name__ == "__main__":
    main()
