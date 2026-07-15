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
100 real reshares shows ~1, so "h posts with >= h reshares each" cannot form. That is
why this is worth running against the full archive rather than a sample.

Reshares are attributed to the original author by parsing the DID out of the AT-URI
(arles.arles.author_of_uri), which resolves 100% of reposts, instead of requiring the
original post to be present in the file (0.09% in the sampled data).

Cost
----
Bounded memory (per-window counters only), and only the files overlapping the window
are opened -- with a binary search to the window's start inside the first of them. On
the 146 GB archive a 5-day window reads roughly one file's worth of bytes, not 146 GB.

Usage
-----
    # a directory of sequential CSVs, or a single CSV
    python scripts/check_reshare_density.py /Volumes/Uniform/bluesky_full --days 5 --start 2024-10-12
    python scripts/check_reshare_density.py data/bluesky_sampled_clean_small.csv

Reading the result
------------------
    ~all authors at h=1  -> TASH is degenerate: it collapses to "did I get reshared at
                            all this slot". The super-spreader axis needs reshare counts
                            instead, or a denser sample.
    h spread over 1..10+ -> TASH is a real h-index; choose delta so a typical slot holds
                            a few reshares per active author, with delta << the window.
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
    DEFAULT_MAX_DISORDER,
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="a CSV file, or a directory of sequential CSVs")
    ap.add_argument("--days", type=float, default=5.0,
                    help="window length in days (default: 5, the paper's window)")
    ap.add_argument("--start", default=None, type=parse_cli_datetime,
                    help="window start: YYYY-MM-DD or full ISO timestamp "
                         "(default: first timestamp in the archive)")
    ap.add_argument("--no-seek", action="store_true",
                    help="scan from the start of each file instead of binary-searching "
                         "to the window. Slower; use if you suspect rows are unsorted.")
    ap.add_argument("--max-disorder-hours", type=float, default=24.0,
                    help="how far created_at may deviate from ingestion order "
                         "(default: 24; measured max on the archive is 16h). Bounds "
                         "how far the reader seeks and when it may stop.")
    ap.add_argument("--index-cache", default=None,
                    help="where to cache the per-file time index "
                         "(default: <path>/.arles_index.json when path is a directory)")
    args = ap.parse_args()

    t0 = time.time()

    files = discover_files(args.path)
    print(f"archive           : {args.path}")
    print(f"files             : {len(files)}")
    total_gb = sum(Path(p).stat().st_size for p in files) / 1e9
    print(f"total size        : {total_gb:.1f} GB")

    cache = args.index_cache
    if cache is None and Path(args.path).is_dir():
        cache = str(Path(args.path) / ".arles_index.json")

    print("\nindexing (2 rows per file, not a full scan):")
    spans = build_index(files, cache_path=cache, verbose=True)
    check_contiguous(spans)
    print(f"  archive spans   : {spans[0].start.isoformat()[:19]} -> "
          f"{spans[-1].end.isoformat()[:19]}")

    start = args.start or spans[0].start
    end = start + timedelta(days=args.days)

    # reshares per (author, post) inside the window
    per_post = defaultdict(int)
    post_author = {}
    n_rows = n_reposts = n_window = n_unattributed = 0

    for row in iter_window(
        files, start, end, spans=spans, progress=True, seek=not args.no_seek,
        max_disorder=timedelta(hours=args.max_disorder_hours),
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
        n_window += 1

    elapsed = time.time() - t0

    if not per_post:
        print("\nNo reposts found in the window.")
        return

    by_author = defaultdict(list)
    for post, cnt in per_post.items():
        by_author[post_author[post]].append(cnt)

    received = {a: sum(c) for a, c in by_author.items()}
    hs = {a: h_index(c) for a, c in by_author.items()}

    vals = sorted(received.values())
    n_auth = len(vals)

    def pct(p):
        return vals[min(int(p * n_auth), n_auth - 1)]

    print()
    print(f"window            : {start.isoformat()} -> {end.isoformat()} ({args.days} days)")
    print(f"rows in window    : {n_rows:,}")
    print(f"reposts in window : {n_window:,}   (unattributable URIs: {n_unattributed:,})")
    print(f"distinct authors  : {n_auth:,}")
    print(f"elapsed           : {elapsed/60:.1f} min")
    print()
    print("reshares received per author")
    print(f"  mean {sum(vals)/n_auth:.2f} | median {pct(.5)} | p90 {pct(.9)} "
          f"| p99 {pct(.99)} | max {vals[-1]}")
    print()
    print("h-index per author (whole window as ONE slot = the best case for TASH;")
    print("any smaller delta can only lower it)")
    dist = defaultdict(int)
    for h in hs.values():
        dist[h] += 1
    for h in sorted(dist):
        print(f"  h = {h}: {dist[h]:,} authors ({100*dist[h]/n_auth:.2f}%)")
    for k in (2, 3, 5, 10):
        n = sum(1 for h in hs.values() if h >= k)
        print(f"  h >= {k}: {n:,} ({100*n/n_auth:.2f}%)")
    print()

    # The verdict keys on the share pinned at h=1, not on the h>=2 tail. An earlier
    # version passed anything with >1% at h>=2, which called 97.9%-at-h=1 "usable" --
    # far too lenient: if almost every author is pinned at 1 in the best case, then at
    # delta << window the per-slot h is {0,1} and TASH is just "reshared this slot?".
    share_h1 = 100 * dist.get(1, 0) / n_auth
    share_h3 = 100 * sum(1 for h in hs.values() if h >= 3) / n_auth
    if share_h1 > 90:
        print(f"VERDICT: DEGENERATE. {share_h1:.1f}% of authors are pinned at h=1 even")
        print( "         with the whole window as one slot. At any usable delta the")
        print( "         per-slot h-index is {0,1} and TASH reduces to 'was I reshared")
        print( "         at all this slot'. No (alpha, delta) fixes this. Build the")
        print( "         super-spreader axis from reshare counts, which here span")
        print(f"         1..{vals[-1]}, or use a denser sample.")
    elif share_h3 < 5:
        print(f"VERDICT: WEAK. {share_h1:.1f}% at h=1 and only {share_h3:.2f}% reach h>=3.")
        print( "         TASH will separate a thin tail and say little about the rest.")
    else:
        print(f"VERDICT: USABLE. h is spread ({share_h1:.1f}% at h=1, {share_h3:.2f}% at")
        print( "         h>=3). Choose delta so a typical slot holds a few reshares per")
        print( "         active author, with delta << the analysis window.")


if __name__ == "__main__":
    main()
