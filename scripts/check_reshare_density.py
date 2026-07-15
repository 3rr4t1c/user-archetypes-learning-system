#!/usr/bin/env python3
"""Measure whether the reshare graph is dense enough to support the super-spreader axis.

Why this exists
---------------
The super-spreader indicators (reshare yield, virality, TASH-index) all describe how
often, how fast and how consistently a user's *own posts* get reshared. They are only
estimable if authors actually accumulate reshares within an analysis window.

The TASH-index is the binding constraint. It is an EMA over per-slot social h-indices,
and an h-index of h requires h posts each with >= h reshares *inside the slot*. If the
median author receives ~1 reshare per window, the h-index is capped at 1 for everyone
and TASH carries no information -- and no choice of alpha or delta can repair that,
because it is a property of the data, not of the parameters.

This script measures, for one window, the reshares-received distribution and the
resulting h-index ceiling. It attributes each reshare to the original author by parsing
the DID out of the AT-URI (see arles.arles.author_of_uri), which works for 100% of
reposts, rather than requiring the original post to be present in the sample (0.09%).

It streams the CSV with the stdlib csv module and holds only per-window counters, so it
runs in bounded memory on the full 3.7 GB file.

Usage
-----
    python scripts/check_reshare_density.py data/bluesky_sampled_clean_full_sorted.csv
    python scripts/check_reshare_density.py <csv> --days 5 --start 2024-10-12

Reading the result
------------------
    "authors with h>=2" near 0%   -> TASH cannot discriminate; the super-spreader axis
                                     is not estimable at this sampling rate.
    a broad h-index distribution  -> TASH is usable; calibrate delta so that a typical
                                     slot contains a few reshares per active author.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

# Work regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arles.arles import MalformedActionError, author_of_uri, parse_timestamp  # noqa: E402

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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("--days", type=float, default=5.0,
                    help="window length in days (default: 5, the paper's window)")
    ap.add_argument("--start", default=None, type=parse_cli_datetime,
                    help="window start: YYYY-MM-DD or full ISO timestamp "
                         "(default: first timestamp in the file)")
    ap.add_argument("--scan-all", action="store_true",
                    help="do not stop at the end of the window. Required if the file "
                         "is not sorted by created_at; slower on large files.")
    ap.add_argument("--report-every", type=int, default=2_000_000)
    args = ap.parse_args()

    # reshares per (author, post) inside the window
    per_post = defaultdict(int)
    post_author = {}

    start = args.start
    end = start + timedelta(days=args.days) if start else None
    n_rows = n_reposts = n_window = n_bad = 0
    prev_ts = None
    unsorted_seen = False

    with open(args.csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            if n_rows % args.report_every == 0:
                print(f"  ... {n_rows:,} rows, {n_window:,} reposts in window",
                      file=sys.stderr, flush=True)

            if row.get("activity_type") != "repost":
                continue
            n_reposts += 1

            orig = row.get("original_action_id")
            if not orig:
                continue

            try:
                ts = parse_timestamp(row["created_at"])
            except MalformedActionError:
                n_bad += 1
                continue

            # The early-exit below assumes the file is ordered by created_at.
            # Verify rather than trust the filename.
            if prev_ts is not None and ts < prev_ts and not unsorted_seen:
                unsorted_seen = True
                print(f"WARNING: rows are not sorted by created_at (saw {ts.isoformat()} "
                      f"after {prev_ts.isoformat()}).\n"
                      f"         Re-run with --scan-all or the window will be truncated.",
                      file=sys.stderr, flush=True)
            prev_ts = ts

            if start is None:
                start = ts
                end = start + timedelta(days=args.days)
            if ts < start:
                continue
            if ts >= end:
                if not args.scan_all:
                    break
                continue

            author = author_of_uri(orig)
            if author is None:
                continue
            per_post[orig] += 1
            post_author[orig] = author
            n_window += 1

    if not per_post:
        print("No reposts found in the window.")
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
    print(f"rows scanned      : {n_rows:,}")
    print(f"reposts in window : {n_window:,}   (unparseable timestamps: {n_bad:,})")
    print(f"distinct authors  : {n_auth:,}")
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
    for k in (2, 3, 5):
        n = sum(1 for h in hs.values() if h >= k)
        print(f"  h >= {k}: {n:,} ({100*n/n_auth:.2f}%)")
    print()
    if sum(1 for h in hs.values() if h >= 2) / n_auth < 0.01:
        print("VERDICT: TASH-index is degenerate on this file -- <1% of authors can even")
        print("         reach h=2. The super-spreader axis is not estimable here, and no")
        print("         (alpha, delta) fixes it. Use a denser sample or drop the axis.")
    else:
        print("VERDICT: h-index has spread; TASH is usable. Pick delta so a typical slot")
        print("         holds a few reshares per active author, and delta << 5-day window.")


if __name__ == "__main__":
    main()
