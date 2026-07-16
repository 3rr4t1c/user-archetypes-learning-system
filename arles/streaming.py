"""Windowed, bounded-memory reading of the multi-file action archive.

The full archive is ~146 GB across 13 sequential CSV files on an external drive. A
5-day analysis window is ~2-3 GB of that, so reading the archive end-to-end wastes
about 98% of the I/O -- which matters a great deal when the platter is behind USB.

Three facts make it cheap instead:

1. The files are contiguous and ordered: file N ends exactly where file N+1 begins
   (processed_enrico0: Aug 23 -> Sep 10, processed_enrico1: Sep 10 -> Sep 21, ...).
2. So a window's first/last timestamps identify which files can possibly contain it,
   from an index built by reading two rows per file rather than all of them.
3. Rows within a file are *nearly* ordered by created_at, so the window's start can be
   found by binary search over byte offsets, and reading can stop past its end.

Together: 146 GB -> the ~1 file and ~3 GB that actually hold the window.

The "nearly" in (3) is the important caveat
-------------------------------------------
The archive is ordered by INGESTION, not by created_at. created_at is client-supplied
and some clients report local time labelled as UTC, so a row's timestamp can sit up to
~16 h away from its neighbours' (measured: the largest backward deviation from the
running maximum over 1M rows of processed_enrico0.csv is exactly 16.00 h, and world
timezones span +/-14 h, +/-16 h with DST).

Every shortcut is therefore widened by `max_disorder`, and the stop condition keys on
the running maximum rather than the current row. Assuming strict order was not a
theoretical error: stopping at the first row with ts >= end truncated a 53,103-row
window to 115 rows -- and still printed a confident verdict.

Ordering note: the files must be sorted *numerically*. Lexicographically,
"processed_enrico10.csv" sorts before "processed_enrico2.csv", which would silently
interleave September into January.
"""

import csv
import io
import json
import os
import re
import sys
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .actions import MalformedActionError, parse_timestamp

# Rows can carry long post text; the stdlib default (128 KB) is generous but the
# archive has been seen to exceed it on pathological rows.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

_TRAILING_INT_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def numeric_key(path: str):
    """Sort key that orders processed_enrico2.csv before processed_enrico10.csv."""
    m = _TRAILING_INT_RE.search(os.path.basename(path))
    return (0, int(m.group(1)), path) if m else (1, 0, path)


def discover_files(path: str) -> List[str]:
    """Return the CSV files at `path` (a file, or a directory of them) in order."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    found = [
        os.path.join(path, n) for n in os.listdir(path) if n.lower().endswith(".csv")
    ]
    if not found:
        raise FileNotFoundError(f"no .csv files in {path}")
    return sorted(found, key=numeric_key)


@dataclass
class FileSpan:
    """The time range covered by one CSV file."""

    path: str
    start: datetime
    end: datetime
    size: int

    def overlaps(self, lo: datetime, hi: datetime) -> bool:
        return self.start < hi and self.end >= lo


def _header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        return next(csv.reader(f))


def _first_row(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return next(reader)


def _last_row(
    path: str,
    n_fields: int,
    ts_index: int,
    tail_bytes: int = 1 << 20,
    max_tail: int = 1 << 26,
) -> Optional[List[str]]:
    """Read the final complete row without scanning the file.

    A single seek to the end is what keeps indexing a 12 GB file to milliseconds.

    Two things make this fiddly, and an earlier version got both wrong:

    * Post text contains newlines, so splitting the tail on "\\n" does not yield rows.
      The tail is parsed as CSV instead, after discarding the first fragment (a blind
      seek lands mid-row), which lets the reader resynchronise on quoting.
    * A truncated fragment still *parses* -- as a one-field row. Accepting the first
      thing that parsed made processed_enrico0.csv report its end timestamp as equal
      to its start, collapsing its span so that no file matched the window at all.
      A candidate must now have the full field count and a parseable timestamp.

    Falls back to progressively larger tails; returns None if nothing valid is found.
    """
    size = os.path.getsize(path)
    tail = tail_bytes

    while True:
        with open(path, "rb") as f:
            f.seek(max(0, size - tail))
            blob = f.read()

        text = blob.decode("utf-8", errors="replace")
        if size > tail:  # drop the partial first row
            nl = text.find("\n")
            text = text[nl + 1 :] if nl >= 0 else ""

        best: Optional[List[str]] = None
        try:
            for row in csv.reader(io.StringIO(text, newline="")):
                if len(row) == n_fields and row[ts_index]:
                    try:
                        parse_timestamp(row[ts_index])
                    except MalformedActionError:
                        continue
                    best = row
        except csv.Error:
            pass  # ragged tail; try a bigger one

        if best is not None:
            return best
        if tail >= max_tail or tail >= size:
            return None
        tail = min(tail * 8, max(size, tail + 1))


def build_index(
    paths: Sequence[str],
    ts_column: str = "created_at",
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> List[FileSpan]:
    """Map each file to its [start, end] timestamps by reading two rows from it.

    Cached to JSON, keyed on (path, size, mtime): re-probing 13 files on a sleeping
    external drive is slow enough to be worth never doing twice.
    """
    cache: Dict[str, dict] = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}

    spans: List[FileSpan] = []
    dirty = False

    for path in paths:
        stat = os.stat(path)
        key = os.path.abspath(path)
        hit = cache.get(key)
        if hit and hit.get("size") == stat.st_size and hit.get("mtime") == stat.st_mtime:
            spans.append(
                FileSpan(
                    path,
                    datetime.fromisoformat(hit["start"]),
                    datetime.fromisoformat(hit["end"]),
                    stat.st_size,
                )
            )
            continue

        if verbose:
            print(f"  indexing {os.path.basename(path)} ...", end="", flush=True)

        header = _header(path)
        if ts_column not in header:
            raise ValueError(f"{path}: no '{ts_column}' column (found {header})")
        i = header.index(ts_column)

        start = parse_timestamp(_first_row(path)[i])
        last = _last_row(path, n_fields=len(header), ts_index=i)
        if last is None:
            # Refuse to guess: a collapsed span silently excludes the file from every
            # window that should have matched it.
            raise ValueError(
                f"{path}: could not read a complete final row to determine the file's "
                f"end timestamp. Re-run with a larger tail or check the file is intact."
            )
        end = parse_timestamp(last[i])
        if end < start:
            raise ValueError(
                f"{path}: end timestamp {end.isoformat()} precedes start "
                f"{start.isoformat()}; the file is not sorted by {ts_column}."
            )

        spans.append(FileSpan(path, start, end, stat.st_size))
        cache[key] = {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        dirty = True
        if verbose:
            print(f" {start.isoformat()[:19]} -> {end.isoformat()[:19]}", flush=True)

    if cache_path and dirty:
        try:
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=2)
        except OSError:
            pass  # a cache we cannot write is not worth failing over

    spans.sort(key=lambda s: s.start)
    return spans


def check_contiguous(spans: Sequence[FileSpan], verbose: bool = True) -> bool:
    """Warn if the archive has gaps or overlaps.

    Early exit assumes global time order; if that does not hold, a window could be
    silently truncated. Better to say so than to return a confidently wrong number.
    """
    ok = True
    for a, b in zip(spans, spans[1:]):
        if b.start < a.start:
            ok = False
            if verbose:
                print(
                    f"WARNING: {os.path.basename(b.path)} starts before "
                    f"{os.path.basename(a.path)}; archive is not ordered.",
                    file=sys.stderr,
                )
    return ok


def _probe_row(
    f,
    offset: int,
    ts_index: int,
    n_fields: int,
    max_scan: int = 1 << 21,
    skip_partial: bool = True,
) -> Optional[Tuple[int, datetime]]:
    """Find the first genuine row after `offset`; return (its offset, its timestamp).

    With `skip_partial` (the default, and what a binary search needs), the line
    containing `offset` is discarded first: an arbitrary byte almost always lands
    mid-row, and half a row is not a row. Pass skip_partial=False when `offset` is
    already known to be a row boundary and that row itself is wanted.

    Post text contains newlines, so a "line" is not a row either. A candidate is only
    accepted if it splits into exactly `n_fields` and its timestamp parses -- a
    fragment of prose essentially never does both.

    That validation is also what makes the returned offset safe to start a csv.reader
    from: it is a true row boundary, so the reader's quoting state begins clean.

    Returns None if no row is found within `max_scan` bytes.
    """
    f.seek(offset)
    if offset > 0 and skip_partial:
        f.readline()  # discard the row we almost certainly landed inside

    scanned = 0
    while scanned < max_scan:
        pos = f.tell()
        line = f.readline()
        if not line:
            return None
        scanned += len(line)
        try:
            row = next(csv.reader([line.decode("utf-8", errors="replace")]))
        except (csv.Error, StopIteration):
            continue
        if len(row) != n_fields:
            continue
        try:
            return pos, parse_timestamp(row[ts_index])
        except (MalformedActionError, IndexError):
            continue
    return None


def _seek_before(f, target: datetime, ts_index: int, n_fields: int, size: int) -> int:
    """Byte offset of a genuine row at or before the first one with ts >= target.

    Deliberately conservative: it may land early, never late. Callers skip rows below
    `target` anyway, so landing early costs a little I/O, whereas landing late would
    silently drop data.

    The previous version degraded to returning 0 -- correct, but it read all 12 GB of
    the file instead of seeking to the window. Two causes, both fixed here: probes
    landing on newline fragments failed to parse and nudged `lo` forward, sometimes
    past `hi`, ending the search with nothing found; and `lo` was advanced past the
    probe's line rather than kept within the bracket.
    """
    lo, hi = 0, size
    best = 0
    while lo < hi:
        mid = (lo + hi) // 2
        probe = _probe_row(f, mid, ts_index, n_fields)
        if probe is None:
            # Nothing readable between mid and the scan limit: the answer, if any,
            # lies to the left.
            hi = mid
            continue
        pos, ts = probe
        if ts < target:
            if pos > best:
                best = pos
            # Always make progress, and never step outside the bracket.
            lo = min(max(mid + 1, pos + 1), hi)
            if lo >= hi:
                break
        else:
            hi = mid
    return best


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


#: How far a row's created_at may deviate from its ingestion position.
#:
#: The archive is ordered by ingestion, NOT by created_at, because created_at is
#: client-supplied. Measured over 1M rows of processed_enrico0.csv, the largest
#: backward deviation from the running maximum is exactly 16.00 h -- consistent with
#: clients reporting local time labelled as UTC (world timezones span +/-14 h, +/-16 h
#: once DST is involved).
#:
#: Everything here that skips data is bounded by this constant. Getting it wrong
#: truncates windows silently, so it is deliberately generous.
DEFAULT_MAX_DISORDER = timedelta(hours=24)


def iter_window(
    paths: Sequence[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    ts_column: str = "created_at",
    spans: Optional[Sequence[FileSpan]] = None,
    progress: bool = True,
    seek: bool = True,
    max_disorder: timedelta = DEFAULT_MAX_DISORDER,
) -> Iterator[dict]:
    """Yield rows with start <= created_at < end, as dicts, in bounded memory.

    Only files that can overlap the window are opened; within the first of them the
    start is located by binary search; reading stops once no later row can fall in
    the window.

    All three shortcuts are widened by `max_disorder`, because the archive is sorted
    by ingestion and created_at is client-supplied and can run up to ~16 h ahead of
    or behind its neighbours.

    An earlier version stopped at the *first* row with ts >= end. On the real archive
    that row arrived 6 h early, 119 rows in: the window returned 115 rows instead of
    53,103, and reported a confident verdict on 0.2% of the data. The stop condition
    is now the running maximum, which is provably safe given the bound: once
    max(created_at) seen so far exceeds end + max_disorder, no later row can still
    fall inside the window.

    Deviations larger than `max_disorder` are detected and warned about rather than
    silently dropping data.
    """
    if spans is None:
        spans = build_index(paths, ts_column=ts_column, verbose=progress)

    # Widen file selection: a file's indexed span comes from its first and last rows,
    # which under disorder do not bound its contents.
    lo = start - max_disorder if start is not None else None
    hi = end + max_disorder if end is not None else None
    relevant = [
        s
        for s in spans
        if (lo is None or s.end >= lo) and (hi is None or s.start < hi)
    ]
    if not relevant:
        if progress:
            print("No file covers the requested window.", file=sys.stderr)
        return

    total_bytes = sum(s.size for s in relevant)
    if progress:
        print(
            f"\n{len(relevant)} of {len(spans)} file(s) overlap the window "
            f"({_fmt_bytes(total_bytes)} of {_fmt_bytes(sum(s.size for s in spans))}):",
            file=sys.stderr,
        )
        for s in relevant:
            print(f"  {os.path.basename(s.path)}  {_fmt_bytes(s.size)}", file=sys.stderr)
        print(file=sys.stderr)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    running_max: Optional[datetime] = None
    worst_disorder = timedelta(0)
    disorder_warned = False

    for span in relevant:
        header = _header(span.path)
        if ts_column not in header:
            raise ValueError(f"{span.path}: no '{ts_column}' column")
        ts_index = header.index(ts_column)

        with open(span.path, "rb") as fb:
            offset = 0
            if seek and lo is not None and span.start < lo:
                offset = _seek_before(fb, lo, ts_index, len(header), span.size)
                if progress and offset:
                    print(
                        f"  seek: skipped {_fmt_bytes(offset)} "
                        f"({100 * offset / span.size:.1f}%) of "
                        f"{os.path.basename(span.path)} to reach the window",
                        file=sys.stderr,
                    )
            fb.seek(offset)

            remaining = span.size - offset
            bar = None
            if progress and tqdm is not None:
                bar = tqdm(
                    total=remaining,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=os.path.basename(span.path)[:24],
                    initial=0,
                    smoothing=0.05,
                )
            elif progress:
                print(
                    f"  reading {os.path.basename(span.path)} "
                    f"from offset {_fmt_bytes(offset)} ...",
                    file=sys.stderr,
                )

            text = io.TextIOWrapper(fb, encoding="utf-8", errors="replace", newline="")
            reader = csv.reader(text)
            consumed = 0
            stopped_early = False

            for row in reader:
                # Cheap byte accounting for the bar; exact enough for an ETA.
                if bar is not None:
                    consumed += sum(len(c) for c in row) + len(row) + 1
                    if consumed >= (1 << 22):
                        bar.update(consumed)
                        consumed = 0

                if len(row) <= ts_index:
                    continue
                if row[ts_index] == ts_column:  # header, if we started at offset 0
                    continue
                try:
                    ts = parse_timestamp(row[ts_index])
                except MalformedActionError:
                    continue

                # Track disorder against the running maximum, and complain if it
                # exceeds what the stop condition assumes.
                if running_max is None or ts > running_max:
                    running_max = ts
                else:
                    behind = running_max - ts
                    if behind > worst_disorder:
                        worst_disorder = behind
                        if behind > max_disorder and not disorder_warned:
                            disorder_warned = True
                            print(
                                f"WARNING: {os.path.basename(span.path)} has a row "
                                f"{behind} behind the running maximum of "
                                f"{ts_column}, exceeding max_disorder="
                                f"{max_disorder}. Rows may be missed; re-run with a "
                                f"larger --max-disorder-hours.",
                                file=sys.stderr,
                            )

                # Safe stop: given the disorder bound, once the newest timestamp seen
                # is past end + max_disorder, nothing later can land in the window.
                if hi is not None and running_max >= hi:
                    stopped_early = True
                    break

                if start is not None and ts < start:
                    continue
                if end is not None and ts >= end:
                    continue

                yield dict(zip(header, row))

            if bar is not None:
                bar.update(consumed)
                bar.close()
            if stopped_early:
                if progress:
                    print(
                        f"  passed window end + {max_disorder} inside "
                        f"{os.path.basename(span.path)}; skipping the rest of the "
                        f"archive. (largest observed disorder: {worst_disorder})",
                        file=sys.stderr,
                    )
                return
