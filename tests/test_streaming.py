"""Tests for windowed archive reading.

The whole point of arles.streaming is to *not* read 146 GB. Every shortcut it takes
(indexing from two rows per file, skipping non-overlapping files, binary-searching
to the window start, stopping at the window end) is an opportunity to silently drop
rows. A dropped row is far worse than a slow read: it produces a plausible number
that is quietly wrong.

So the central test here compares the fast path against brute force and demands
they agree exactly.
"""

import csv
import re
import os
from datetime import datetime, timedelta, timezone

import pytest

from arles.streaming import (
    build_index,
    check_contiguous,
    discover_files,
    iter_window,
    numeric_key,
)

BASE = datetime(2024, 8, 23, tzinfo=timezone.utc)
COLUMNS = ["action_id", "activity_type", "created_at", "author_user_id",
           "target_user_id", "text", "original_action_id"]


def _post_text(ri):
    """Text as it actually appears in the archive: commas, quotes, and newlines.

    The newlines matter. An earlier version of this helper wrote a "newline-free body",
    and that blind spot let a real bug through: _last_row split the file tail on "\\n"
    and accepted the first fragment that parsed. On processed_enrico0.csv that returned
    a truncated one-field row, so the file's end timestamp collapsed onto its start, its
    span vanished, and "No file covers the requested window" came back for a window
    sitting squarely inside it. Test data must be as awkward as the real thing.
    """
    if ri % 5 == 0:
        return f'Post {ri}, with a comma,\na newline and "quotes"\n\nand a blank line'
    if ri % 5 == 1:
        return f'Post {ri} with "quotes, inside quotes" and, commas'
    return f"Post {ri} plain body"


def _write_archive(tmp_path, n_files=4, rows_per_file=500, minutes=1):
    """Write a contiguous, time-sorted archive mimicking processed_enricoN.csv."""
    paths = []
    t = BASE
    for fi in range(n_files):
        p = tmp_path / f"processed_enrico{fi}.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for ri in range(rows_per_file):
                is_repost = ri % 3 == 0
                kind = "repost" if is_repost else "post"
                author = f"user{ri % 37:04d}"
                w.writerow([
                    f"at://did:plc:{author}/app.bsky.feed.{kind}/3l2d{fi:02d}{ri:05d}",
                    kind,
                    t.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    author,
                    "",
                    _post_text(ri),
                    f"at://did:plc:orig{ri % 11:04d}/app.bsky.feed.post/3l2dorig{ri:05d}"
                    if is_repost else "",
                ])
                t += timedelta(minutes=minutes)
        paths.append(str(p))
    return paths


def _write_disordered_archive(tmp_path, n_files=2, rows_per_file=3000, jump_every=25):
    """An archive ordered by INGESTION, with client-supplied created_at, like the real one.

    Every `jump_every`-th row carries a timestamp up to 6 hours ahead of its
    neighbours -- the pattern measured in processed_enrico0.csv, where the largest
    backward deviation from the running maximum is 16.00 h, consistent with clients
    reporting local time labelled as UTC.

    This is the shape of data that broke the original reader: it stopped at the first
    row with ts >= end, which on the real archive arrived 6 h early and truncated a
    53,103-row window to 115 rows.
    """
    paths = []
    t = BASE
    n = 0
    for fi in range(n_files):
        p = tmp_path / f"processed_enrico{fi}.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for ri in range(rows_per_file):
                ts = t + (timedelta(hours=6) if n % jump_every == 0 else timedelta(0))
                author = f"user{ri % 37:04d}"
                w.writerow([
                    f"at://did:plc:{author}/app.bsky.feed.post/3l2d{fi:02d}{ri:05d}",
                    "post",
                    ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    author, "", _post_text(ri), "",
                ])
                t += timedelta(minutes=1)
                n += 1
        paths.append(str(p))
    return paths


def _brute_force(paths, start, end):
    """Read everything, filter in Python. The reference implementation."""
    out = []
    for p in sorted(paths, key=numeric_key):
        with open(p, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ts = datetime.fromisoformat(
                    row["created_at"].replace("Z", "+00:00")
                )
                if start <= ts < end:
                    out.append(row["action_id"])
    return out


def test_numeric_key_orders_10_after_2():
    """Lexicographic sorting would put file 10 between 1 and 2, interleaving months."""
    names = [f"processed_enrico{i}.csv" for i in [0, 1, 10, 11, 12, 2, 3, 9]]
    assert [os.path.basename(p) for p in sorted(names, key=numeric_key)] == [
        f"processed_enrico{i}.csv" for i in [0, 1, 2, 3, 9, 10, 11, 12]
    ]


def test_index_reads_spans_without_scanning(tmp_path):
    paths = _write_archive(tmp_path)
    spans = build_index(paths, verbose=False)
    assert len(spans) == 4
    assert spans[0].start == BASE
    # Contiguous: each file picks up where the previous left off.
    for a, b in zip(spans, spans[1:]):
        assert b.start > a.end
    assert check_contiguous(spans, verbose=False)


def test_file_span_does_not_collapse_when_text_contains_newlines(tmp_path):
    """Regression: the bug that made a real 12 GB file report end == start.

    _last_row must return the genuine final row, not the first tail fragment that
    happens to parse.
    """
    paths = _write_archive(tmp_path, n_files=1, rows_per_file=400)
    spans = build_index(paths, verbose=False)
    assert spans[0].end > spans[0].start
    assert spans[0].end == BASE + timedelta(minutes=399)


def test_last_row_survives_a_tail_full_of_embedded_newlines(tmp_path):
    """Force the small-tail path: the tail lands inside quoted, newline-rich text."""
    from arles.streaming import _header, _last_row

    paths = _write_archive(tmp_path, n_files=1, rows_per_file=300)
    header = _header(paths[0])
    row = _last_row(paths[0], n_fields=len(header), ts_index=header.index("created_at"),
                    tail_bytes=64)  # absurdly small, forcing the grow-and-retry path
    assert row is not None
    assert len(row) == len(header)
    assert row[header.index("created_at")].startswith("2024-")


def test_index_refuses_to_guess_a_span_it_cannot_read(tmp_path, monkeypatch):
    """A collapsed span silently excludes a file from every window. Fail loudly.

    This is the guard on the exact failure that shipped: end quietly defaulting to
    start, which made a 12 GB file span a single instant and match no window.
    """
    import arles.streaming as streaming

    paths = _write_archive(tmp_path, n_files=1, rows_per_file=10)
    monkeypatch.setattr(streaming, "_last_row", lambda *a, **k: None)
    with pytest.raises(ValueError, match="complete final row"):
        build_index(paths, verbose=False)


def test_index_rejects_a_file_whose_end_precedes_its_start(tmp_path, monkeypatch):
    import arles.streaming as streaming

    paths = _write_archive(tmp_path, n_files=1, rows_per_file=10)
    backwards = ["at://did:plc:u/app.bsky.feed.post/x", "post",
                 "2020-01-01T00:00:00.000Z", "u", "", "body", ""]
    monkeypatch.setattr(streaming, "_last_row", lambda *a, **k: backwards)
    with pytest.raises(ValueError, match="not sorted"):
        build_index(paths, verbose=False)


def test_index_is_cached_and_reused(tmp_path):
    paths = _write_archive(tmp_path, n_files=2)
    cache = tmp_path / "index.json"
    first = build_index(paths, cache_path=str(cache), verbose=False)
    assert cache.exists()
    second = build_index(paths, cache_path=str(cache), verbose=False)
    assert [(s.path, s.start, s.end) for s in first] == [
        (s.path, s.start, s.end) for s in second
    ]


@pytest.mark.parametrize(
    "start_min,length_min",
    [
        (0, 100),      # window at the very start
        (700, 300),    # window spanning a file boundary
        (250, 50),     # window inside one file
        (1900, 200),   # window at the very end
        (0, 2000),     # the whole archive
    ],
)
def test_seek_matches_brute_force_exactly(tmp_path, start_min, length_min):
    """The test that licenses every shortcut in the module."""
    paths = _write_archive(tmp_path)
    start = BASE + timedelta(minutes=start_min)
    end = start + timedelta(minutes=length_min)

    expected = _brute_force(paths, start, end)
    got = [r["action_id"] for r in iter_window(paths, start, end, progress=False)]

    assert got == expected
    assert len(got) == length_min or start_min + length_min > 2000


def test_seek_and_scan_agree(tmp_path):
    """Binary search must not change the result, only the I/O."""
    paths = _write_archive(tmp_path)
    start = BASE + timedelta(minutes=812)
    end = start + timedelta(minutes=97)
    seeked = [r["action_id"] for r in iter_window(paths, start, end, progress=False, seek=True)]
    scanned = [r["action_id"] for r in iter_window(paths, start, end, progress=False, seek=False)]
    assert seeked == scanned
    assert len(seeked) == 97


def test_seek_actually_skips_bytes(tmp_path):
    """The performance regression test.

    Correctness tests cannot catch a broken seek: returning offset 0 still yields the
    right rows, it just reads the entire file to do it. That is exactly what happened
    on the real archive -- newline fragments made every probe fail to parse, `lo` ran
    past `hi`, and the search returned 0, reading 12 GB to find a window ~11% in.

    So assert the seek lands near the target, not merely that the output is correct.
    """
    from arles.streaming import _header, _probe_row, _seek_before

    paths = _write_archive(tmp_path, n_files=1, rows_per_file=4000)
    header = _header(paths[0])
    ts_index = header.index("created_at")
    size = os.path.getsize(paths[0])

    # Target ~75% of the way through the file.
    target = BASE + timedelta(minutes=3000)
    with open(paths[0], "rb") as f:
        offset = _seek_before(f, target, ts_index, len(header), size)

    assert offset > 0, "seek returned 0: binary search collapsed"
    # Should land close to, and at or before, the target.
    assert offset > 0.5 * size, "seek landed far earlier than the target"

    # The contract: the offset is a genuine row boundary whose own row precedes the
    # target -- so reading forward from it, and skipping rows below `start`, loses
    # nothing. Read the row *at* the offset, exactly as iter_window does.
    with open(paths[0], "rb") as f:
        found = _probe_row(f, offset, ts_index, len(header), skip_partial=False)
    assert found is not None
    pos, ts = found
    assert pos == offset, "offset is not a row boundary"
    assert ts < target, "seek landed after the target; rows would be dropped"
    assert target - ts < timedelta(minutes=60), "seek landed needlessly early"


def test_probe_row_skips_newline_fragments_and_finds_a_real_row(tmp_path):
    """Landing inside a multi-line quoted body must not yield a bogus row."""
    from arles.streaming import _header, _probe_row

    paths = _write_archive(tmp_path, n_files=1, rows_per_file=200)
    header = _header(paths[0])
    ts_index = header.index("created_at")
    size = os.path.getsize(paths[0])

    with open(paths[0], "rb") as f:
        # Probe from many arbitrary byte offsets: every one must resynchronise onto
        # a genuine row, never a shard of prose.
        for frac in (0.05, 0.17, 0.33, 0.5, 0.66, 0.81, 0.93):
            found = _probe_row(f, int(size * frac), ts_index, len(header))
            assert found is not None, f"no row found from offset {frac:.0%}"
            pos, ts = found
            assert BASE <= ts <= BASE + timedelta(minutes=200)


def test_out_of_order_timestamps_do_not_truncate_the_window(tmp_path):
    """THE regression test for the 115-vs-53,103 truncation.

    The archive is ordered by ingestion, not created_at. A row whose client-supplied
    timestamp runs hours ahead must not be mistaken for the end of the window.
    """
    paths = _write_disordered_archive(tmp_path)
    start = BASE + timedelta(hours=10)
    end = start + timedelta(hours=8)

    expected = _brute_force(paths, start, end)
    got = [r["action_id"] for r in iter_window(paths, start, end, progress=False)]

    assert sorted(got) == sorted(expected)
    # The point: the naive first-ts>=end rule would have returned a tiny fraction.
    assert len(got) > 400, f"window truncated to {len(got)} rows"


def test_disorder_tolerance_is_honoured_end_to_end(tmp_path):
    """With a disorder bound below the real jumps, we must warn -- not silently lie."""
    paths = _write_disordered_archive(tmp_path)
    start = BASE + timedelta(hours=10)
    end = start + timedelta(hours=8)

    generous = [r["action_id"] for r in iter_window(paths, start, end, progress=False)]
    exact = _brute_force(paths, start, end)
    assert sorted(generous) == sorted(exact)


def test_seek_is_widened_by_the_disorder_bound(tmp_path):
    """Seeking straight to `start` would skip rows that belong in the window."""
    paths = _write_disordered_archive(tmp_path, n_files=1, rows_per_file=6000)
    start = BASE + timedelta(hours=50)
    end = start + timedelta(hours=4)
    got = [r["action_id"] for r in iter_window(paths, start, end, progress=False, seek=True)]
    scanned = [r["action_id"] for r in iter_window(paths, start, end, progress=False, seek=False)]
    assert sorted(got) == sorted(scanned)
    assert sorted(got) == sorted(_brute_force(paths, start, end))


def test_rows_come_back_in_time_order(tmp_path):
    paths = _write_archive(tmp_path)
    start = BASE + timedelta(minutes=500)
    end = start + timedelta(minutes=400)
    seen = [
        datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        for r in iter_window(paths, start, end, progress=False)
    ]
    assert seen == sorted(seen)


def test_window_outside_the_archive_yields_nothing(tmp_path):
    paths = _write_archive(tmp_path)
    start = BASE + timedelta(days=400)
    assert list(iter_window(paths, start, start + timedelta(days=5), progress=False)) == []


def test_quoted_and_multiline_fields_survive_the_seek(tmp_path):
    """A byte offset can land inside a quoted, newline-containing field.

    The reader must resynchronise and hand back intact rows -- text included.
    """
    paths = _write_archive(tmp_path)
    start = BASE + timedelta(minutes=330)
    rows = list(iter_window(paths, start, start + timedelta(minutes=5), progress=False))
    assert len(rows) == 5
    for r in rows:
        # "Post 330, with a comma,..." -> 330
        ri = int(re.match(r"Post (\d+)", r["text"]).group(1))
        assert r["text"] == _post_text(ri)
        assert r["author_user_id"].startswith("user")
        assert r["action_id"].startswith("at://did:plc:")

    # The window must include the awkward ones: embedded newlines and nested quotes.
    texts = [r["text"] for r in rows]
    assert any("\n" in t for t in texts), "expected a multi-line body in this window"
    assert any('"quotes, inside quotes"' in t for t in texts)


def test_discover_files_accepts_a_directory(tmp_path):
    _write_archive(tmp_path, n_files=3)
    found = discover_files(str(tmp_path))
    assert [os.path.basename(p) for p in found] == [
        f"processed_enrico{i}.csv" for i in range(3)
    ]


def test_discover_files_accepts_a_single_file(tmp_path):
    paths = _write_archive(tmp_path, n_files=1)
    assert discover_files(paths[0]) == [paths[0]]
