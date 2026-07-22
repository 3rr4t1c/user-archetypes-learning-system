#!/usr/bin/env python3
"""Paired significance test for the archetype shifts, on incumbent users.

Answers the question a reviewer will ask about the micro-scale result: did the *same*
accounts change their behaviour after an event, or did the aggregate move only because a
different population arrived? For each event and archetype axis it takes the users present
in BOTH the pre-event window and a post-event window (the incumbents), pairs each user's
score with itself, and runs a Wilcoxon signed-rank test on the within-user differences.

Why incumbents, and why paired
------------------------------
The population turns over at every migration event -- most post-event users are newcomers.
An unpaired comparison of pre- vs post-window scores therefore confounds "users changed"
with "different users". Restricting to incumbents and pairing each user with itself removes
that confound: a significant, positive median difference means the accounts we were already
watching shifted, which is the claim that survives the newcomer critique.

Confidence gate
---------------
Users are filtered to confidence >= --min-confidence (default 0.3, the pipeline default)
in BOTH windows before pairing, so the test speaks only for well-observed accounts -- the
same population the figures describe.

Cost
----
Two archive passes per window, like make_figures; the window cache makes reruns cheap.
The Wilcoxon itself is instant. Needs scipy for the exact test; without it, falls back to a
normal approximation and says so.

Usage
-----
    python scripts/paired_test.py /Volumes/Uniform/bluesky_full --out figures/paired_test.csv
    python scripts/paired_test.py <archive> --events E1,E2 --post-window 2
"""

import argparse
import csv
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from arles.embedding import AXES, ArchetypeEmbedder, fit_pooled  # noqa: E402
from arles.streaming import build_index, discover_files  # noqa: E402

# Reuse make_figures' window machinery so this test scores users identically to the figure.
import importlib.util  # noqa: E402

_MF = Path(__file__).resolve().parent / "make_figures.py"
_spec = importlib.util.spec_from_file_location("make_figures", _MF)
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)


def wilcoxon(diffs: np.ndarray):
    """(statistic, p_value, method) for a one-sample Wilcoxon signed-rank on diffs.

    Uses scipy when available (exact/continuity-corrected); otherwise a normal
    approximation so the script still runs, flagged in the returned method string.
    """
    d = diffs[diffs != 0.0]
    n = d.size
    if n == 0:
        return 0.0, 1.0, "no non-zero differences"
    try:
        from scipy.stats import wilcoxon as _w
        stat, p = _w(diffs, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p), "scipy exact/asymptotic"
    except ImportError:
        ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
        w_plus = ranks[d > 0].sum()
        mean = n * (n + 1) / 4.0
        sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (w_plus - mean) / sd if sd > 0 else 0.0
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        return float(w_plus), float(p), "normal approximation (install scipy for exact)"


def score_window(files, spans, start, end, cache_dir, emb, min_conf):
    """{user_id: 3-vector} for well-observed users of one window, on the frozen fit."""
    fx, index = mf.load_window(files, spans, start, end, cache_dir)
    ids, X = fx.finish()
    if not ids:
        return {}
    _, conf = fx.confidence()
    Z = emb.transform(X)
    keep = np.asarray(conf) >= min_conf
    return {u: Z[i] for i, (u, k) in enumerate(zip(ids, keep)) if k}


def paired_rows(pre, post):
    """(axis -> (pre_scores, post_scores)) over the incumbents shared by both windows."""
    shared = [u for u in pre if u in post]
    out = {}
    for a, axis in enumerate(AXES):
        pre_a = np.array([pre[u][a] for u in shared])
        post_a = np.array([post[u][a] for u in shared])
        out[axis] = (pre_a, post_a)
    return len(shared), out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="a CSV file, or a directory of sequential CSVs")
    ap.add_argument("--out", default="figures/paired_test.csv")
    ap.add_argument("--events", default="", help="comma-separated ids, e.g. E1,E2")
    ap.add_argument("--post-window", type=int, default=2,
                    help="which window is 'post' (1=pre-event, 2=first post-event; default 2)")
    ap.add_argument("--cache-dir", default=".arles_windows")
    ap.add_argument("--min-confidence", type=float, default=mf.DEFAULT_MIN_CONFIDENCE)
    args = ap.parse_args()

    events = mf.EVENTS
    if args.events:
        wanted = {e.strip().upper() for e in args.events.split(",")}
        events = [e for e in mf.EVENTS if e["id"] in wanted]

    files = discover_files(args.path)
    cache = str(Path(args.path) / ".arles_index.json") if Path(args.path).is_dir() else None
    spans = build_index(files, cache_path=cache, verbose=True)

    # The frozen fit from make_figures, so scores match the figure exactly.
    fit_path = Path(args.out).parent / "archetype_fit.json"
    if fit_path.exists():
        emb = ArchetypeEmbedder.from_json(fit_path.read_text())
        print(f"using frozen fit {fit_path}")
    else:
        print("no archetype_fit.json found; fitting on the pre/post windows in play "
              "(scores will not match the figure -- run make_figures first for that)")
        emb = None

    rows = []
    for ev in events:
        first = mf._date(ev["start"])
        pre_s, pre_e = first, first + timedelta(days=mf.WINDOW_DAYS)
        k = args.post_window - 1
        post_s = first + timedelta(days=mf.WINDOW_DAYS * k)
        post_e = post_s + timedelta(days=mf.WINDOW_DAYS)
        print(f"\n{ev['id']}: pre {pre_s.date()}  vs  post {post_s.date()}")

        if emb is None:
            # Fit on just these two windows -- diagnostic only.
            f_pre, i_pre = mf.load_window(files, spans, pre_s, pre_e, args.cache_dir)
            f_post, i_post = mf.load_window(files, spans, post_s, post_e, args.cache_dir)
            _, Xp = f_pre.finish(); _, Xq = f_post.finish()
            emb_local = fit_pooled([Xp, Xq])
            pre = score_window(files, spans, pre_s, pre_e, args.cache_dir, emb_local, args.min_confidence)
            post = score_window(files, spans, post_s, post_e, args.cache_dir, emb_local, args.min_confidence)
        else:
            pre = score_window(files, spans, pre_s, pre_e, args.cache_dir, emb, args.min_confidence)
            post = score_window(files, spans, post_s, post_e, args.cache_dir, emb, args.min_confidence)

        n_pairs, by_axis = paired_rows(pre, post)
        print(f"  incumbents (in both, confidence>={args.min_confidence:g}): {n_pairs:,}")
        for axis in AXES:
            a, b = by_axis[axis]
            diffs = b - a
            stat, p, method = wilcoxon(diffs)
            med = float(np.median(diffs)) if diffs.size else float("nan")
            rows.append({
                "event": ev["id"], "axis": axis, "n_pairs": n_pairs,
                "median_pre": round(float(np.median(a)), 4) if a.size else "",
                "median_post": round(float(np.median(b)), 4) if b.size else "",
                "median_diff": round(med, 4), "wilcoxon_stat": round(stat, 1),
                "p_value": p, "method": method,
            })
            print(f"    {axis:<14} n={n_pairs:>7,}  median dpre->post={med:+.4f}  "
                  f"p={p:.2e}  ({method})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
