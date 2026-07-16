#!/usr/bin/env python3
"""Regenerate the archetype evolution figure, and everything behind it.

What it produces
----------------
    archetype_evolution_e1_e2.pdf   the figure: mean archetype score per 5-day window
    archetype_fit.json              the frozen embedding -- the ruler
    loadings.txt                    what each axis is actually made of
    windows.csv                     every number behind the figure, N included
    archetype_space.pdf             pairwise scatter of the three axes (talks)
    cohorts_e1_e2.pdf               --cohorts only: incumbents vs newcomers

One ruler for every window
--------------------------
The embedding is fitted ONCE on all windows pooled, then frozen and applied to each.
This is not a preference. Fitted per window, each window's own extremes become its
reference and the most extreme user scores 1.0 by construction. Across E1's seven
windows the maximum reshares received per author runs 6,842 -> 275,198 -- a 40x swing in
the min-max denominator -- so identical behaviour scores 1.0 in the pre-event window and
0.025 six windows later. A line plotted over that is a picture of the ruler changing.

Pooling across E1 *and* E2 together, rather than per event, is what lets the two panels
share a y-axis: "amplifier = 0.15" then means the same thing in both. The paper compares
the events directly ("forced migration" vs "disengagement migration"), so they must be
measured on one scale.

Does that let one event "see" the other?
----------------------------------------
No. ArLeS is a windowed batch estimator, not an online one: every window builds its own
WindowIndex and FeatureExtractor from its own actions, so a window's twelve features are
bit-identical whether or not any other window was ever processed. Nothing accumulates.
The only thing shared is the fit itself -- 14 numbers per axis (4 means, 4 scales, 4
loadings, 2 bounds). A tape measure, not data.

The pooled bounds do come from whichever window held the largest outlier, so if later
periods carry more diffusion, earlier scores come out lower. That is the finding, not a
distortion: the platform grew. A per-window fit would set every window's top user to 1.0
and hide exactly that.

Cost
----
14 windows x ~2 archive passes. First run reads the archive (tens of minutes on an
external drive); afterwards the window cache makes it minutes. Bounded memory: one
window's buffers at a time.

Usage
-----
    python scripts/make_figures.py /Volumes/Uniform/bluesky_full --out figures/
    python scripts/make_figures.py <archive> --out figures/ --cohorts
"""

import argparse
import csv
import json
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from arles.actions import MalformedActionError, parse_timestamp  # noqa: E402
from arles.embedding import AXES, fit_pooled  # noqa: E402
from arles.features import FeatureExtractor, WindowIndex  # noqa: E402
from arles.mappers.bluesky import map_row  # noqa: E402
from arles.streaming import build_index, discover_files, iter_window  # noqa: E402

#: The two events Figure 8 shows, and the window each panel starts from.
#:
#: E1 spans Aug 25 -> Sep 24 and E2 Oct 12 -> Nov 11: seven consecutive 5-day windows
#: each, the first being the pre-event window. Dates are the paper's.
EVENTS = [
    {"id": "E1", "name": "X/Twitter ban in Brazil", "start": "2024-08-25", "event": "2024-08-30"},
    {"id": "E2", "name": "X/Twitter Terms & Privacy update", "start": "2024-10-12", "event": "2024-10-17"},
]
N_WINDOWS = 7
WINDOW_DAYS = 5.0

AXIS_LABEL = {
    "superspreader": "Super-Spreader",
    "amplifier": "Amplifier",
    "coordinated": "Coordinated",
}
AXIS_COLOUR = {"superspreader": "#1f77b4", "amplifier": "#ff7f0e", "coordinated": "#2ca02c"}
AXIS_MARKER = {"superspreader": "o", "amplifier": "s", "coordinated": "^"}


def _date(s):
    return parse_timestamp(f"{s} 00:00:00+00:00")


def _cache_key(start, end):
    return f"win_{start.strftime('%Y%m%dT%H%M%S')}_{end.strftime('%Y%m%dT%H%M%S')}"


def load_window(files, spans, start, end, cache_dir=None):
    """Both passes over one window, cached. Returns (extractor, index)."""
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        npz = Path(cache_dir) / f"{_cache_key(start, end)}.npz"
        meta = Path(cache_dir) / f"{_cache_key(start, end)}.index.json"
        if npz.exists() and meta.exists():
            print(f"    cache hit: {npz.name}")
            m = json.loads(meta.read_text())
            index = WindowIndex(n_reposts=m["n_reposts"], n_unattributed=m["n_unattributed"])
            return FeatureExtractor.load_buffer(str(npz), index), index

    index = WindowIndex()
    for row in iter_window(files, start, end, spans=spans, progress=True):
        try:
            index.add(map_row(row))
        except MalformedActionError:
            continue

    fx = FeatureExtractor(index, start, end)
    for row in iter_window(files, start, end, spans=spans, progress=True):
        try:
            fx.add(map_row(row))
        except MalformedActionError:
            continue

    if cache_dir:
        fx.save_buffer(str(npz))
        meta.write_text(json.dumps({"n_reposts": index.n_reposts,
                                    "n_unattributed": index.n_unattributed}))
    return fx, index


def collect(files, spans, cache_dir):
    """Every window of every event. Returns a list of dicts, in time order."""
    out = []
    for ev in EVENTS:
        first = _date(ev["start"])
        for w in range(N_WINDOWS):
            start = first + timedelta(days=WINDOW_DAYS * w)
            end = start + timedelta(days=WINDOW_DAYS)
            print(f"\n  {ev['id']} window {w + 1}/{N_WINDOWS}: {start.date()} -> {end.date()}")
            fx, index = load_window(files, spans, start, end, cache_dir)
            ids, X = fx.finish()
            _, conf = fx.confidence()
            print(f"    {index.n_reposts:,} reposts, {len(ids):,} users")
            out.append({
                "event": ev["id"], "event_name": ev["name"], "w": w,
                "start": start, "end": end, "ids": ids, "X": X, "conf": conf,
                "n_reposts": index.n_reposts,
            })
    return out


def write_windows_csv(windows, Zs, path):
    """The numbers behind the figure. A figure nobody can check is a picture."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "window", "start", "end", "n_users", "n_reposts",
                    "mean_confidence"]
                   + [f"mean_{a}" for a in AXES] + [f"median_{a}" for a in AXES])
        for win, Z in zip(windows, Zs):
            w.writerow([
                win["event"], win["w"] + 1, win["start"].date(), win["end"].date(),
                len(win["ids"]), win["n_reposts"],
                f"{float(np.mean(win['conf'])):.4f}" if len(win["conf"]) else "",
            ] + [f"{Z[:, i].mean():.4f}" for i in range(3)]
              + [f"{np.median(Z[:, i]):.4f}" for i in range(3)])


def _panels(windows):
    """Only the events that actually have windows, so a partial run still plots.

    plt.subplots(1, 1) returns a bare Axes rather than an array, which is why the
    caller must not index blindly.
    """
    return [ev for ev in EVENTS
            if any(w["event"] == ev["id"] and len(w["ids"]) for w in windows)]


def plot_evolution(windows, Zs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = _panels(windows)
    if not events:
        return
    fig, axes = plt.subplots(len(events), 1, figsize=(7.0, 3.0 * len(events)),
                             squeeze=False)
    axes = axes[:, 0]
    for panel, ev in enumerate(events):
        ax = axes[panel]
        idx = [i for i, w in enumerate(windows)
               if w["event"] == ev["id"] and len(w["ids"])]
        xs = [windows[i]["start"] for i in idx]
        for a, axis in enumerate(AXES):
            ys = [Zs[i][:, a].mean() for i in idx]
            ax.plot(xs, ys, marker=AXIS_MARKER[axis], color=AXIS_COLOUR[axis],
                    label=AXIS_LABEL[axis], linewidth=1.6, markersize=5)
        ax.axvline(_date(ev["event"]), color="red", linestyle="--", linewidth=1.2)
        ax.annotate(ev["id"], xy=(_date(ev["event"]), ax.get_ylim()[1]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=10)
        ax.set_ylabel("Archetype score")
        ax.grid(alpha=0.25, linestyle=":")
        ax.margins(x=0.02)
        for label in ax.get_xticklabels():
            label.set_rotation(0)
    # One legend, one shared scale: the whole point of the pooled fit.
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_archetype_space(windows, Zs, path, max_points=4000):
    """Pairwise scatter of the three axes. Shows the archetypes are not exclusive."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z = np.vstack(Zs)
    if Z.shape[0] > max_points:
        rng = np.random.default_rng(0)  # seeded: the figure must be reproducible
        Z = Z[rng.choice(Z.shape[0], max_points, replace=False)]

    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(Z[:, i], Z[:, j], s=4, alpha=0.15, color="#33475b", linewidths=0)
        ax.set_xlabel(AXIS_LABEL[AXES[i]])
        ax.set_ylabel(AXIS_LABEL[AXES[j]])
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cohorts(windows, Zs, path):
    """Incumbents vs newcomers, on the same pooled ruler.

    The aggregate cannot separate "users changed how they spread" from "different users
    arrived": the population roughly tripled at each event (E1: 104,503 -> 331,527
    authors in one window). Incumbents are users already active in the pre-event window;
    newcomers are first seen after it. If the incumbent line moves, behaviour adapted.
    If it is flat and only newcomers sit high, the platform changed by replacement.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = _panels(windows)
    if not events:
        return
    fig, axes = plt.subplots(len(events), 1, figsize=(7.0, 3.2 * len(events)),
                             squeeze=False)
    axes = axes[:, 0]
    for panel, ev in enumerate(events):
        ax = axes[panel]
        idx = [i for i, w in enumerate(windows)
               if w["event"] == ev["id"] and len(w["ids"])]
        incumbents = set(windows[idx[0]]["ids"])  # active in the pre-event window
        xs = [windows[i]["start"] for i in idx]

        for a, axis in enumerate(AXES):
            inc_y, new_y = [], []
            for i in idx:
                ids = np.array(windows[i]["ids"])
                mask = np.array([u in incumbents for u in ids]) if len(ids) else np.zeros(0, bool)
                inc_y.append(Zs[i][mask, a].mean() if mask.any() else np.nan)
                new_y.append(Zs[i][~mask, a].mean() if (~mask).any() else np.nan)
            ax.plot(xs, inc_y, marker="o", color=AXIS_COLOUR[axis], linewidth=1.6,
                    label=f"{AXIS_LABEL[axis]} (incumbent)")
            ax.plot(xs, new_y, marker="x", color=AXIS_COLOUR[axis], linewidth=1.2,
                    linestyle="--", label=f"{AXIS_LABEL[axis]} (newcomer)")
        ax.axvline(_date(ev["event"]), color="red", linestyle="--", linewidth=1.2)
        ax.set_ylabel("Archetype score")
        ax.set_title(f"{ev['id']}: {ev['name']}", fontsize=10)
        ax.grid(alpha=0.25, linestyle=":")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.55), ncol=3,
                   frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="a CSV file, or a directory of sequential CSVs")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--cache-dir", default=".arles_windows",
                    help="window buffer cache; '' to disable")
    ap.add_argument("--cohorts", action="store_true",
                    help="also split incumbents vs newcomers (see plot_cohorts)")
    ap.add_argument("--no-log", action="store_true",
                    help="do not log1p the features before scaling. Only for comparison: "
                         "on raw counts one account defines the axis and the other 99.9%% "
                         "are pinned at zero.")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = discover_files(args.path)
    cache = str(Path(args.path) / ".arles_index.json") if Path(args.path).is_dir() else None
    print(f"archive: {args.path} ({len(files)} file(s))")
    spans = build_index(files, cache_path=cache, verbose=True)

    windows = collect(files, spans, args.cache_dir or None)
    if not any(len(w["ids"]) for w in windows):
        print("\nNo users in any window. Nothing to plot.")
        return

    # ONE fit, all 14 windows, then frozen. See the module docstring.
    print("\nfitting the embedding on all windows pooled ...")
    embedder = fit_pooled([w["X"] for w in windows if len(w["ids"])],
                          log_scale=not args.no_log)
    Zs = [embedder.transform(w["X"]) if len(w["ids"]) else np.zeros((0, 3))
          for w in windows]

    for w in embedder.warnings():
        print(f"\n  ! {w}")

    (out / "archetype_fit.json").write_text(embedder.to_json())
    (out / "loadings.txt").write_text(embedder.loadings_table() + "\n")
    write_windows_csv(windows, Zs, out / "windows.csv")

    plot_evolution(windows, Zs, out / "archetype_evolution_e1_e2.pdf")
    plot_archetype_space(windows, Zs, out / "archetype_space.pdf")
    if args.cohorts:
        plot_cohorts(windows, Zs, out / "cohorts_e1_e2.pdf")

    print("\n" + "=" * 72)
    print(embedder.loadings_table())
    print("=" * 72)
    print(f"\n{'event':<6} {'window':<12} {'users':>9} {'reposts':>10} "
          + " ".join(f"{AXIS_LABEL[a]:>15}" for a in AXES))
    for win, Z in zip(windows, Zs):
        if not len(win["ids"]):
            continue
        print(f"{win['event']:<6} {str(win['start'].date()):<12} {len(win['ids']):>9,} "
              f"{win['n_reposts']:>10,} "
              + " ".join(f"{Z[:, i].mean():>15.4f}" for i in range(3)))

    print(f"\nwritten to {out}/")
    for name in sorted(p.name for p in out.iterdir()):
        print(f"  {name}")
    print(f"\nelapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
