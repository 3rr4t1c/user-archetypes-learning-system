#!/usr/bin/env python3
"""Regenerate the archetype evolution figure, and everything behind it.

What it produces
----------------
    fig_coordination_population.pdf THE figure (population): median coordinated score --
                                    only E1 moves the typical user
    fig_archetype_prevalence.pdf    THE figure (head): prevalence per archetype, each on
                                    its own scale -- shows every axis's event structure
    archetype_fit.json              the frozen embedding -- the ruler
    loadings.txt                    what each axis is actually made of
    windows.csv                     per-window means and medians of the three axes, N
    features.csv                    per-window means of the twelve raw features
    prevalence.csv                  every head statistic, per window per axis
    diagnostics/                    everything a reviewer might want, out of the way:
      fig_mean_score_per_window.pdf   the published figure: mean archetype score
      fig_prevalence.pdf              accounts above a common bar: count and rate/100k
      fig_threshold_sweep.pdf         the same at five bars: the bar is not the finding
      fig_head_intensity.pdf          p99/p99.9 per window -- no denominator
      fig_concentration.pdf           Gini and top-1% mass share -- no threshold
      fig_prepost_prevalence.pdf      pre / post / late per event, absolute and fold change
      fig_archetype_space.pdf         pairwise scatter of the three axes (talks)
      fig_cohorts.pdf                 --cohorts only: incumbents vs newcomers

Why the mean figure is not enough
---------------------------------
The published figure plots a mean over every account in the window. None of the three
archetypes describes a typical account, so that mean is mostly a statement about the
~99% who are no archetype at all. Measured on the archive, `median_superspreader` is
0.0000 in all fourteen E1/E2 windows and the mean never leaves [0.021, 0.037]: a
statistic that cannot move cannot support a claim, and a flat line in it is not evidence
of a flat phenomenon.

The other figures measure the head instead: how many accounts clear a fixed bar
(prevalence), how extreme the extremes are (intensity), and how unequally the axis is
spread (concentration). They fail differently from each other, which is the point.

The bar is common to all three axes
-----------------------------------
The three archetypes are rare at three *different* scales -- amplifiers outnumber
super-spreaders by an order of magnitude, with coordinated accounts in between -- and the
figures have to preserve that, because it is the thing that makes their counts
incomparable and their fold changes the only fair comparison.

So `--bar` is one score applied to all three axes. It is emphatically NOT a per-axis
quantile: the top q of each axis puts (1-q) of the population above every axis's bar by
construction, which reports the three archetypes as equally common. On the real
pre-event window that gave 194 / 198 / 304 accounts; the common bar at 0.5 gives
538 / 3,137 / 616. See arles.prevalence for the argument in full.

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
    # quick look at the figures: two cached windows of one event, ~1 min
    python scripts/make_figures.py <archive> --out figures_preview/ \
        --events E1 --max-windows 2

    # the study
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
from arles.features import FEATURE_NAMES, FeatureExtractor, WindowIndex  # noqa: E402
from arles.mappers.bluesky import map_row  # noqa: E402
from arles.prevalence import (  # noqa: E402
    COMMON_BAR,
    SWEEP_BARS,
    fold_change,
    head_stats,
)
from arles.streaming import build_index, discover_files, iter_window  # noqa: E402

#: The paper's four migration events, and the window each panel starts from.
#:
#: Each event gets seven consecutive 5-day windows, the first being the pre-event window,
#: so the event date always falls just inside window 1 and the panel shows one window of
#: "before". Dates are the paper's (Sec. 4.1).
#:
#: E4 is short. The archive ends 2025-01-28, so only five of E4's seven windows are
#: complete (Jan 1 -> Jan 26); `n_windows` caps it rather than plotting a partial window
#: as though it were a whole one, which would read as a collapse in every count.
#:
#: E2 and E3 overlap in time (E2 runs to Nov 16, E3 from Oct 31), so the pooled fit sees
#: Oct 31 -> Nov 16 twice. That slightly overweights those days in the ruler and in the
#: pooled threshold; it does not affect any single window's features, which are built
#: from that window's actions alone.
EVENTS = [
    {"id": "E1", "name": "X/Twitter ban in Brazil",
     "start": "2024-08-25", "event": "2024-08-30", "n_windows": 7},
    {"id": "E2", "name": "X/Twitter Terms & Privacy update",
     "start": "2024-10-12", "event": "2024-10-17", "n_windows": 7},
    {"id": "E3", "name": "US Presidential election",
     "start": "2024-10-31", "event": "2024-11-05", "n_windows": 7},
    {"id": "E4", "name": "Broad social-media ToS updates",
     "start": "2025-01-01", "event": "2025-01-06", "n_windows": 5},
]
N_WINDOWS = 7
WINDOW_DAYS = 5.0

#: Minimum confidence for a user's vector to enter the analysis.
#:
#: Confidence (arles.features.confidence) is data availability: 0.5*volume + 0.3*recency
#: + 0.2*lifespan, where volume saturates around 20 actions. An archetype vector built on
#: one or two observed actions is not evidence, so it should not be scored as though it
#: were -- describing a confidence score and then ignoring it would be indefensible.
#:
#: 0.3 is the point below which a user has effectively been seen act only once or twice.
#: Worked cases in a 5-day window: a single action scores ~0.13-0.42 (depending only on
#: how recent it was); ~3 actions spread across a couple of days score ~0.37; ~5 score
#: ~0.6. So 0.3 keeps users with a few actions and drops the barely-seen. It is a
#: deliberately low, interpretable bar, not a percentile -- the kept fraction falls out
#: of the data (about a third of a pre-event window, fewer at an influx).
#:
#: This does not erase a forced-migration influx, it delays it: newly-arrived accounts
#: cross the bar once they have acted a few times, so the effect surfaces a little later
#: rather than vanishing. Set --min-confidence 0.0 to keep everyone.
DEFAULT_MIN_CONFIDENCE = 0.3

#: Where the archive stops. Any window ending after this is incomplete by definition and
#: is not plotted -- see EVENTS["n_windows"] for E4.
ARCHIVE_END = "2025-01-28"

AXIS_LABEL = {
    "superspreader": "Super-Spreader",
    "amplifier": "Amplifier",
    "coordinated": "Coordinated",
}

#: Seaborn's "colorblind" palette, hard-coded.
#:
#: The published figure was drawn with sns.set_palette("colorblind"), so these are its
#: first three entries. Hard-coding them keeps the published look without making seaborn
#: a dependency of a repo that otherwise needs only numpy -- and matplotlib's default
#: tab10 is visibly not the same (#ff7f0e is a brighter orange than #DE8F05, #2ca02c a
#: grassier green than the teal #029E73).
AXIS_COLOUR = {
    "superspreader": "#0173B2",
    "amplifier": "#DE8F05",
    "coordinated": "#029E73",
}
AXIS_MARKER = {"superspreader": "o", "amplifier": "s", "coordinated": "^"}

#: Approximates seaborn's "whitegrid": white panel, light grey dashed grid, light box.
STYLE = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.facecolor": "white",
    "axes.edgecolor": "#b0b0b0",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#d0d0d0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
}


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
        n = ev.get("n_windows", N_WINDOWS)
        for w in range(n):
            start = first + timedelta(days=WINDOW_DAYS * w)
            end = start + timedelta(days=WINDOW_DAYS)
            print(f"\n  {ev['id']} window {w + 1}/{n}: {start.date()} -> {end.date()}")
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


def write_features_csv(windows, path):
    """Per-window mean of each of the twelve raw features, before the embedding.

    The axis means cannot tell you *why* an axis moved -- PC1 mixes four features, and
    it weights them by variance, not by how much you trust them. On the real archive
    the coordinated loadings came out co_action_size +0.614, co_action_rate +0.594,
    co_action_latency +0.463, niche_co_action +0.234: PC1 down-weighted the one feature
    designed to exclude viral pile-ons, because it varies least.

    That matters, because a migration event makes everyone reshare the same few posts,
    which is exactly what co_action_size rewards. So a jump in the coordinated axis is
    ambiguous between coordination and a pile-on -- unless niche_co_action moves too.
    This file is how you check.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "window", "start", "n_users"] + list(FEATURE_NAMES))
        for win in windows:
            X = win["X"]
            means = [f"{X[:, i].mean():.4f}" if len(win["ids"]) else ""
                     for i in range(len(FEATURE_NAMES))]
            w.writerow([win["event"], win["w"] + 1, win["start"].date(),
                        len(win["ids"])] + means)


def write_prevalence_csv(windows, Zs, bar, path):
    """Every head statistic, per window per axis. The numbers behind the new figures."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "window", "start", "axis", "threshold", "n_users", "count",
                    "rate_per_100k", "p99", "p999", "pmax", "gini", "top1pct_share",
                    "anchor_feature", "anchor_median"])
        for win, Z in zip(windows, Zs):
            if not len(win["ids"]):
                continue
            for a, axis in enumerate(AXES):
                h = head_stats(Z[:, a], bar, axis, X=win["X"])
                w.writerow([win["event"], win["w"] + 1, win["start"].date(), axis,
                            f"{h.threshold:.6f}", h.n_users, h.count,
                            f"{h.rate_per_100k:.3f}", f"{h.p99:.4f}", f"{h.p999:.4f}",
                            f"{h.pmax:.4f}", f"{h.gini:.4f}", f"{h.top1pct_share:.4f}",
                            h.anchor_feature, f"{h.anchor_median:.1f}"])


def _panels(windows):
    """Only the events that actually have windows, so a partial run still plots.

    plt.subplots(1, 1) returns a bare Axes rather than an array, which is why the
    caller must not index blindly.
    """
    return [ev for ev in EVENTS
            if any(w["event"] == ev["id"] and len(w["ids"]) for w in windows)]


#: Overlay palette for the event-comparison figures.
#:
#: Seaborn's colorblind palette (the paper's), so the four events are distinguishable
#: to colourblind readers; E1 (the forced migration) is drawn heavier because it is the
#: one that behaves differently, but every event is a full member of the plot.
CLAIM_COLOUR = {"E1": "#0173B2", "E2": "#DE8F05", "E3": "#029E73", "E4": "#D55E00"}
CLAIM_MARKER = {"E1": "o", "E2": "s", "E3": "^", "E4": "D"}
#: Legend labels are just the event ids. The events are defined in the paper body
#: (Sec. 4.1), so repeating "Brazil ban" etc. in the legend only adds width; the ids
#: sit above the panels as a compact key.
CLAIM_NAME = {"E1": "E1", "E2": "E2", "E3": "E3", "E4": "E4"}


def _overlay(ax, series_by_event, ylog=False):
    """Draw one panel: one line per event, all weighted equally, colourblind palette.

    No event is emphasised. Drawing E1 heavier pre-loaded the conclusion into the styling
    -- the four events are distinguished by colour and marker, and the reader is left to
    see which one behaves differently.
    """
    for ev_id, (xs, ys) in series_by_event.items():
        ax.plot(xs, ys, color=CLAIM_COLOUR[ev_id], marker=CLAIM_MARKER[ev_id],
                label=CLAIM_NAME[ev_id], linewidth=1.8, markersize=5.5,
                markeredgewidth=0, zorder=3)
    if ylog:
        ax.set_yscale("log")
    # The event is at x=0: window 1 (pre-event, at -5) covers the five days before it,
    # window 2 (at 0) is the first window that starts at the event. The line marks it.
    ax.axvline(0, color="#666666", linestyle=":", linewidth=1.1, zorder=1)
    ax.set_xticks([-5, 0, 10, 20])
    ax.set_xlabel("days since event")
    ax.set_axisbelow(True)


def _rows_for(windows, Zs, ev_id):
    rows = [(w, Z) for w, Z in zip(windows, Zs)
            if w["event"] == ev_id and len(w["ids"])]
    rows.sort(key=lambda r: r[0]["w"])
    xs = [r[0]["w"] * 5 - 5 for r in rows]  # window 1 -> -5 days (pre), window 2 -> 0
    return xs, rows


def plot_population_median(windows, Zs, path):
    """The population-level signature: the coordinated score of the MEDIAN user.

    This is the striking, threshold-free half of the finding. A median cannot be moved by
    a rare minority, so a change in it means the *typical* account changed. Only the
    forced migration (E1) moves it -- the median user goes from co-acting with no one to
    co-acting, and stays there for a month. At the three disengagement events it never
    leaves zero: whatever coordination they provoke is confined to a small head group
    (see plot_prevalence_by_archetype) and never becomes a population phenomenon.

    A super-spreader or amplifier median would be zero everywhere by construction (those
    classes are far too rare to reach a population median), which is exactly why this
    panel is coordinated-only and the head view is a separate figure.
    """
    plt = _mpl()
    events = _panels(windows)
    if not events:
        return
    coord = AXES.index("coordinated")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        series = {}
        for ev in events:
            xs, rows = _rows_for(windows, Zs, ev["id"])
            series[ev["id"]] = (xs, [float(np.median(Z[:, coord])) for _, Z in rows])
        _overlay(ax, series)
        ax.set_ylabel("median coordinated score")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.10), columnspacing=1.6)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_prevalence_by_archetype(windows, Zs, bar, path):
    """The head-level view, one panel per archetype, each on its OWN scale.

    The archetypes occur at scales orders of magnitude apart -- super-spreaders are a
    handful, amplifiers are many, coordinated accounts sit between -- so they cannot share
    a y-axis and a population mean cannot see the rare ones at all. Each panel therefore
    plots prevalence (accounts per 100k scoring above `bar` on that axis) over time, on
    its own log scale, all four events overlaid.

    This is the figure that shows the scale asymmetry was accounted for, and it corrects a
    tempting error: at the head, super-spreaders are NOT flat -- their prevalence rises
    ~2.7x after E1 and climbs at E4 -- even though the population mean and median (which a
    rare class cannot move) look dead. Coordinated prevalence jumps 6-17x at the shocks;
    amplification is the one axis that barely responds to any event.
    """
    plt = _mpl()
    events = _panels(windows)
    if not events:
        return

    with plt.rc_context(STYLE):
        # A two-column-spanning figure* (\includegraphics[width=\textwidth]). At 10.2in
        # wide the three panels are proper landscape rectangles; height 3.0in (down from
        # the original 3.5) squeezes the vertical extent for the page budget without
        # forcing the panels toward square. LaTeX scales the whole thing to \textwidth, so
        # only the aspect ratio matters on the page.
        fig, axes = plt.subplots(1, len(AXES), figsize=(3.4 * len(AXES), 3.0))
        for ax, axis in zip(axes, AXES):
            a = AXES.index(axis)
            series = {}
            for ev in events:
                xs, rows = _rows_for(windows, Zs, ev["id"])
                ys = [head_stats(Z[:, a], bar, axis).rate_per_100k for _, Z in rows]
                series[ev["id"]] = (xs, [y if y > 0 else np.nan for y in ys])
            _overlay(ax, series, ylog=True)
            ax.set_title(AXIS_LABEL[axis], fontsize=11)
        axes[0].set_ylabel("accounts per 100k users")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(events), frameon=False,
                   bbox_to_anchor=(0.5, 1.03), columnspacing=2.2, handletextpad=0.4)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_evolution(windows, Zs, path):
    """The published figure's layout, on the pooled ruler.

    Three details are deliberate, and each was wrong in the first version:

    * Ticks sit exactly on the window starts, one per data point. matplotlib's automatic
      date locator picks its own interval (6 days against 5-day windows), so most ticks
      land between points and label nothing in particular.
    * Labels are "%d %b" -- "25 Aug". ISO dates are twice as wide and collide.
    * Colours are seaborn's colorblind palette, as the published figure used. tab10's
      orange and green are visibly different.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    events = _panels(windows)
    if not events:
        return

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(events), 1, figsize=(7.2, 3.1 * len(events)),
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
                        label=AXIS_LABEL[axis], linewidth=1.8, markersize=5.5,
                        markeredgewidth=0, clip_on=False, zorder=3)

            # One tick per window, on the point. Nothing is being interpolated between
            # them, so a tick anywhere else invites the reader to think otherwise.
            ax.set_xticks(xs)
            ax.set_xticklabels([d.strftime("%d %b") for d in xs])
            ax.set_xlim(xs[0] - timedelta(days=1.2), xs[-1] + timedelta(days=1.2))

            ax.axvline(_date(ev["event"]), color="#d62728", linestyle="--",
                       linewidth=1.3, zorder=2)
            ax.set_ylabel("Archetype score")
            ax.set_axisbelow(True)

            # The event label goes above the panel, clear of the lines.
            ax.annotate(ev["id"], xy=(_date(ev["event"]), 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

        # One legend for both panels: they share a scale, which is the whole point.
        axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.10), ncol=3,
                       frameon=False, handlelength=2.4, columnspacing=2.5)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def _event_windows(windows, ev_id):
    return [i for i, w in enumerate(windows)
            if w["event"] == ev_id and len(w["ids"])]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _date_ticks(ax, xs, rotate=False):
    """One tick per window, on the point, labelled '25 Aug'.

    matplotlib's automatic date locator picks its own interval (6 days against 5-day
    windows), so ticks land between points and label nothing in particular.

    `rotate` is not cosmetic: seven "25 Aug" labels fit across the 7.2-inch single-column
    figure and do not fit across a 4.3-inch half of a two-column one, where they render
    as "25 Aug30 Aug04 Sep".
    """
    ax.set_xticks(xs)
    ax.set_xticklabels([d.strftime("%d %b") for d in xs],
                       rotation=45 if rotate else 0,
                       ha="right" if rotate else "center")
    ax.set_xlim(xs[0] - timedelta(days=1.2), xs[-1] + timedelta(days=1.2))
    ax.set_axisbelow(True)


def plot_prevalence(windows, Zs, bar, path):
    """How many accounts clear a fixed bar, and what share of the platform they are.

    The figure the mean-over-everyone plot cannot be. Left column: absolute head count.
    Right column: the same as a rate per 100,000 accounts. Both, deliberately -- the
    population tripled at E1 and at E2, so a count that triples means the rate did not
    move, and a count that holds means the rate fell 3x. Either line alone lets a reader
    draw whichever conclusion they arrived with.

    Log y throughout: the rates live between ~1 and ~1000 per 100k, and a linear axis
    would render the super-spreader series as a line along zero -- the same failure as
    the mean, one step later.
    """
    plt = _mpl()
    events = _panels(windows)
    if not events:
        return

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(events), 2, figsize=(9.6, 2.5 * len(events)),
                                 squeeze=False)
        for row, ev in enumerate(events):
            idx = _event_windows(windows, ev["id"])
            xs = [windows[i]["start"] for i in idx]
            for col, (what, label) in enumerate(
                [("count", "Accounts above bar"), ("rate", "per 100k accounts")]
            ):
                ax = axes[row, col]
                for a, axis in enumerate(AXES):
                    ys = []
                    for i in idx:
                        h = head_stats(Zs[i][:, a], bar, axis)
                        v = h.count if what == "count" else h.rate_per_100k
                        ys.append(v if v > 0 else np.nan)  # log axis: 0 has no place
                    ax.plot(xs, ys, marker=AXIS_MARKER[axis], color=AXIS_COLOUR[axis],
                            label=AXIS_LABEL[axis], linewidth=1.8, markersize=5,
                            markeredgewidth=0, zorder=3)
                ax.set_yscale("log")
                _date_ticks(ax, xs, rotate=True)
                ax.axvline(_date(ev["event"]), color="#d62728", linestyle="--",
                           linewidth=1.3, zorder=2)
                # The row is already labelled with the event id, so the dashed line
                # needs no annotation repeating it.
                ax.set_ylabel(f"{ev['id']}\n{label}" if col == 0 else label)

        axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=3,
                          frameon=False, handlelength=2.2, columnspacing=2.0)
        fig.suptitle(f"Accounts scoring >= {bar:g} on each axis", y=1.005, fontsize=10)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_threshold_sweep(windows, Zs, path):
    """Prevalence at five bars, so the reader can see the bar is not the finding.

    Any threshold invites "why that one?", and the honest answer is not a better
    threshold -- it is that the answer does not depend on it. One column per archetype,
    one line per bar, each normalised to its own pre-event window so five series
    spanning orders of magnitude can share an axis.

    Normalising per series is also what makes this figure legitimate across archetypes:
    amplifiers outnumber super-spreaders by an order of magnitude, so plotting their
    absolute rates together would show one flat line at the bottom and nothing else. The
    absolute scale is fig_prevalence's job; this figure's job is the shape of the change.

    If the lines move together, the shape is the data's. If they cross, it is the bar's,
    and nothing at any single bar should be reported.
    """
    plt = _mpl()
    events = _panels(windows)
    if not events:
        return

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(events), len(AXES),
                                 figsize=(3.4 * len(AXES), 2.4 * len(events)),
                                 squeeze=False, sharey="row")
        shades = plt.cm.viridis(np.linspace(0.15, 0.85, len(SWEEP_BARS)))
        for row, ev in enumerate(events):
            idx = _event_windows(windows, ev["id"])
            xs = [windows[i]["start"] for i in idx]
            for col, axis in enumerate(AXES):
                ax = axes[row, col]
                a = AXES.index(axis)
                for k, theta in enumerate(SWEEP_BARS):
                    rates = [head_stats(Zs[i][:, a], theta, axis).rate_per_100k
                             for i in idx]
                    ys = [fold_change(rates[0], r) for r in rates]
                    ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.4,
                            color=shades[k], markeredgewidth=0,
                            label=f"θ = {theta:g}", zorder=3)
                ax.axhline(1.0, color="#888888", linewidth=0.9, zorder=1)
                ax.axvline(_date(ev["event"]), color="#d62728", linestyle="--",
                           linewidth=1.2, zorder=2)
                _date_ticks(ax, xs, rotate=True)
                if row == 0:
                    ax.set_title(AXIS_LABEL[axis], fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"{ev['id']}\nprevalence / pre-event")

        axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                           fontsize=8, title="bar", title_fontsize=8)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_head_intensity(windows, Zs, path):
    """How extreme the extremes are: the p99.9 of each axis per window.

    The one view here with no denominator at all. Prevalence asks how many accounts
    cleared a bar, and its answer moves when the population moves; this asks how high
    the top of the distribution reached, and 600,000 uninvolved accounts cannot dilute
    it. If an event brought more extreme accounts rather than merely more accounts,
    this is where it shows.
    """
    plt = _mpl()
    events = _panels(windows)
    if not events:
        return

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(events), 1, figsize=(7.2, 2.4 * len(events)),
                                 squeeze=False, sharey=True)
        for row, ev in enumerate(events):
            ax = axes[row, 0]
            idx = _event_windows(windows, ev["id"])
            xs = [windows[i]["start"] for i in idx]
            for a, axis in enumerate(AXES):
                p999 = [head_stats(Zs[i][:, a], 0.0, axis).p999 for i in idx]
                p99 = [head_stats(Zs[i][:, a], 0.0, axis).p99 for i in idx]
                ax.plot(xs, p999, marker=AXIS_MARKER[axis], color=AXIS_COLOUR[axis],
                        label=f"{AXIS_LABEL[axis]} p99.9", linewidth=1.8, markersize=5,
                        markeredgewidth=0, zorder=3)
                ax.plot(xs, p99, marker=AXIS_MARKER[axis], color=AXIS_COLOUR[axis],
                        label=f"{AXIS_LABEL[axis]} p99", linewidth=1.0, markersize=3,
                        linestyle=":", alpha=0.75, markeredgewidth=0, zorder=3)
            _date_ticks(ax, xs)
            ax.axvline(_date(ev["event"]), color="#d62728", linestyle="--",
                       linewidth=1.3, zorder=2)
            ax.set_ylabel(f"{ev['id']}\nArchetype score")

        axes[0, 0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3,
                          frameon=False, fontsize=8, columnspacing=1.6)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_concentration(windows, Zs, path):
    """Gini and the top-1% mass share: the same question with no threshold at all.

    Prevalence and intensity can both be argued with by arguing about where the bar is.
    Concentration cannot -- it reads the whole distribution. If diffusion power
    concentrated into fewer hands after an event, Gini rises and the top 1% hold more of
    the axis, and neither statement depends on a choice anyone made.

    This is the most defensible of the four figures and the least specific: it says the
    shape changed, not who changed it.
    """
    plt = _mpl()
    events = _panels(windows)
    if not events:
        return

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(events), 2, figsize=(9.6, 2.4 * len(events)),
                                 squeeze=False, sharey="col")
        for row, ev in enumerate(events):
            idx = _event_windows(windows, ev["id"])
            xs = [windows[i]["start"] for i in idx]
            for col, (attr, label) in enumerate(
                [("gini", "Gini"), ("top1pct_share", "Top-1% share of axis mass")]
            ):
                ax = axes[row, col]
                for a, axis in enumerate(AXES):
                    ys = [getattr(head_stats(Zs[i][:, a], 0.0, axis), attr) for i in idx]
                    ax.plot(xs, ys, marker=AXIS_MARKER[axis], color=AXIS_COLOUR[axis],
                            label=AXIS_LABEL[axis], linewidth=1.8, markersize=5,
                            markeredgewidth=0, zorder=3)
                _date_ticks(ax, xs, rotate=True)
                ax.axvline(_date(ev["event"]), color="#d62728", linestyle="--",
                           linewidth=1.3, zorder=2)
                ax.set_ylabel(f"{ev['id']}\n{label}" if col == 0 else label)

        axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=3,
                          frameon=False, handlelength=2.2)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def plot_prepost(windows, Zs, bar, path):
    """Pre vs post per event, in the units a rare class is actually read in.

    Replaces the old per-event comparison figure. Two panels:

    (a) prevalence per 100k in the pre-event window, the window straight after the event,
        and the late windows (5 onwards) -- absolute, log y, so the enormous difference
        in how common the three archetypes are is visible rather than normalised away.
    (b) the same as a fold change against the pre-event window, which is the number the
        shock is actually read in: "10 became 15" is 1.5x whatever the base rate.

    Panel (a) alone would show super-spreaders as a sliver next to amplifiers and invite
    "nothing happened". Panel (b) alone would show a 1.5x next to a 1.5x and invite the
    reader to think the two are equally consequential when one is 15 accounts and the
    other 15,000. They are two halves of one claim.
    """
    plt = _mpl()
    events = _panels(windows)
    if not events:
        return

    phases = [("pre", "Pre-event"), ("post", "Post-event"), ("late", "Late (w5+)")]
    hatches = {"pre": "", "post": "///", "late": "..."}

    def rate(idx_list, a, axis):
        if not idx_list:
            return float("nan")
        return float(np.mean([head_stats(Zs[i][:, a], bar, axis).rate_per_100k
                              for i in idx_list]))

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True)
        width = 0.26
        # The tick says only the event. Colour already says which archetype, and
        # spelling it out under every bar collides the labels into unreadable mush --
        # "Super-SpreadAmplifiCoordinated" was the first attempt.
        ticks, labels = [], []
        for e, ev in enumerate(events):
            idx = _event_windows(windows, ev["id"])
            pre, post, late = idx[:1], idx[1:2], idx[4:]
            for a, axis in enumerate(AXES):
                pos = e * (len(AXES) + 1) + a
                vals = {"pre": rate(pre, a, axis), "post": rate(post, a, axis),
                        "late": rate(late, a, axis)}
                for k, (key, _) in enumerate(phases):
                    v = vals[key]
                    if not np.isfinite(v):
                        continue
                    axes[0].bar(pos + (k - 1) * width, max(v, 1e-3), width * 0.92,
                                color=AXIS_COLOUR[axis], alpha=1.0 - 0.22 * k,
                                hatch=hatches[key], edgecolor="white", linewidth=0.4,
                                zorder=3)
                for k, key in enumerate(["post", "late"]):
                    fc = fold_change(vals["pre"], vals[key])
                    if not np.isfinite(fc):
                        continue  # 0 -> n is not a fold change; leave the slot empty
                    axes[1].bar(pos + (k - 0.5) * width, fc, width * 0.92,
                                color=AXIS_COLOUR[axis], alpha=1.0 - 0.22 * (k + 1),
                                hatch=hatches[key], edgecolor="white", linewidth=0.4,
                                zorder=3)
            ticks.append(e * (len(AXES) + 1) + 1)  # centre of the event's group
            labels.append(ev["id"])

        axes[0].set_yscale("log")
        axes[0].set_ylabel(f"Accounts per 100k\nscoring >= {bar:g}")
        axes[1].axhline(1.0, color="#444444", linewidth=1.0, zorder=2)
        axes[1].set_ylabel("Fold change vs pre-event")
        for ax in axes:
            ax.set_axisbelow(True)
        axes[1].set_xticks(ticks)
        axes[1].set_xticklabels(labels)

        phase_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#777777",
                                       alpha=1.0 - 0.22 * k, hatch=hatches[key],
                                       edgecolor="white")
                         for k, (key, _) in enumerate(phases)]
        axis_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=AXIS_COLOUR[a])
                        for a in AXES]
        # Two legends, stacked: colour says which archetype, hatch says which phase.
        # Side by side they are six entries across one panel width and the last
        # archetype lands on top of the first phase.
        #
        # fig.legend, not ax.legend: a second ax.legend() call detaches the first, and
        # add_artist does not survive it -- the archetype legend silently vanished.
        fig.tight_layout()
        fig.legend(axis_handles, [AXIS_LABEL[a] for a in AXES], loc="lower center",
                   bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
        fig.legend(phase_handles, [lab for _, lab in phases], loc="lower center",
                   bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
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
    ap.add_argument("--bar", type=float, default=COMMON_BAR,
                    help="score bar, common to all three axes (default %(default)s). "
                         "Common on purpose: a per-axis quantile equalises the three "
                         "archetypes' prevalence by construction and hides that they are "
                         "rare at different scales. The figures sweep it anyway.")
    ap.add_argument("--events", default="",
                    help="comma-separated event ids to run, e.g. 'E1,E2'. Default: all "
                         "four. E3/E4 need archive passes the E1/E2 cache cannot serve.")
    ap.add_argument("--max-windows", type=int, default=0,
                    help="cap the windows per event. For a quick look at the figures "
                         "before committing to a full run; the numbers in a capped run "
                         "are real but the pooled fit is fitted on less, so do not "
                         "report them.")
    ap.add_argument("--no-log", action="store_true",
                    help="do not log1p the features before scaling. Only for comparison: "
                         "on raw counts one account defines the axis and the other 99.9%% "
                         "are pinned at zero.")
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                    help="drop users whose confidence (data-availability score, "
                         "arles.features.confidence) is below this before fitting and "
                         "plotting (default %(default)s). Confidence combines volume, "
                         "recency and lifespan; below ~0.3 a user has essentially been "
                         "seen act only once or twice, too little to characterise. A "
                         "vector must not be trusted where there is no evidence for it, "
                         "so the gate is on by default; set 0.0 to keep everyone. The "
                         "kept fraction is reported per window.")
    args = ap.parse_args()

    if args.events:
        wanted = {e.strip().upper() for e in args.events.split(",")}
        unknown = wanted - {e["id"] for e in EVENTS}
        if unknown:
            ap.error(f"unknown event id(s): {', '.join(sorted(unknown))}")
        EVENTS[:] = [e for e in EVENTS if e["id"] in wanted]

    if args.max_windows:
        for e in EVENTS:
            e["n_windows"] = min(e.get("n_windows", N_WINDOWS), args.max_windows)
        print(f"PREVIEW: {args.max_windows} window(s) per event. The pooled fit and the "
              f"bar are fitted on this subset, so the numbers are not the study's.\n")

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

    if args.min_confidence > 0.0:
        # Keep only well-observed users. This is what the prior work did; here it is
        # opt-in, and reported, because at a forced-migration influx the low-confidence
        # users ARE the phenomenon. See --min-confidence.
        print(f"\nfiltering users at confidence >= {args.min_confidence:g}")
        for w in windows:
            if not len(w["ids"]):
                continue
            keep = np.asarray(w["conf"]) >= args.min_confidence
            kept, tot = int(keep.sum()), len(w["ids"])
            print(f"  {w['event']} w{w['w'] + 1}: {kept:,}/{tot:,} "
                  f"({100.0 * kept / tot:.1f}%) kept")
            w["ids"] = [u for u, k in zip(w["ids"], keep) if k]
            w["X"] = w["X"][keep]
            w["conf"] = np.asarray(w["conf"])[keep]

    # ONE fit, all 14 windows, then frozen. See the module docstring.
    print("\nfitting the embedding on all windows pooled ...")
    embedder = fit_pooled([w["X"] for w in windows if len(w["ids"])],
                          log_scale=not args.no_log)
    Zs = [embedder.transform(w["X"]) if len(w["ids"]) else np.zeros((0, 3))
          for w in windows]

    for w in embedder.warnings():
        print(f"\n  ! {w}")

    # ONE bar, common to all three axes. Not a per-axis quantile: that puts the same
    # fraction above every axis's bar by construction and reports the three archetypes
    # as equally common, which they are not -- see arles.prevalence.
    bar = args.bar

    (out / "archetype_fit.json").write_text(embedder.to_json())
    (out / "loadings.txt").write_text(embedder.loadings_table() + "\n")
    write_windows_csv(windows, Zs, out / "windows.csv")
    write_features_csv(windows, out / "features.csv")
    write_prevalence_csv(windows, Zs, bar, out / "prevalence.csv")

    # Diagnostics live out of the main directory so the one paper figure is not lost
    # among the many that support it.
    diag = out / "diagnostics"
    diag.mkdir(exist_ok=True)

    # THE paper figure: prevalence per archetype, each on its own scale, all four events.
    plot_prevalence_by_archetype(windows, Zs, bar, out / "fig_archetype_prevalence.pdf")

    # The population-median figure is retired from the main output (kept as a diagnostic).
    # It made one point -- only E1 moves the median user -- but three of its four lines
    # are pinned at exactly zero, which reads as a broken plot rather than a result, and
    # the same point is one sentence of text plus the coordinated panel above. See the
    # guide's note on why it was dropped.
    plot_population_median(windows, Zs, diag / "fig_population_median.pdf")

    # The published figure: mean over every account. Named for what it is -- the mean is
    # dominated by the ~99% of accounts that are no archetype at all, which is why
    # median_superspreader is 0.0000 in every window and why the head figures exist.
    plot_evolution(windows, Zs, diag / "fig_mean_score_per_window.pdf")
    plot_prevalence(windows, Zs, bar, diag / "fig_prevalence.pdf")
    plot_threshold_sweep(windows, Zs, diag / "fig_threshold_sweep.pdf")
    plot_head_intensity(windows, Zs, diag / "fig_head_intensity.pdf")
    plot_concentration(windows, Zs, diag / "fig_concentration.pdf")
    plot_prepost(windows, Zs, bar, diag / "fig_prepost_prevalence.pdf")
    plot_archetype_space(windows, Zs, diag / "fig_archetype_space.pdf")
    if args.cohorts:
        plot_cohorts(windows, Zs, diag / "fig_cohorts.pdf")

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
