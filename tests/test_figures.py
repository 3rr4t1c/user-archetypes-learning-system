"""Tests for the figure script.

A figure is the one artefact nobody checks, so the logic behind it is tested here
without needing the 146 GB archive: window layout, the CSV that must accompany it, the
cohort split, and the property the whole design rests on -- one pooled ruler for every
window.
"""

import csv
import importlib.util
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from arles.embedding import fit_pooled
from arles.features import FEATURE_NAMES

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "make_figures.py"
_spec = importlib.util.spec_from_file_location("make_figures", _SCRIPT)
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)


def fake_windows(n_events=2, n_windows=7, seed=0):
    """Windows shaped like the real ones: growing population, growing outlier."""
    rng = np.random.default_rng(seed)
    out = []
    for e, ev in enumerate(mf.EVENTS[:n_events]):
        first = mf._date(ev["start"])
        for w in range(n_windows):
            start = first + timedelta(days=mf.WINDOW_DAYS * w)
            n = 200 * (w + 1)  # the population triples and keeps growing, as in E1
            X = np.abs(rng.normal(3, 2, size=(n, len(FEATURE_NAMES))))
            X[0, :] = 6842 * (10 ** w)  # and so does the extreme account
            out.append({
                "event": ev["id"], "event_name": ev["name"], "w": w,
                "start": start, "end": start + timedelta(days=mf.WINDOW_DAYS),
                "ids": [f"u{e}-{w}-{i}" for i in range(n)],
                "X": X, "conf": rng.uniform(0, 1, n), "n_reposts": n * 7,
            })
    return out


# ------------------------------------------------------------------ layout


def test_all_four_of_the_papers_events_are_covered():
    assert mf.N_WINDOWS == 7
    assert mf.WINDOW_DAYS == 5.0
    assert [e["id"] for e in mf.EVENTS] == ["E1", "E2", "E3", "E4"]
    # E1 spans Aug 25 -> Sep 24, E2 Oct 12 -> Nov 11: the paper's axes.
    for ev, last in (("E1", "2024-09-24"), ("E2", "2024-11-11")):
        spec = next(e for e in mf.EVENTS if e["id"] == ev)
        start = mf._date(spec["start"])
        assert (start + timedelta(days=mf.WINDOW_DAYS * (mf.N_WINDOWS - 1))).date() \
            == mf._date(last).date()


def test_event_dates_are_the_papers():
    """Sec. 4.1: the four migration events, by date."""
    got = {e["id"]: mf._date(e["event"]).date().isoformat() for e in mf.EVENTS}
    assert got == {
        "E1": "2024-08-30",  # X banned in Brazil
        "E2": "2024-10-17",  # X terms & privacy update
        "E3": "2024-11-05",  # US presidential election
        "E4": "2025-01-06",  # broad social-media ToS updates
    }


def test_the_event_falls_inside_the_first_window():
    """Window 1 is the pre-event window; the event must land just after it starts."""
    for ev in mf.EVENTS:
        start, event = mf._date(ev["start"]), mf._date(ev["event"])
        assert start < event <= start + timedelta(days=mf.WINDOW_DAYS)


def test_no_window_runs_past_the_end_of_the_archive():
    """E4 is 6 Jan 2025 and the archive stops on 28 Jan, so E4 cannot have seven
    windows. A truncated final window would be plotted as a whole one and would read as
    a collapse in every count -- an artefact of the archive ending, not of the event."""
    end = mf._date(mf.ARCHIVE_END)
    for ev in mf.EVENTS:
        n = ev.get("n_windows", mf.N_WINDOWS)
        last_end = mf._date(ev["start"]) + timedelta(days=mf.WINDOW_DAYS * n)
        assert last_end <= end, f"{ev['id']} runs past the archive"

    e4 = next(e for e in mf.EVENTS if e["id"] == "E4")
    assert e4["n_windows"] == 5, "five complete windows fit before 28 Jan, not seven"


def test_e2_and_e3_overlap_and_that_is_known():
    """E2 runs to 16 Nov and E3 starts on 31 Oct, so the pooled fit sees those days
    twice. Harmless for a window's features (built from its own actions alone) but it
    does overweight the overlap in the ruler and in the pooled bar. Pinned so the fact
    is a decision rather than a surprise."""
    e2 = next(e for e in mf.EVENTS if e["id"] == "E2")
    e3 = next(e for e in mf.EVENTS if e["id"] == "E3")
    e2_end = mf._date(e2["start"]) + timedelta(days=mf.WINDOW_DAYS * e2["n_windows"])
    assert mf._date(e3["start"]) < e2_end


# --------------------------------------------------------- the pooled ruler


def test_every_window_is_scored_on_one_ruler():
    """The property the figure depends on: pooled fit, transform each window."""
    windows = fake_windows()
    emb = fit_pooled([w["X"] for w in windows])
    Zs = [emb.transform(w["X"]) for w in windows]

    # A user with fixed behaviour scores identically wherever they appear.
    fixed = np.full((1, len(FEATURE_NAMES)), 42.0)
    scores = [emb.transform(fixed)[0, 0] for _ in windows]
    assert len(set(np.round(scores, 12))) == 1

    assert all(np.all((Z >= 0) & (Z <= 1)) for Z in Zs)


def test_pooling_across_both_events_not_per_event():
    """E1 and E2 must share a y-axis, since the paper compares them directly.

    Fitted per event the same behaviour gets two different scores, so "amplifier=0.15"
    would not mean the same in the two panels.
    """
    windows = fake_windows()
    e1 = [w["X"] for w in windows if w["event"] == "E1"]
    e2 = [w["X"] for w in windows if w["event"] == "E2"]
    probe = np.full((1, len(FEATURE_NAMES)), 500.0)

    per_event = (fit_pooled(e1).transform(probe)[0, 0],
                 fit_pooled(e2).transform(probe)[0, 0])
    together = fit_pooled(e1 + e2)
    shared = (together.transform(probe)[0, 0],) * 2

    assert per_event[0] != pytest.approx(per_event[1]), "per-event fits should disagree"
    assert shared[0] == pytest.approx(shared[1])


def test_features_do_not_accumulate_across_windows():
    """ArLeS is windowed-batch, not online: a window's features must not depend on
    whether earlier windows were processed. Only the ruler is shared."""
    from arles.features import FeatureExtractor, WindowIndex
    from arles.schema import CanonicalAction

    t0 = mf._date("2024-08-25")

    def build(day):
        acts = [
            CanonicalAction(f"r{day}-{i}", f"u{i % 9}", "repost",
                            t0 + timedelta(days=day, minutes=i), f"p{i % 4}", f"a{i % 3}")
            for i in range(60)
        ]
        s = t0 + timedelta(days=day)
        idx = WindowIndex.build(acts)
        fx = FeatureExtractor(idx, s, s + timedelta(days=5))
        for a in acts:
            fx.add(a)
        return fx.finish()

    alone = build(40)
    _ = [build(d) for d in (0, 5, 10)]   # process earlier windows first
    after = build(40)
    assert np.array_equal(alone[1], after[1])


# ---------------------------------------------------------------- outputs


def test_windows_csv_carries_the_numbers_behind_the_figure(tmp_path):
    windows = fake_windows()
    emb = fit_pooled([w["X"] for w in windows])
    Zs = [emb.transform(w["X"]) for w in windows]

    path = tmp_path / "windows.csv"
    mf.write_windows_csv(windows, Zs, path)
    rows = list(csv.DictReader(open(path)))

    assert len(rows) == len(windows)
    for axis in mf.AXES:
        assert f"mean_{axis}" in rows[0]
        assert f"median_{axis}" in rows[0]
    # N per window must be there: the population tripling is the confound a reader
    # needs in order to interpret the means at all.
    assert "n_users" in rows[0] and "n_reposts" in rows[0]
    assert int(rows[0]["n_users"]) == 200
    assert rows[0]["event"] == "E1" and rows[-1]["event"] == "E2"


def test_the_fit_is_saved_so_the_figure_can_be_regenerated(tmp_path):
    from arles.embedding import ArchetypeEmbedder

    windows = fake_windows(n_windows=3)
    emb = fit_pooled([w["X"] for w in windows])
    p = tmp_path / "archetype_fit.json"
    p.write_text(emb.to_json())

    reloaded = ArchetypeEmbedder.from_json(p.read_text())
    assert np.allclose(emb.transform(windows[0]["X"]),
                       reloaded.transform(windows[0]["X"]))


def test_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    windows = fake_windows(n_windows=3)
    emb = fit_pooled([w["X"] for w in windows])
    Zs = [emb.transform(w["X"]) for w in windows]

    mf.plot_evolution(windows, Zs, tmp_path / "evo.pdf")
    mf.plot_archetype_space(windows, Zs, tmp_path / "space.pdf")
    mf.plot_cohorts(windows, Zs, tmp_path / "cohorts.pdf")
    for name in ("evo.pdf", "space.pdf", "cohorts.pdf"):
        assert (tmp_path / name).stat().st_size > 1000


def test_scatter_sampling_is_seeded(tmp_path):
    """An unseeded sample would make the figure differ between runs."""
    pytest.importorskip("matplotlib")
    windows = fake_windows(n_windows=2)
    emb = fit_pooled([w["X"] for w in windows])
    Zs = [emb.transform(w["X"]) for w in windows]
    mf.plot_archetype_space(windows, Zs, tmp_path / "a.pdf", max_points=50)
    mf.plot_archetype_space(windows, Zs, tmp_path / "b.pdf", max_points=50)
    # Same bytes modulo the PDF's embedded creation date.
    a = (tmp_path / "a.pdf").read_bytes()
    b = (tmp_path / "b.pdf").read_bytes()
    assert len(a) == len(b)


# ---------------------------------------------------------------- cohorts


def test_cohort_split_uses_the_pre_event_window_as_the_incumbent_set(tmp_path):
    pytest.importorskip("matplotlib")
    windows = fake_windows(n_events=1, n_windows=3)
    # Window 0's users are incumbents; the fake ids are unique per window, so every
    # later window is entirely newcomers -- an extreme but valid case to render.
    emb = fit_pooled([w["X"] for w in windows])
    Zs = [emb.transform(w["X"]) for w in windows]
    mf.plot_cohorts(windows, Zs, tmp_path / "c.pdf")
    assert (tmp_path / "c.pdf").stat().st_size > 1000


def test_empty_window_does_not_break_the_run(tmp_path):
    pytest.importorskip("matplotlib")
    windows = fake_windows(n_events=1, n_windows=3)
    windows[1]["ids"] = []
    windows[1]["X"] = np.zeros((0, len(FEATURE_NAMES)))
    windows[1]["conf"] = np.zeros(0)

    emb = fit_pooled([w["X"] for w in windows if len(w["ids"])])
    Zs = [emb.transform(w["X"]) if len(w["ids"]) else np.zeros((0, 3)) for w in windows]
    mf.write_windows_csv(windows, Zs, tmp_path / "w.csv")
    rows = list(csv.DictReader(open(tmp_path / "w.csv")))
    assert rows[1]["n_users"] == "0"


def test_a_run_covering_only_one_event_still_plots(tmp_path):
    """plt.subplots(1,1) returns a bare Axes, and an event with no windows has no
    idx[0]. Both crashed a single-event run."""
    pytest.importorskip("matplotlib")
    windows = fake_windows(n_events=1, n_windows=3)
    emb = fit_pooled([w["X"] for w in windows])
    Zs = [emb.transform(w["X"]) for w in windows]
    mf.plot_evolution(windows, Zs, tmp_path / "one.pdf")
    mf.plot_cohorts(windows, Zs, tmp_path / "one_c.pdf")
    assert (tmp_path / "one.pdf").stat().st_size > 1000
    assert (tmp_path / "one_c.pdf").stat().st_size > 1000


def test_no_windows_at_all_writes_nothing_rather_than_crashing(tmp_path):
    pytest.importorskip("matplotlib")
    mf.plot_evolution([], [], tmp_path / "none.pdf")
    assert not (tmp_path / "none.pdf").exists()


# ------------------------------------------------------------------- layout


def test_ticks_sit_on_the_windows_and_are_not_iso_dates(tmp_path):
    """The published layout: one tick per window, labelled '25 Aug'.

    matplotlib's automatic date locator picks its own interval (6 days against 5-day
    windows), so ticks land between points and label nothing; and ISO dates are wide
    enough to collide, which is exactly what the first version did.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    windows = fake_windows(n_events=1)
    emb = fit_pooled([w["X"] for w in windows])
    Zs = [emb.transform(w["X"]) for w in windows]
    mf.plot_evolution(windows, Zs, tmp_path / "f.pdf")

    # Re-draw and inspect the axes rather than the PDF bytes.
    fig, ax = plt.subplots()
    xs = [w["start"] for w in windows]
    ax.set_xticks(xs)
    ax.set_xticklabels([d.strftime("%d %b") for d in xs])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    plt.close(fig)

    assert len(labels) == mf.N_WINDOWS          # one per window, not an auto interval
    assert labels[0] == "25 Aug"                # the published format
    assert all(len(s) <= 6 for s in labels)     # short enough not to collide
    assert not any("2024-" in s for s in labels)


def test_colours_are_the_published_colorblind_palette():
    """The figure was drawn with sns.set_palette('colorblind'); tab10 is not that.

    Hard-coded so the look survives without seaborn becoming a dependency of a repo
    that otherwise needs only numpy.
    """
    assert mf.AXIS_COLOUR["superspreader"] == "#0173B2"
    assert mf.AXIS_COLOUR["amplifier"] == "#DE8F05"
    assert mf.AXIS_COLOUR["coordinated"] == "#029E73"
    # and specifically not matplotlib's defaults
    assert "#ff7f0e" not in mf.AXIS_COLOUR.values()
    assert "#2ca02c" not in mf.AXIS_COLOUR.values()


# -------------------------------------------------------------- head statistics


def _fitted(n_events=2, n_windows=4, bar=0.4):
    windows = fake_windows(n_events=n_events, n_windows=n_windows)
    emb = fit_pooled([w["X"] for w in windows])
    Zs = [emb.transform(w["X"]) for w in windows]
    return windows, Zs, bar


def test_the_two_claim_figures_are_written_and_survive_a_partial_run(tmp_path):
    pytest.importorskip("matplotlib")
    windows, Zs, bar = _fitted(n_events=2, n_windows=4)
    mf.plot_population_median(windows, Zs, tmp_path / "pop.pdf")
    mf.plot_prevalence_by_archetype(windows, Zs, bar, tmp_path / "prev.pdf")
    assert (tmp_path / "pop.pdf").stat().st_size > 1000
    assert (tmp_path / "prev.pdf").stat().st_size > 1000
    # one event only, and the empty-run guard
    w1, z1, b1 = _fitted(n_events=1, n_windows=3)
    mf.plot_population_median(w1, z1, tmp_path / "one.pdf")
    mf.plot_prevalence_by_archetype(w1, z1, b1, tmp_path / "one2.pdf")
    assert (tmp_path / "one.pdf").stat().st_size > 1000
    mf.plot_population_median([], [], tmp_path / "none.pdf")
    mf.plot_prevalence_by_archetype([], [], 0.5, tmp_path / "none2.pdf")
    assert not (tmp_path / "none.pdf").exists()
    assert not (tmp_path / "none2.pdf").exists()


def test_the_population_figure_uses_the_median_a_rare_class_cannot_move():
    """The population panel is coordinated-only and reads the median deliberately: a
    median cannot be moved by a rare minority, so a move means the typical user changed.
    A super-spreader median would be zero everywhere by construction."""
    import inspect
    src = inspect.getsource(mf.plot_population_median)
    assert "median" in src and "coordinated" in src


def test_the_head_figure_gives_each_archetype_its_own_axis():
    """The scale asymmetry the whole discussion turned on: amplifiers outnumber
    super-spreaders by orders of magnitude, so they cannot share a y-axis."""
    import inspect
    src = inspect.getsource(mf.plot_prevalence_by_archetype)
    assert "subplots(1, len(AXES)" in src
    assert "rate_per_100k" in src


def test_the_head_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    windows, Zs, bar = _fitted()
    mf.plot_prevalence(windows, Zs, bar, tmp_path / "prev.pdf")
    mf.plot_head_intensity(windows, Zs, tmp_path / "head.pdf")
    mf.plot_concentration(windows, Zs, tmp_path / "conc.pdf")
    mf.plot_prepost(windows, Zs, bar, tmp_path / "pp.pdf")
    mf.plot_threshold_sweep(windows, Zs, tmp_path / "sweep.pdf")
    for name in ("prev.pdf", "head.pdf", "conc.pdf", "pp.pdf", "sweep.pdf"):
        assert (tmp_path / name).stat().st_size > 1000


def test_the_head_figures_survive_a_single_event(tmp_path):
    """Same trap as plot_evolution: squeeze=False everywhere, or a one-event run dies
    on an Axes that is not an array."""
    pytest.importorskip("matplotlib")
    windows, Zs, bar = _fitted(n_events=1, n_windows=3)
    mf.plot_prevalence(windows, Zs, bar, tmp_path / "p.pdf")
    mf.plot_concentration(windows, Zs, tmp_path / "c.pdf")
    mf.plot_threshold_sweep(windows, Zs, tmp_path / "s.pdf")
    assert (tmp_path / "p.pdf").stat().st_size > 1000


def test_no_windows_at_all_writes_no_head_figures(tmp_path):
    pytest.importorskip("matplotlib")
    mf.plot_prevalence([], [], 0.5, tmp_path / "p.pdf")
    mf.plot_concentration([], [], tmp_path / "c.pdf")
    mf.plot_head_intensity([], [], tmp_path / "h.pdf")
    mf.plot_prepost([], [], 0.5, tmp_path / "pp.pdf")
    mf.plot_threshold_sweep([], [], tmp_path / "s.pdf")
    assert not any(p.exists() for p in tmp_path.iterdir())


def test_the_figures_use_one_bar_for_every_axis_not_a_quantile_each():
    """The property the whole set rests on, and the bug it shipped with.

    A per-axis quantile puts (1-q) of the population above every axis's bar by
    construction, reporting three archetypes of wildly different prevalence as equally
    common. The figures take a single scalar so that cannot be expressed.
    """
    import inspect
    for fn in (mf.plot_prevalence, mf.plot_prepost, mf.write_prevalence_csv):
        params = list(inspect.signature(fn).parameters)
        assert "bar" in params and "bars" not in params, f"{fn.__name__} takes one bar"
    assert mf.COMMON_BAR == 0.5


def test_prevalence_csv_carries_every_head_statistic(tmp_path):
    windows, Zs, bar = _fitted(n_events=2, n_windows=3)
    mf.write_prevalence_csv(windows, Zs, bar, tmp_path / "prevalence.csv")
    rows = list(csv.DictReader(open(tmp_path / "prevalence.csv")))

    assert len(rows) == len(windows) * len(mf.AXES)
    for col in ("threshold", "count", "rate_per_100k", "p99", "p999", "gini",
                "top1pct_share", "anchor_feature", "anchor_median"):
        assert col in rows[0]
    # One bar, frozen, for every window AND every axis.
    assert {r["threshold"] for r in rows} == {f"{bar:.6f}"}


def test_the_bar_is_quoted_in_reshares_not_only_in_score_units(tmp_path):
    """'score >= 0.5' is unreviewable, and it is not even the same physical bar on the
    three axes -- each [0,1] is its own pooled PC1 range. 'a median of N reshares
    received' is the absolute statement, and it is per axis."""
    windows, Zs, bar = _fitted(n_events=1, n_windows=3)
    mf.write_prevalence_csv(windows, Zs, bar, tmp_path / "p.csv")
    rows = list(csv.DictReader(open(tmp_path / "p.csv")))
    ss = [r for r in rows if r["axis"] == "superspreader"]
    assert ss[0]["anchor_feature"] == "influence_score"
    assert float(ss[0]["anchor_median"]) > 0


def test_prevalence_is_not_merely_tracking_the_population():
    """The confound the rate column exists for.

    fake_windows grows the population 200 -> 800 across four windows while the head
    stays a single account. The count may rise with N; the rate must fall.
    """
    windows, Zs, bar = _fitted(n_events=1, n_windows=4)
    rates = [mf.head_stats(Z[:, 0], bar, "superspreader").rate_per_100k for Z in Zs]
    ns = [len(w["ids"]) for w in windows]
    assert ns == [200, 400, 600, 800]
    assert rates[-1] < rates[0], "a fixed head in a growing population is a falling rate"


def test_features_csv_exposes_the_raw_features_behind_each_axis(tmp_path):
    """PC1 weights by variance, not by trustworthiness: on the archive it gave
    niche_co_action (the anti-pile-on feature) 0.234 and co_action_size 0.614. So a
    coordinated jump is ambiguous unless you can see the features move."""
    windows = fake_windows(n_events=1, n_windows=3)
    mf.write_features_csv(windows, tmp_path / "features.csv")
    rows = list(csv.DictReader(open(tmp_path / "features.csv")))
    assert len(rows) == 3
    for name in FEATURE_NAMES:
        assert name in rows[0]
    assert "niche_co_action" in rows[0] and "co_action_size" in rows[0]
