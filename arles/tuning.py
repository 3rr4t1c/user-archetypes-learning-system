"""Re-tune the time-aware parameters (alpha, delta) on this dataset.

Why this exists
---------------
Verdolotti et al. tune the TASH-Index and TAI-Score by grid search over alpha (the EMA
smoothing factor) and delta (the time-slot length), scoring each combination with nDCG
against a target that quantifies "a user's strength in the misinformation reshare
network ... the total number of reshares in which the user is engaged -- either as the
original author whose posts are amplified by others, or as a user who actively reshares
such content". They obtain alpha=0.5, delta=14 days for TASH and alpha=0.6, delta=18
days for TAI.

Those values cannot simply be carried over:

  * delta = 14 days does not fit inside a 5-day analysis window. The EMA would contain
    exactly one term, so TASH would collapse to a plain per-window h-index and the
    "time-aware" part would do nothing at all -- which is exactly what it does on a
    single-slot window.
  * their grid was optimised on a different platform with a different action rate, so
    the argument "reuse their optimum" is weak even where it is arithmetically possible.

But the *procedure* transfers intact. Drop the misinformation restriction from their
target and what remains -- total reshares a user is involved in, as resharer or as
reshared author -- is well defined on any reshare stream, and is already what ArLeS
uses as its confidence volume. So we run their optimisation, unchanged in method, on
this data: rank users by the metric computed on window t, score that ranking against
their involvement in window t+1, and grid-search (alpha, delta).

That makes the parameter choice an empirical result on this dataset rather than an
inherited constant or a round number, which is what a reviewer will ask for.

Predictive, not descriptive
---------------------------
The target is measured on the *next* window. Verdolotti et al. frame the task as
predicting "each user's future contribution to misinformation based on their historical
behavior", and a metric that only described the window it was computed on would be
trivially maximised by the plain influence score.

Cost
----
The (alpha, delta) sweep re-scores an in-memory buffer; the archive is read once per
window, not once per grid point.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .features import FeatureExtractor, WindowIndex, time_aware_scores
from .schema import CanonicalAction

#: Grid searched by default.
#:
#: The first sweep over 6 window pairs put BOTH optima on a grid edge: TASH rose
#: monotonically in alpha to the 0.9 boundary (0.630 -> 0.750 at delta=1h) and TAI rose
#: monotonically in delta to the 2d boundary (0.622 -> 0.797 at alpha=0.4). An argmax on
#: an edge is not an optimum, it is where the grid stopped. Both directions point the
#: same way -- longer memory, fewer slots -- i.e. towards the metric's own static
#: counterpart, so the grid now runs all the way there and the baselines are scored
#: alongside (see STATIC_BASELINES).
DEFAULT_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
DEFAULT_DELTAS = (
    timedelta(minutes=15),
    timedelta(minutes=30),
    timedelta(hours=1),
    timedelta(hours=3),
    timedelta(hours=6),
    timedelta(hours=12),
    timedelta(days=1),
    timedelta(days=2),
    timedelta(days=5),  # one slot for a 5-day window: the EMA degenerates to static
)

#: The static rankings the time-aware ones must beat to justify their existence.
#:
#: Verdolotti et al. evaluate the TASH-Index and TAI-Score against exactly these, and
#: report that TASH closely follows their ML model while the H-Index and Influence Score
#: "consistently underperform". That comparison has to be repeated here rather than
#: assumed: if influence_score alone matches TAI, the time-aware variant is buying
#: nothing on this data and does not deserve a slot in the feature bucket.
STATIC_BASELINES = ("influence_score", "h_index")


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """nDCG@k of the ranking induced by `scores`, graded by `relevance`.

    Ties are broken deterministically (by index) so a re-run reproduces the number.
    """
    if scores.shape[0] == 0 or k <= 0:
        return 0.0
    k = min(k, scores.shape[0])

    order = np.lexsort((np.arange(scores.shape[0]), -scores))[:k]
    gains = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))

    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: ideal.shape[0]]))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class TuningResult:
    metric: str
    alpha: float
    delta: timedelta
    ndcg: Dict[int, float]
    #: Standard deviation of nDCG across window pairs, when fitted on more than one.
    ndcg_std: Dict[int, float] = None
    #: How many (t, t+1) pairs this was averaged over.
    n_pairs: int = 1
    #: Per-pair nDCG@100, for inspecting stability.
    per_pair: Tuple[float, ...] = ()

    @property
    def score(self) -> float:
        """The figure optimised: nDCG@100, the top-k regime the paper cares about."""
        return self.ndcg.get(100, 0.0)

    @property
    def std(self) -> float:
        return (self.ndcg_std or {}).get(100, 0.0)


def aggregate(per_pair_results: Sequence[Sequence[TuningResult]]) -> List[TuningResult]:
    """Average a grid across several (t, t+1) window pairs.

    Fitting on a single period overfits it. Measured on this archive: the TASH optimum
    is (alpha=0.7, delta=15min) on the E1 pair and (alpha=0.5, delta=2d) on the E2 pair
    -- a 192x disagreement in delta, from the same dataset. Both are noise on a surface
    that is flat to within ~0.01 nDCG across a wide region, and adopting either one
    costs the other period real accuracy. Parameters belong to the dataset, so they are
    fitted across pairs spanning the whole observation period.
    """
    if not per_pair_results:
        return []

    # NaN never compares equal to itself, so a NaN alpha silently gives every row its
    # own bucket. That is exactly what happened: the static baselines carry alpha=NaN,
    # so a 6-pair run reported influence_score twelve times instead of averaging it
    # twice, and the paired comparison then zipped a 6-tuple against a 1-tuple and
    # truncated to one. Key on a hashable, self-equal sentinel instead.
    def key(r: TuningResult):
        alpha = "static" if r.alpha != r.alpha else r.alpha  # NaN check
        return (r.metric, alpha, r.delta)

    buckets: Dict[Tuple[str, object, timedelta], List[TuningResult]] = defaultdict(list)
    for pair in per_pair_results:
        for r in pair:
            buckets[key(r)].append(r)

    out: List[TuningResult] = []
    for rows in buckets.values():
        ks = sorted(rows[0].ndcg)
        mean = {k: float(np.mean([r.ndcg[k] for r in rows])) for k in ks}
        std = {k: float(np.std([r.ndcg[k] for r in rows])) for k in ks}
        out.append(
            TuningResult(
                # Take metric/alpha/delta from the row, not the key: the key's alpha is
                # a sentinel for the NaN of a static baseline, not a real value.
                metric=rows[0].metric,
                alpha=rows[0].alpha,
                delta=rows[0].delta,
                ndcg=mean,
                ndcg_std=std,
                n_pairs=len(rows),
                per_pair=tuple(r.ndcg.get(100, 0.0) for r in rows),
            )
        )
    return out


def beats_on_every_pair(best: TuningResult, other: TuningResult) -> bool:
    """Did `best` outscore `other` on every single window pair?

    A sign test, which is the right instrument for this design and this n. The same
    window pairs score every setting, so their scores are paired -- and the variation
    between pairs is mostly pair *difficulty*, not setting quality. Measured on 6 pairs:
    nDCG@100 was 0.442, 0.898, 0.607, 0.871, 0.845, 0.834 for TASH and 0.494, 0.902,
    0.701, 0.920, 0.882, 0.885 for TAI -- pair 1 is the outlier for both, because early
    August is sparse for every setting alike.

    That common difficulty inflates the unpaired standard deviation to ~0.16, which
    swamps the ~0.01 differences between settings and makes "mean +/- std" useless: it
    would call 43 of 80 settings indistinguishable. Paired, it cancels.

    With n = 6, winning all six is p = 2^-6 = 0.016 one-sided; five of six is p = 0.11,
    which is not evidence. So the bar is a clean sweep.
    """
    if not best.per_pair or not other.per_pair:
        return False
    if len(best.per_pair) != len(other.per_pair):
        return False
    if best is other:
        return False
    return all(b > o for b, o in zip(best.per_pair, other.per_pair))


def plateau(
    results: Sequence[TuningResult],
    metric: str,
    tolerance: float = 0.01,
    paired: bool = True,
):
    """The settings that cannot be distinguished from the best.

    With `paired` (the default, and correct when every setting was scored on the same
    window pairs), a setting is excluded only if the best beats it on *every* pair.
    Everything else is in the plateau: the data does not separate them.

    Without per-pair data, falls back to an absolute `tolerance` -- which is what the
    first version did, and it reported "plateau: 2/80" on a surface whose standard
    error was 0.068, i.e. 16x the tolerance. That was false precision: the honest
    answer was 43/80.
    """
    rows = [r for r in results if r.metric == metric]
    if not rows:
        return [], None
    best = max(rows, key=lambda r: r.score)

    if paired and best.per_pair and len(best.per_pair) > 1:
        near = [r for r in rows if r is best or not beats_on_every_pair(best, r)]
    else:
        near = [r for r in rows if best.score - r.score <= tolerance]
    return sorted(near, key=lambda r: -r.score), best


def involvement_target(
    actions: Iterable[CanonicalAction], user_ids: Sequence[str]
) -> np.ndarray:
    """Reshares each user is involved in, as resharer or as reshared author.

    The content-agnostic form of the target in Verdolotti et al. Users absent from the
    window score 0 -- that is a real prediction failure, not missing data.
    """
    index = {u: i for i, u in enumerate(user_ids)}
    target = np.zeros(len(user_ids), dtype=np.float64)
    for action in actions:
        if action.activity_type != "repost" or action.is_self_reshare:
            continue
        i = index.get(action.actor_id)
        if i is not None:
            target[i] += 1.0
        if action.parent_actor_id:
            j = index.get(action.parent_actor_id)
            if j is not None:
                target[j] += 1.0
    return target


def baseline_results(
    extractor: FeatureExtractor,
    target: np.ndarray,
    ks: Sequence[int] = (10, 100, 1000),
) -> List[TuningResult]:
    """Score the static rankings the time-aware metrics have to beat.

    Reported as TuningResults with alpha/delta of None so they sit alongside the grid.
    Without these the sweep can only say which TASH is the best TASH -- never whether
    any TASH is worth having.
    """
    from .features import FEATURE_NAMES

    _, X = extractor.finish()
    out: List[TuningResult] = []
    for name in STATIC_BASELINES:
        col = X[:, FEATURE_NAMES.index(name)]
        out.append(
            TuningResult(
                metric=name,
                alpha=float("nan"),
                delta=timedelta(0),
                ndcg={k: ndcg_at_k(col, target, k) for k in ks},
            )
        )
    return out


def grid_search(
    extractor: FeatureExtractor,
    target: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    deltas: Sequence[timedelta] = DEFAULT_DELTAS,
    ks: Sequence[int] = (10, 100, 1000),
    progress: bool = True,
    include_baselines: bool = True,
) -> Tuple[TuningResult, TuningResult, List[TuningResult]]:
    """Sweep (alpha, delta) for TASH and TAI, plus the static baselines.

    Returns (best_tash, best_tai, all_results). The extractor's buffer is built once
    and re-scored per grid point; the archive is not re-read.
    """
    ts, content, actor, content_author, content_n = extractor.sorted_buffer()
    n_actors = len(extractor._actor_ids)
    origin = extractor.window_start.timestamp()

    results: List[TuningResult] = []
    if include_baselines:
        results.extend(baseline_results(extractor, target, ks))
    total = len(alphas) * len(deltas)
    done = 0

    for delta in deltas:
        for alpha in alphas:
            tai, tash = time_aware_scores(
                ts, content, content_author, n_actors,
                origin=origin,
                slot_seconds=delta.total_seconds(),
                tash_alpha=alpha,
                tai_alpha=alpha,
            )
            for name, scores in (("tash_index", tash), ("tai_score", tai)):
                results.append(
                    TuningResult(
                        metric=name,
                        alpha=alpha,
                        delta=delta,
                        ndcg={k: ndcg_at_k(scores, target, k) for k in ks},
                    )
                )
            done += 1
            if progress:
                print(
                    f"  [{done:>3}/{total}] delta={_fmt_delta(delta):<8} alpha={alpha:.1f}"
                    f"  nDCG@100  TASH={results[-2].score:.4f}  TAI={results[-1].score:.4f}",
                    flush=True,
                )

    best_tash = max((r for r in results if r.metric == "tash_index"), key=lambda r: r.score)
    best_tai = max((r for r in results if r.metric == "tai_score"), key=lambda r: r.score)
    return best_tash, best_tai, results


def format_baselines(results: Sequence[TuningResult]) -> str:
    """The comparison that decides whether the time-aware features earn their slot."""
    rows = [r for r in results if r.metric in STATIC_BASELINES]
    if not rows:
        return ""
    out = ["Static baselines vs the best time-aware setting", ""]
    n_pairs = rows[0].n_pairs
    for r in sorted(rows, key=lambda r: -r.score):
        line = f"  {r.metric:<16} nDCG@100={r.score:.4f}"
        if n_pairs > 1:
            line += f" +/- {r.std:.4f}"
        out.append(line)

    for metric in ("tash_index", "tai_score"):
        near, best = plateau(results, metric)
        if best is None:
            continue
        out.append(f"  {metric:<16} nDCG@100={best.score:.4f} "
                   f"(alpha={best.alpha:.2f}, delta={_fmt_delta(best.delta)})")
        for b in rows:
            if not b.per_pair or not best.per_pair:
                continue
            wins = sum(1 for x, y in zip(best.per_pair, b.per_pair) if x > y)
            n = len(best.per_pair)
            verdict = (
                f"beats {b.metric} on {wins}/{n} pairs"
                if wins == n
                else f"does NOT consistently beat {b.metric} ({wins}/{n} pairs)"
            )
            out.append(f"      -> {verdict}")
    out.append("")
    out.append("  If a time-aware metric does not consistently beat its static")
    out.append("  counterpart, it is not earning its place in the feature bucket and")
    out.append("  the paper should say so rather than include it for symmetry.")
    return "\n".join(out)


def _fmt_delta(d: timedelta) -> str:
    s = d.total_seconds()
    if s < 3600:
        return f"{int(s // 60)}min"
    if s < 86400:
        return f"{s / 3600:g}h"
    return f"{s / 86400:g}d"


def format_grid(
    results: Sequence[TuningResult], metric: str, tolerance: float = 0.01
) -> str:
    """The (alpha, delta) nDCG@100 surface, as a text table.

    The analogue of the optimisation grid figure in Verdolotti et al. Printing the
    whole surface, not just the argmax, is what lets a reader see whether the optimum
    is a plateau or a spike -- and therefore how much the choice actually matters.

    Cells within `tolerance` of the best are marked '+', the best '*'. If the '+'
    region is large, the argmax is not meaningful and the paper should say so.
    """
    rows = [r for r in results if r.metric == metric]
    if not rows:
        return ""
    alphas = sorted({r.alpha for r in rows})
    deltas = sorted({r.delta for r in rows})
    n_pairs = rows[0].n_pairs

    near, best = plateau(rows, metric, tolerance)
    near_set = {(r.alpha, r.delta) for r in near}

    header = f"nDCG@100 surface for {metric}"
    if n_pairs > 1:
        header += f"  (mean of {n_pairs} window pairs)"
    out = [header, ""]
    out.append("  delta \\ alpha  " + "".join(f"{a:>8.2f}" for a in alphas))
    for d in deltas:
        cells = []
        for a in alphas:
            hit = next((r for r in rows if r.alpha == a and r.delta == d), None)
            v = hit.score if hit else float("nan")
            if hit is best:
                mark = "*"
            elif (a, d) in near_set:
                mark = "+"
            else:
                mark = " "
            cells.append(f"{v:>7.4f}{mark}")
        out.append(f"  {_fmt_delta(d):<14}" + "".join(cells))
    out.append("")
    out.append(
        f"  best: alpha={best.alpha:.1f}, delta={_fmt_delta(best.delta)}, "
        f"nDCG@100={best.score:.4f}"
        + (f" +/- {best.std:.4f} across pairs" if n_pairs > 1 else "")
    )
    criterion = (
        "not beaten by the best on every window pair"
        if (best.per_pair and len(best.per_pair) > 1)
        else f"within {tolerance:.3f} nDCG of the best"
    )
    out.append(f"  plateau: {len(near)}/{len(rows)} settings {criterion} (marked + above)")
    if len(near) > max(3, len(rows) // 10):
        d_lo, d_hi = min(r.delta for r in near), max(r.delta for r in near)
        a_lo, a_hi = min(r.alpha for r in near), max(r.alpha for r in near)
        out.append(
            f"  => the surface is FLAT: delta {_fmt_delta(d_lo)}..{_fmt_delta(d_hi)} "
            f"and alpha {a_lo:.2f}..{a_hi:.2f} are all indistinguishable from the best."
        )
        out.append(
            "     Report the region and the chosen value, not a bare argmax: a re-run "
            "would move it."
        )
    return "\n".join(out)
