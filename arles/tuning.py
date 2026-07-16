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
#: counterpart, so the grid now runs all the way there (see STATIC_DEGENERATE_NOTE).
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

#: The static counterpart of each time-aware metric is already in the grid.
#:
#: At delta = the analysis window there is exactly one slot, so the EMA degenerates and
#: TASH is the plain h-index, TAI the plain influence score. That row therefore *is* the
#: static baseline -- computed by the same code path, over the same window pairs, and so
#: directly comparable. Verified on the archive: the delta=5d row reads 0.7341 and 0.7876,
#: matching the independently computed h_index and influence_score to four decimals.
#:
#: Scoring the baselines separately was redundant, and it was the only thing that needed
#: an alpha of NaN -- which silently broke aggregation, since NaN never equals itself.
STATIC_DEGENERATE_NOTE = (
    "the delta = window row is the static counterpart: one slot, EMA degenerates"
)


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

    # Defensive: NaN never compares equal to itself, so a NaN alpha would give every
    # row its own bucket and silently defeat the averaging. Nothing produces a NaN
    # alpha any more, but the failure mode was invisible in the output when it did.
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


@dataclass
class PairedComparison:
    """The result of comparing two settings across the same window pairs."""

    a: str
    b: str
    diffs: Tuple[float, ...]
    wins: int
    n: int
    mean_diff: float
    sem_diff: float
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def describe(self) -> str:
        direction = "better" if self.mean_diff > 0 else "worse"
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.a} vs {self.b}: {self.mean_diff:+.4f} +/- {self.sem_diff:.4f} "
            f"({direction} on {self.wins}/{self.n} pairs, Wilcoxon p={self.p_value:.3f}, "
            f"{verdict})"
        )


def compare_paired(a: TuningResult, b: TuningResult) -> Optional[PairedComparison]:
    """Compare two settings using the paired design, with magnitudes.

    Every setting is scored on the *same* window pairs, so the scores are paired and the
    between-pair variation -- which is mostly pair difficulty, not setting quality --
    cancels. Measured on 6 pairs, nDCG@100 ranged 0.44..0.90 for every setting alike
    because early August is sparse for all of them; that common difficulty inflates the
    unpaired std to ~0.16 and swamps the ~0.01 differences we care about.

    An earlier version used a bare sign test requiring a 6/6 sweep. That was too crude
    in two ways: it discarded magnitude entirely (a setting winning by +0.02 on five
    pairs and losing by -0.001 on one scored 5/6 and was called "not significant"), and
    with n=6 a sign test cannot return p<0.05 for anything short of unanimity. The
    Wilcoxon signed-rank test uses the size of the differences, not just their sign.

    n=6 is small: the minimum attainable Wilcoxon p is 0.031, so this can detect a
    consistent effect but has little power against a marginal one. That is a real limit
    of the design and worth stating rather than papering over.
    """
    if not a.per_pair or not b.per_pair:
        return None
    if len(a.per_pair) != len(b.per_pair) or len(a.per_pair) < 2:
        return None

    diffs = tuple(x - y for x, y in zip(a.per_pair, b.per_pair))
    n = len(diffs)
    wins = sum(1 for d in diffs if d > 0)
    mean_diff = float(np.mean(diffs))
    sem = float(np.std(diffs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    p = 1.0
    if any(d != 0 for d in diffs):
        try:
            from scipy.stats import wilcoxon

            p = float(wilcoxon(diffs, alternative="two-sided", zero_method="zsplit").pvalue)
        except Exception:
            # Fall back to an exact two-sided sign test.
            from math import comb

            k = min(wins, n - wins)
            p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))

    return PairedComparison(
        a=f"{a.metric}", b=f"{b.metric}", diffs=diffs, wins=wins, n=n,
        mean_diff=mean_diff, sem_diff=sem, p_value=p,
    )


def beats_on_every_pair(best: TuningResult, other: TuningResult) -> bool:
    """Did `best` outscore `other` on every window pair? (sign test, unanimity)

    Retained for the plateau, where a deliberately conservative criterion is wanted:
    a setting is only dropped from the plateau if it loses on every pair. Use
    compare_paired for the question "is A actually better than B", which needs
    magnitudes.
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


def grid_search(
    extractor: FeatureExtractor,
    target: np.ndarray,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    deltas: Sequence[timedelta] = DEFAULT_DELTAS,
    ks: Sequence[int] = (10, 100, 1000),
    progress: bool = True,
) -> Tuple[TuningResult, TuningResult, List[TuningResult]]:
    """Sweep (alpha, delta) for TASH and TAI.

    Returns (best_tash, best_tai, all_results). The extractor's buffer is built once
    and re-scored per grid point; the archive is not re-read.

    Include delta = the analysis window in `deltas` to get each metric's static
    counterpart for free: one slot means the EMA degenerates (see
    STATIC_DEGENERATE_NOTE).
    """
    ts, content, actor, content_author, content_n = extractor.sorted_buffer()
    n_actors = len(extractor._actor_ids)
    origin = extractor.window_start.timestamp()

    results: List[TuningResult] = []
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
