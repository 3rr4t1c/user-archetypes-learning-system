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

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .features import FeatureExtractor, WindowIndex, time_aware_scores
from .schema import CanonicalAction

#: Grid searched by default.
#:
#: delta is capped well below a 5-day window: at delta >= the window the EMA has one
#: term and the metric stops being time-aware. The upper end (2 days) still leaves only
#: 2-3 slots, and is included so the sweep can show that rather than assume it.
DEFAULT_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_DELTAS = (
    timedelta(minutes=15),
    timedelta(minutes=30),
    timedelta(hours=1),
    timedelta(hours=3),
    timedelta(hours=6),
    timedelta(hours=12),
    timedelta(days=1),
    timedelta(days=2),
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

    @property
    def score(self) -> float:
        """The figure optimised: nDCG@100, the top-k regime the paper cares about."""
        return self.ndcg.get(100, 0.0)


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


def format_grid(results: Sequence[TuningResult], metric: str) -> str:
    """The (alpha, delta) nDCG@100 surface, as a text table.

    The analogue of the optimisation grid figure in Verdolotti et al. Printing the
    whole surface, not just the argmax, is what lets a reader see whether the optimum
    is a plateau or a spike -- and therefore how much the choice actually matters.
    """
    rows = [r for r in results if r.metric == metric]
    if not rows:
        return ""
    alphas = sorted({r.alpha for r in rows})
    deltas = sorted({r.delta for r in rows})

    out = [f"nDCG@100 surface for {metric}", ""]
    out.append("  delta \\ alpha  " + "".join(f"{a:>8.1f}" for a in alphas))
    best = max(rows, key=lambda r: r.score)
    for d in deltas:
        cells = []
        for a in alphas:
            hit = next((r for r in rows if r.alpha == a and r.delta == d), None)
            v = hit.score if hit else float("nan")
            mark = "*" if hit is best else " "
            cells.append(f"{v:>7.4f}{mark}")
        out.append(f"  {_fmt_delta(d):<14}" + "".join(cells))
    out.append("")
    out.append(f"  best: alpha={best.alpha:.1f}, delta={_fmt_delta(best.delta)}, "
               f"nDCG@100={best.score:.4f}  (* above)")
    return "\n".join(out)
