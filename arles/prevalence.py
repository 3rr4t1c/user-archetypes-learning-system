"""Measuring rare classes of different rarity: head statistics for the archetype axes.

Why a mean over everyone is the wrong statistic
-----------------------------------------------
None of the three archetypes describes a typical account, so a population mean has
almost no room to move. Measured on the full archive, `median_superspreader` is 0.0000
in all fourteen E1/E2 windows and the mean sits between 0.021 and 0.037: more than half
of every window scores exactly zero, so the mean is mostly a statement about how many
uninvolved accounts are in the denominator. If 10 super-spreaders before an event become
15 after it, that is +50% of the phenomenon and about +1e-5 of the mean -- against a base
that already moves more than that when the population triples, which it did twice.

A flat line in a mean-over-everyone figure is therefore not evidence of a flat
phenomenon. It is what that statistic does to a rare class.

The three archetypes are rare at three different scales
-------------------------------------------------------
This is the constraint the module is built around, and it is not the same claim as
"archetypes are rare".

    super-spreader   orders of magnitude below the user base: a handful of accounts.
                     If many accounts were super-spreaders, none would be.
    amplifier        far more numerous -- resharing is ordinary behaviour, and being
                     good at it is not exceptional the way being a hub is.
    coordinated      in between. Coordination needs a decent-sized synchronised group,
                     so it cannot be a handful; but it stays a small set of accounts
                     pushing particular content.

Measured on E1's and E2's pre-event windows at a common bar of 0.5, the ordering holds:
261, 1522 and 299 accounts per 100k respectively. At 0.7 it is 27, 114 and 12 -- the
coordinated axis thins out fastest, super-spreaders persist, amplifiers stay an order
above.

Never compare axes at a per-axis quantile
-----------------------------------------
The trap this module walked into first. Take the top 0.1% of *each axis* separately and
every axis has 0.1% of the population above its bar -- by construction, whatever the data
says. On the real pre-event window that gave 194 super-spreaders, 198 amplifiers and 304
coordinated accounts: three numbers that look like a finding and are arithmetic. The
statistic destroys exactly the scale difference above.

So the figures use one common bar `theta` applied to all three axes. The axes are all
min-max'd onto [0,1] over the same pooled fit, so a single theta is one question asked
three times, and the three answers are free to differ -- which is the point.

The honest caveat: each axis's [0,1] is its own pooled PC1 range, set by that axis's
most extreme account, so theta is not a *physically* identical bar across axes. What it
supports is the comparison of shape and of change; for an absolute statement about one
archetype, read `anchor_median`, which is in reshares.

    prevalence     how many accounts clear a fixed bar, and what share of users that is
    intensity      how extreme the extremes are (upper quantiles) -- no denominator
    concentration  how unequally the axis is distributed (Gini, top-1% mass share)

Compare each archetype to its own baseline, never to another's
--------------------------------------------------------------
Because the scales differ by orders of magnitude, the raw numbers are not comparable
across archetypes and never will be. 15 super-spreaders and 15,000 amplifiers can be the
same 1.5x. `fold_change` against the archetype's own pre-event window is the quantity a
shock is read in; the absolute rate is reported beside it so that a 1.5x on 10 accounts
is not mistaken for a 1.5x on 10,000.

Count and rate are both reported, deliberately
----------------------------------------------
They answer different questions and can disagree. The population roughly tripled at E1
(206,158 -> 973,996 users in one window), so a head count that triples means the rate
did not move: the platform got bigger and brought proportional numbers of everything. A
head count that stays flat while the population triples means the rate fell by 3x. Only
showing one of the two lets a reader draw either conclusion.

Anchoring the bar
-----------------
A threshold on an archetype axis is a PC1 projection through four log-scaled features
and a min-max, so "score >= 0.5" is not a quantity anyone has intuition about. For each
axis one raw feature is designated an anchor, and its median across the accounts above
the bar is reported: "the accounts above the super-spreader bar received a median of N
reshares in the window" is a sentence a reader can check.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .features import FEATURE_NAMES

#: The bar the figures use: one score, applied to all three axes.
#:
#: 0.5 is "the upper half of the axis's pooled range" and is a headline, not a discovery.
#: What makes it usable is that it is *common* to the three axes, so their counts are
#: free to differ and the scale difference survives into the figure. See the module
#: docstring: at 0.5 the pre-event windows give 261 / 1522 / 299 accounts per 100k.
COMMON_BAR = 0.5

#: The bars the figures sweep. Any single theta invites "why that one?", and the answer
#: worth having is that it does not matter: a change visible at 0.3 and at 0.7 belongs to
#: the data; one that inverts between them belongs to the bar and is not a finding.
SWEEP_BARS = (0.3, 0.4, 0.5, 0.6, 0.7)

#: A per-axis quantile. Kept because it is the right tool for one axis over time, and
#: because `sweep_quantiles` uses it -- but NOT for comparing archetypes to each other.
#:
#: The top q of each axis puts (1-q) of the population above every axis's bar by
#: construction. On the real pre-event window, q=0.999 gives 194 super-spreaders, 198
#: amplifiers and 304 coordinated accounts -- three numbers that look like a measurement
#: and are arithmetic. The three archetypes are rare at three different scales, and this
#: statistic cannot see that.
DEFAULT_QUANTILE = 0.999

#: The raw feature quoted alongside each axis, to make the bar interpretable.
#: Each is the most legible member of its bucket: a count of something that happened.
AXIS_ANCHOR: Dict[str, str] = {
    "superspreader": "influence_score",      # reshares received
    "amplifier": "repost_count",             # reshares made
    "coordinated": "co_action_size",         # accounts co-acting on the same content
}


def pooled_threshold(columns: Sequence[np.ndarray], q: float = DEFAULT_QUANTILE) -> float:
    """The score at the q-th quantile of every window pooled, for ONE axis.

    Fitted once on all the windows that will be compared, then frozen -- the same
    argument as the pooled embedding. A per-window quantile puts a fixed fraction above
    the bar by construction and can only ever draw a flat line.

    Do not use the result to compare one axis against another: applied per axis, a
    quantile equalises the three archetypes' prevalence by construction. Use COMMON_BAR.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0,1), got {q}")
    parts = [np.asarray(c, dtype=np.float64).ravel() for c in columns]
    parts = [p for p in parts if p.size]
    if not parts:
        return float("nan")
    return float(np.quantile(np.concatenate(parts), q))


def gini(values: np.ndarray) -> float:
    """Gini coefficient: 0 = every account identical, 1 = one account holds everything.

    Reported because it needs no threshold at all. Prevalence and intensity both depend
    on where the bar is put; concentration does not, so it is the one statement in this
    module that a reviewer cannot argue with by arguing about 0.999.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    v = np.clip(v, 0.0, None)
    total = v.sum()
    if total <= 0:
        return 0.0  # everyone at zero is perfect equality, not a divide-by-zero
    v = np.sort(v)
    n = v.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * v).sum()) / (n * total) - (n + 1.0) / n)


def top_share(values: np.ndarray, fraction: float = 0.01) -> float:
    """Share of the axis total held by the top `fraction` of accounts."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0,1], got {fraction}")
    v = np.asarray(values, dtype=np.float64).ravel()
    v = np.clip(v[np.isfinite(v)], 0.0, None)
    if v.size == 0:
        return float("nan")
    total = v.sum()
    if total <= 0:
        return 0.0
    k = max(1, int(round(v.size * fraction)))
    return float(np.sort(v)[-k:].sum() / total)


@dataclass
class WindowHead:
    """Everything about one axis in one window that a mean cannot say."""

    axis: str
    threshold: float
    n_users: int
    count: int              # accounts above the bar
    rate_per_100k: float    # ...as a share of the window's population
    p99: float
    p999: float
    pmax: float
    gini: float
    top1pct_share: float
    anchor_feature: str
    anchor_median: float    # median raw anchor value among the accounts above the bar

    def as_row(self) -> dict:
        return asdict(self)


def head_stats(
    scores: np.ndarray,
    threshold: float,
    axis: str,
    X: Optional[np.ndarray] = None,
) -> WindowHead:
    """Head statistics for one axis of one window, against a frozen pooled bar.

    `scores` is the window's column for this axis; `X` its (n, 12) raw feature matrix,
    used only to quote the anchor feature.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = int(s.size)
    if n == 0:
        nan = float("nan")
        return WindowHead(
            axis=axis, threshold=float(threshold), n_users=0, count=0,
            rate_per_100k=nan, p99=nan, p999=nan, pmax=nan, gini=nan,
            top1pct_share=nan, anchor_feature=AXIS_ANCHOR.get(axis, ""),
            anchor_median=nan,
        )

    above = s >= threshold
    count = int(above.sum())

    anchor = AXIS_ANCHOR.get(axis, "")
    anchor_median = float("nan")
    if X is not None and anchor in FEATURE_NAMES and count:
        col = np.asarray(X, dtype=np.float64)[:, FEATURE_NAMES.index(anchor)]
        anchor_median = float(np.median(col[above]))

    return WindowHead(
        axis=axis,
        threshold=float(threshold),
        n_users=n,
        count=count,
        rate_per_100k=float(count) / n * 1e5,
        p99=float(np.quantile(s, 0.99)),
        p999=float(np.quantile(s, 0.999)),
        pmax=float(s.max()),
        gini=gini(s),
        top1pct_share=top_share(s, 0.01),
        anchor_feature=anchor,
        anchor_median=anchor_median,
    )


def sweep_quantiles(
    columns: Sequence[np.ndarray],
    axis: str,
    quantiles: Sequence[float] = (0.99, 0.999, 0.9999),
) -> Dict[float, List[float]]:
    """Prevalence per window at several bars, to show the story is not the bar's.

    Any threshold invites "why that one?". The answer worth having is that it does not
    matter: if the shape of the prevalence curve is the same at the top 1%, the top 0.1%
    and the top 0.01%, the finding is a property of the data. If it inverts, it is a
    property of the threshold and must not be reported as a finding.
    """
    out = {}
    for q in quantiles:
        bar = pooled_threshold(columns, q)
        out[q] = [
            float((np.asarray(c, dtype=np.float64).ravel() >= bar).mean() * 1e5)
            if np.asarray(c).size else float("nan")
            for c in columns
        ]
    return out


def fold_change(pre: float, post: float) -> float:
    """post/pre, the quantity a rare class is actually read in.

    "10 super-spreaders before, 15 after" is +50%; on a mean over 600k accounts it is a
    flat line. Undefined rather than infinite when nothing was there to begin with:
    0 -> 15 is not a fold change, it is a different sentence.
    """
    if not np.isfinite(pre) or not np.isfinite(post) or pre <= 0:
        return float("nan")
    return float(post / pre)
