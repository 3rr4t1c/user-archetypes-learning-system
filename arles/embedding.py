"""Turning twelve features into the 3-dimensional archetype vector.

Each bucket of four features is reduced to one scalar by its first principal component
and rescaled to [0,1]; concatenating the three gives the archetypal vector. PC1 is the
right tool precisely because the four features in a bucket are meant to be correlated:
they are four views of one latent quantity (diffusion power, amplification intensity,
co-action), and PC1 is that shared factor.

Fit once, transform many
------------------------
The scaler, the components and the [0,1] bounds are fitted on the *pooled* windows and
then frozen. This is the whole point of the module, and it is not a detail.

Fitting per window makes the extremes of each window its own reference point: the most
extreme user scores 1.0 by construction, whatever their behaviour. Measured across the
seven windows of E1 on the full archive, the maximum reshares received per author runs
6,842 -> 275,198 -- a 40x swing in the min-max denominator. A user with 6,842 reshares
scores 1.0 in the pre-event window and 0.025 six windows later, for identical
behaviour. The population moves too: 104,503 -> 364,679 authors, and both pre-event
windows have ~3.2x fewer authors than every window after them, because the Brazil ban
and the ToS change each tripled the user base overnight.

Any figure that plots these scores over time is therefore comparing quantities measured
against different rulers, and a rising line can be an artefact of the ruler shrinking.
Pooling gives every window one ruler.

Log scaling
-----------
The count features are extremely heavy-tailed: influence_score spans 1..275,198 with a
median of 3. Standardising raw counts makes the variance -- and therefore PC1 -- a
description of the top handful of accounts, and after min-max everyone else is crushed
towards zero. Every feature is non-negative, so log1p is applied first. It is monotone,
so it changes no ranking; it makes PC1 describe the typical account rather than the
extreme one.

Sign
----
A principal component is only defined up to sign. numpy's SVD is deterministic for a
given matrix but nothing ties the sign across *different* matrices, so a component can
invert between fits and min-max will silently flip the score's direction. The sign is
pinned so the loadings sum positive: higher features always mean a higher score.
"""

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .features import (
    AMPLIFIER_FEATURES,
    COORDINATED_FEATURES,
    FEATURE_NAMES,
    SUPERSPREADER_FEATURES,
)

#: The three axes, and which feature columns feed each.
BUCKETS: Dict[str, Tuple[str, ...]] = {
    "superspreader": SUPERSPREADER_FEATURES,
    "amplifier": AMPLIFIER_FEATURES,
    "coordinated": COORDINATED_FEATURES,
}
AXES = tuple(BUCKETS)

_EPS = 1e-12


@dataclass
class BucketFit:
    """The frozen transform for one archetype axis."""

    features: Tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    loadings: np.ndarray
    lo: float
    hi: float

    def project(self, block: np.ndarray) -> np.ndarray:
        z = (block - self.mean) / self.scale
        raw = z @ self.loadings
        span = self.hi - self.lo
        if span < _EPS:
            return np.full(raw.shape[0], 0.5)
        return np.clip((raw - self.lo) / span, 0.0, 1.0)


class ArchetypeEmbedder:
    """Fit on pooled windows, then transform each window with the frozen fit.

    Usage:
        embedder = ArchetypeEmbedder().fit(np.vstack([X_w1, X_w2, ...]))
        Z_w1 = embedder.transform(X_w1)   # (n_users, 3) in [0,1]

    Fitting and transforming the same single window (fit_transform) reproduces the
    original per-window behaviour and is offered only for comparison; it is not
    suitable for anything that compares windows.
    """

    def __init__(self, log_scale: bool = True):
        self.log_scale = log_scale
        self.buckets: Dict[str, BucketFit] = {}

    # ------------------------------------------------------------------ internals

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected (n, {len(FEATURE_NAMES)}) matching FEATURE_NAMES, got {X.shape}"
            )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.log_scale:
            # Every feature is a non-negative count or rate; log1p tames tails spanning
            # five orders of magnitude without reordering anyone.
            X = np.log1p(np.maximum(X, 0.0))
        return X

    @staticmethod
    def _pc1(z: np.ndarray) -> np.ndarray:
        """First principal component of an already-standardised block."""
        if z.shape[0] < 2:
            # Nothing to decompose; weight the features equally.
            return np.ones(z.shape[1]) / np.sqrt(z.shape[1])
        # Economy SVD of the centred block: rows of Vt are the components.
        _, _, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
        w = vt[0]
        # A component is defined only up to sign, and nothing ties the sign across
        # separate fits. Pin it so that higher features always mean a higher score.
        if w.sum() < 0:
            w = -w
        return w

    # ------------------------------------------------------------------ api

    def fit(self, X: np.ndarray) -> "ArchetypeEmbedder":
        """Fit on a pooled feature matrix -- every window you intend to compare."""
        Xp = self._prepare(X)
        self.buckets = {}
        for axis, feats in BUCKETS.items():
            cols = [FEATURE_NAMES.index(f) for f in feats]
            block = Xp[:, cols]

            mean = block.mean(axis=0)
            scale = block.std(axis=0)
            # A constant column carries no information; leaving scale at 0 would make
            # it inf. Setting it to 1 leaves the column at zero after centring.
            scale = np.where(scale < _EPS, 1.0, scale)

            z = (block - mean) / scale
            w = self._pc1(z)
            raw = z @ w
            self.buckets[axis] = BucketFit(
                features=tuple(feats),
                mean=mean,
                scale=scale,
                loadings=w,
                lo=float(raw.min()),
                hi=float(raw.max()),
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """(n_users, 3) in [0,1], on the frozen pooled scale."""
        if not self.buckets:
            raise RuntimeError("fit() first: the point of this class is a frozen fit")
        Xp = self._prepare(X)
        out = np.zeros((Xp.shape[0], len(AXES)), dtype=np.float64)
        for i, axis in enumerate(AXES):
            fit = self.buckets[axis]
            cols = [FEATURE_NAMES.index(f) for f in fit.features]
            out[:, i] = fit.project(Xp[:, cols])
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform the same matrix.

        Reproduces the original per-window behaviour. Use it on one window only if you
        never compare that window to another: the [0,1] bounds become that window's own
        extremes, which is exactly what makes cross-window comparison meaningless.
        """
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------ reporting

    def loadings_table(self) -> str:
        """What each axis is actually made of -- for the paper, and for sanity."""
        lines = []
        for axis in AXES:
            fit = self.buckets[axis]
            lines.append(f"{axis}:")
            for name, w in zip(fit.features, fit.loadings):
                lines.append(f"    {name:<24} {w:+.3f}")
            lines.append(f"    (raw PC1 range on the pooled fit: "
                         f"{fit.lo:.3f} .. {fit.hi:.3f})")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "log_scale": self.log_scale,
                "buckets": {
                    axis: {
                        "features": list(f.features),
                        "mean": f.mean.tolist(),
                        "scale": f.scale.tolist(),
                        "loadings": f.loadings.tolist(),
                        "lo": f.lo,
                        "hi": f.hi,
                    }
                    for axis, f in self.buckets.items()
                },
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "ArchetypeEmbedder":
        """Reload a frozen fit, so a figure can be regenerated on the same scale."""
        blob = json.loads(text)
        self = cls(log_scale=blob["log_scale"])
        self.buckets = {
            axis: BucketFit(
                features=tuple(b["features"]),
                mean=np.asarray(b["mean"]),
                scale=np.asarray(b["scale"]),
                loadings=np.asarray(b["loadings"]),
                lo=b["lo"],
                hi=b["hi"],
            )
            for axis, b in blob["buckets"].items()
        }
        return self


def fit_pooled(matrices: Iterable[np.ndarray], log_scale: bool = True) -> ArchetypeEmbedder:
    """Fit one embedder across several windows' feature matrices.

    This is the call that makes a temporal figure legitimate: every window is then
    scored against the same reference, so a change in the line is a change in
    behaviour rather than a change in the ruler.
    """
    stacked = np.vstack([np.asarray(m, dtype=np.float64) for m in matrices])
    return ArchetypeEmbedder(log_scale=log_scale).fit(stacked)
