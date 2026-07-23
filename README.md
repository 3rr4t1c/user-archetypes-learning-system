# ArLeS — Archetype Learning System

Learn how users spread information, from a stream of reshares.

ArLeS assigns every user a 3-dimensional vector describing how much they behave like a
**super-spreader** (originates content that travels), an **amplifier** (extends the reach
of others' content), or a **coordinated account** (acts in concert with others). The
three are not mutually exclusive: a user gets a score on each axis, in `[0,1]`, plus a
confidence saying how much of that account we actually saw.

It is **content-agnostic**. ArLeS never looks at what was said, only at who reshared
whose content, and when. The archetypes come from [Verdolotti, Luceri & Giordano
(2025)](#citation), where they are instantiated over misinformation; drop the
credibility filter and the same axes measure spreading power over any content.

---

## What it needs from your data

**Reshares.** That is the whole requirement:

> who reshared it · what they reshared · whose it was · when

If your platform can export that, ArLeS works on it. All twelve features are computed
from reshares alone — a post nobody reshared cannot raise an h-index, and two people
posting are never acting on the same content, so those rows carry no archetype signal.

Everything else (posts, replies, quotes, follows, blocks) is accepted and ignored by the
features. Confidence is the one exception: it counts *total* activity, because "how much
of this account have we seen" is a different question from "what does this account do".
Supply only reshares and it degrades gracefully to the reshare count.

### The canonical schema

```
action_id         unique id of this action
actor_id          who performed it
activity_type     post | repost | reply | quote | follow | block | other
created_at        ISO-8601 with a UTC offset
parent_id         the content acted upon; empty for posts
parent_actor_id   who authored that content; empty for posts
```

**`parent_actor_id` is the one that matters.** The whole super-spreader axis rests on
attributing a reshare to the original author. Deriving it by looking the parent post up
in the stream fails on any sampled export — on the Bluesky sample the original post is
present for **0.09%** of reposts — which silently zeroes reshare yield, the h-index and
the TASH-index. So it is an explicit column, and `describe_coverage()` tells you up front
if your data cannot supply it, rather than returning zeros you might believe.

There is no `text` column. Content-agnostic by construction, not by convention.

### Bring your own platform

Write a mapper: one function turning your rows into `CanonicalAction`s. Platform quirks
live there and nowhere else — `arles/mappers/bluesky.py` is the only module in the
codebase that knows what an AT-URI is.

```bash
python -m arles.mappers.bluesky /path/to/raw_export canonical.csv --start 2024-10-12 --days 5
```

---

## Install

```bash
conda env create -f environment.yml
conda activate arles
pytest            # ~250 tests, a couple of seconds
```

Runtime needs **numpy** only. `scipy` (paired statistics), `tqdm` (progress bars) and
`matplotlib` (figures) are optional and used by the scripts, not by the library.

---

## Quick start

```python
from datetime import timedelta
from arles.features import WindowIndex, FeatureExtractor
from arles.embedding import fit_pooled

def window(actions, start, end):
    index = WindowIndex.build(actions)          # pass 1: reshares per post
    fx = FeatureExtractor(index, start, end)    # pass 2: the twelve features
    for a in actions:
        fx.add(a)
    return fx

w1 = window(actions_week1, t0, t0 + timedelta(days=5))
w2 = window(actions_week2, t0 + timedelta(days=5), t0 + timedelta(days=10))

ids1, X1 = w1.finish()                          # (n_users, 12)
ids2, X2 = w2.finish()

embedder = fit_pooled([X1, X2])                 # ONE fit for every window you compare
Z1 = embedder.transform(X1)                     # (n_users, 3) in [0,1]
_, conf1 = w1.confidence()                      # (n_users,) in [0,1]
```

```
features: 12 | axes: ('superspreader', 'amplifier', 'coordinated')
window 1: (23, 12) -> (23, 3) | confidence: (23,)
user author0 vector: [1. 0. 0.5] confidence: 0.529
```

**`fit_pooled` across every window you intend to compare** is not a stylistic choice.
See [Pitfalls](#pitfalls-learned-the-hard-way).

---

## The pipeline

```
                          ┌─ super-spreader ─┐
 actions ─ reshares ─┬────┤   amplifier      ├─ log1p ─ scale ─ PC1 ─ [0,1] ─┐
 (schema)  (features)│    └─ coordinated ────┘   └─── frozen pooled fit ───┘  ├─ vector ∈ [0,1]³
                     │        4 features each                                 │
                     └─ total activity ────── volume / recency / lifespan ────┴─ confidence ∈ [0,1]
```

**The twelve features**, four per axis, all from reshares:

| super-spreader | amplifier | coordinated |
|---|---|---|
| `influence_score` — reshares received | `repost_count` | `co_action_rate` — share of actions with a co-actor |
| `h_index` — h posts with ≥h reshares | `repost_rate` — per day | `co_action_size` — mean distinct co-actors |
| `tai_score` — EMA of per-slot influence | `ear_index` — early into large cascades | `co_action_latency` — tightness of synchrony |
| `tash_index` — EMA of per-slot h-index | `amplification_breadth` — distinct authors | `niche_co_action` — co-action on non-viral content |

Four correlated views per axis; PC1 extracts the shared factor.

**Confidence** — how much of the account we saw:
`0.5 · log(total actions) + 0.3 · recency + 0.2 · lifespan`, all measured against the
analysis window.

Two design notes worth knowing:

- **Two passes per window.** The EaR-Index needs `N_p`, a post's *final* cascade size,
  which no single streaming pass can know; `niche_co_action` needs it to tell a swarm
  apart from a viral pile-on. Pass 1 counts reshares per post; pass 2 computes features.
- **Pass 2 sorts.** Exports are usually ordered by ingestion, and `created_at` is
  client-supplied; on the Bluesky archive it deviates by up to **18.6 h**. Co-action
  windows, cascade ranks and EMA slots all assume chronological order.

---

## Scale

Bounded memory per window, and only the files that can contain the window are read.

```bash
# 146 GB archive, 13 files, 5-day window -> ~1 file touched, ~1 minute
python scripts/check_reshare_density.py /Volumes/Drive/archive --days 5 --start 2024-10-12
```

`arles.streaming` indexes each file from two rows, skips files that cannot overlap, and
binary-searches to the window's start.

Peak memory for a 5-day window at full density is **under 1 GB**, dominated by pass 1's
per-content counters. Nothing in ArLeS keeps a counter per *pair* of users: one day of
the Bluesky archive implies 19.7M co-reshare pairs (~2 GB) and a 5-day window ~100M
(~10 GB), growing quadratically with a post's popularity. `niche_co_action` captures the
coordination signal in O(users) instead.

---

## Scripts

| | |
|---|---|
| `scripts/make_figures.py` | Score every window on one frozen fit and produce the archetype-prevalence figure plus the tables behind it (`windows.csv`, `prevalence.csv`, `features.csv`). Diagnostics go to `figures/diagnostics/`. |
| `scripts/prevalence_test.py` | Two-proportion test of pre- vs post-event prevalence, per axis per event. Reads `prevalence.csv`; no archive pass. |
| `scripts/paired_test.py` | Incumbent paired Wilcoxon on per-user scores, for testing whether the same accounts changed. |
| `scripts/check_reshare_density.py` | Can your data support the super-spreader axis? Reports reshares per author and the h-index distribution, per window. |
| `scripts/tune_time_aware.py` | Re-fit (α, δ) for the TASH-Index and TAI-Score on *your* data, by the nDCG grid search of the original paper. |
| `scripts/analyse_tuning.py` | Re-read a saved tuning surface in seconds, without touching the archive. |
| `python -m arles.mappers.bluesky` | Raw Bluesky export → canonical CSV. |

### Reproducing the paper's figure and statistics

```bash
python scripts/make_figures.py /path/to/archive --out figures/
python scripts/prevalence_test.py figures/prevalence.csv --out figures/prevalence_test.csv
```

The first scores every window on one pooled fit, applies the confidence gate (0.3 by
default), and writes `fig_archetype_prevalence.pdf` with the tables behind it; the second
runs the significance test. Both are deterministic and the second needs no archive pass.

---

## Pitfalls (learned the hard way)

Each of these was a real bug that produced *plausible numbers*, which is why they are
called out rather than quietly fixed.

**Fit the embedding on pooled windows, never per window.** Per-window fitting makes each
window's own extremes its reference: the most extreme user scores 1.0 by construction.
Across seven windows of one event, the maximum reshares received per author ran
**6,842 → 275,198** — a 40× swing in the min–max denominator. Identical behaviour scored
1.0 in one window and 0.025 in another, and a figure plotting that over time shows a
trend which is purely the ruler changing.

**Check your attribution rate.** `describe_coverage()` reports the share of reshares with
a known `parent_actor_id`. Below ~50%, the super-spreader axis is computed from a small,
non-random subset and should not be trusted.

**Check your density before believing an h-index.** Sampling does not shrink an h-index,
it destroys it: at a 1% sample a post with 100 real reshares shows ~1, so "h posts with
≥h reshares each" cannot form. On the Bluesky sample, 98% of authors were pinned at h=1;
on the full archive, 24% reach h≥2. `check_reshare_density.py` tells you which world you
are in.

**Timestamps are client-supplied.** They can be wrong by hours, and they arrive in
several shapes. `parse_timestamp` raises rather than defaulting — its predecessor fell
back to `datetime.now()`, which replaced an entire multi-day window with the wall-clock
time of the run and silently zeroed every time-dependent feature.

**Reshare cascades are stars, not trees** (on AT-Protocol at least). A repost always
references the *root* post — verified across 1.5M rows, 508,895 reposts, zero exceptions.
So it is structurally impossible to observe that one user's reshare caused another's. The
amplifier axis measures resharing *intensity* and *earliness*, not reach extension.

---

## Layout

```
arles/
  actions.py     how to read one action: timestamps, identifiers
  schema.py      the canonical schema every mapper targets
  mappers/       platform adapters (bluesky.py; add your own)
  streaming.py   windowed, bounded-memory reading of a large archive
  features.py    the twelve features, two passes
  embedding.py   12 -> 3, with a frozen pooled fit
  prevalence.py  head statistics for a rare class: prevalence, tails, concentration
  tuning.py      re-fitting (alpha, delta) by nDCG grid search
scripts/         figures, statistics, diagnostics and fitting, all runnable standalone
tests/           ~250 tests; most pin a bug that actually shipped
```

---

## Citation

The archetypes and the TASH-Index / TAI-Score come from:

> Verdolotti, Luceri & Giordano (2025). *Predicting Misinformation Super-Spreaders: a
> Time-Aware Social H-Index Approach.*

ArLeS drops the misinformation restriction, so the same axes measure spreading power over
any content. α and δ should be **re-fitted on your own data**
(`scripts/tune_time_aware.py`) rather than inherited: the published δ = 14 days does not
fit inside a 5-day analysis window, where the EMA would hold a single term and the
time-aware component would silently do nothing.
