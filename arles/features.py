"""The twelve archetype features, computed over one window in two passes.

What ArLeS needs from your data
-------------------------------
Only reposts. Every one of the twelve features is computed from repost actions alone:

    influence_score, h_index, tai_score, tash_index   (super-spreader)
    repost_count, repost_rate, ear_index, amplification_breadth   (amplifier)
    co_action_rate, co_action_size, co_action_latency, niche_co_action   (coordinated)

Posts, replies, quotes, follows and blocks are accepted by the schema and ignored
here. A post nobody reshared contributes nothing to an h-index, and two people posting
are never acting on the same content -- so those rows carry no archetype signal. The
practical consequence is that ArLeS runs on any platform that can export reshares as
(who reshared, what, whose it was, when), which is close to the minimum any reshare-
bearing platform can offer.

Confidence is the exception, and deliberately so. It measures data availability --
how much of an account we have seen -- and so counts *total* actions, including the
replies and follows no feature reads. Two users with five reposts each are not equally
well observed if one also made 500 replies: we watched that account act 500 times and
it chose not to reshare, which makes its low amplifier score a measurement rather than
missing data. Supplying only reposts is still fine; volume then degrades to the repost
count, which describe_coverage reports.

Why two passes
--------------
The EaR-Index needs N_p, a post's *final* reshare count, which is unknowable while
streaming; `niche_co_action` needs the same to tell a swarm apart from a viral pile-on.
Pass 1 counts reshares per content; pass 2 computes the features. This also lets the
h-index be exact rather than approximated.

Why pass 2 sorts
----------------
The archive is ordered by ingestion, not by created_at, which is client-supplied and
deviates by up to ~18.6 h (see arles.streaming). Co-action windows, reshare ranks and
EMA time slots all assume chronological order, so pass 2 buffers the window's reposts
and sorts them. Buffering is what makes this affordable: three int64 columns, not rows.

What the amplifier axis can and cannot mean
-------------------------------------------
An AT-Protocol repost always references the *root post*, never another repost -- there
is no repost-of-a-repost record. Verified across 1.5M rows: 508,895 reposts, every
parent an app.bsky.feed.post, zero exceptions. So a reshare cascade is a depth-1 star,
not a tree, and it is structurally impossible to observe that one user's reshare caused
another's. The amplifier axis therefore measures resharing *intensity* and *earliness*,
not reach extension. The EaR-Index is the honest proxy: it rewards resharing early into
cascades that eventually grew large.

Memory
------
Bounded, per window, dominated by pass 1's per-content counters:

    content index + counts   ~ O(distinct reshared posts)   ~940k/5-day window, ~200 MB
    actor index + counters   ~ O(actors)                    ~600k, ~70 MB
    pass 2 buffer            ~ 3 x int64 x reposts          ~3M, ~72 MB
    co-action window         ~ O(reposts/sec x delta_t)     ~2k entries, negligible

Metrics needing a counter per *pair* of users -- co-reshare similarity, clustering
coefficient, reciprocal interaction -- are deliberately absent. One day of the archive
implies 19.7M co-reshare pairs (~2 GB); a 5-day window ~100M (~10 GB), and it grows
quadratically with a post's popularity. `niche_co_action` captures the same signal in
O(users).
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .schema import CanonicalAction

#: Feature names per archetype, in the order they appear in the feature matrix.
SUPERSPREADER_FEATURES = ("influence_score", "h_index", "tai_score", "tash_index")
AMPLIFIER_FEATURES = ("repost_count", "repost_rate", "ear_index", "amplification_breadth")
COORDINATED_FEATURES = (
    "co_action_rate",
    "co_action_size",
    "co_action_latency",
    "niche_co_action",
)
FEATURE_NAMES = SUPERSPREADER_FEATURES + AMPLIFIER_FEATURES + COORDINATED_FEATURES

#: EMA smoothing for the time-aware features. Verdolotti et al. grid-search alpha and
#: delta against an nDCG objective on a misinformation-strength target and obtain
#: alpha=0.5, delta=14 days for TASH and alpha=0.6, delta=18 days for TAI. No such
#: target exists here, so no equivalent tuning is possible; alpha is kept at their
#: optimum. delta cannot be: a 14-day slot does not fit in a 5-day analysis window, so
#: the EMA would have a single term and the "time-aware" part would do nothing.
DEFAULT_TASH_ALPHA = 0.5
DEFAULT_TAI_ALPHA = 0.6

#: Slot length for the time-aware features. Must be much shorter than the analysis
#: window for the moving average to accumulate terms; 6 h gives 20 slots in 5 days.
DEFAULT_SLOT = timedelta(hours=6)

#: Two users acting on the same content within this gap are treated as co-acting.
#: Follows the co-reshare convention of Pacheco et al. and Luceri et al.
DEFAULT_CO_ACTION_WINDOW = timedelta(seconds=300)

#: Content with more resharers than this is treated as viral rather than coordinated.
#: Reposting a popular post puts you alongside hundreds of strangers; that is
#: popularity, not coordination. niche_co_action counts co-actors only on content that
#: stayed below this.
DEFAULT_NICHE_THRESHOLD = 50


@dataclass
class WindowIndex:
    """Pass 1: how many times each piece of content was reshared, and by whom authored.

    Memory is O(distinct reshared content). On a 5-day window of the full Bluesky
    archive that is ~940k entries.
    """

    content_reshares: Dict[str, int] = field(default_factory=dict)
    content_author: Dict[str, str] = field(default_factory=dict)
    n_reposts: int = 0
    n_unattributed: int = 0

    def add(self, action: CanonicalAction) -> None:
        if action.activity_type != "repost" or not action.parent_id:
            return
        if action.is_self_reshare:
            # Self-reshares are not diffusion, per Verdolotti et al.
            return
        if not action.parent_actor_id:
            self.n_unattributed += 1
            return
        self.n_reposts += 1
        pid = action.parent_id
        self.content_reshares[pid] = self.content_reshares.get(pid, 0) + 1
        self.content_author[pid] = action.parent_actor_id

    @classmethod
    def build(cls, actions: Iterable[CanonicalAction]) -> "WindowIndex":
        index = cls()
        for action in actions:
            index.add(action)
        return index


def h_index(counts: Sequence[int]) -> int:
    """Largest h such that h items have >= h reshares each."""
    ordered = sorted(counts, reverse=True)
    h = 0
    for i, c in enumerate(ordered, start=1):
        if c >= i:
            h = i
        else:
            break
    return h


def time_aware_scores(
    ts: np.ndarray,
    content: np.ndarray,
    content_author: np.ndarray,
    n_actors: int,
    origin: float,
    slot_seconds: float,
    tash_alpha: float,
    tai_alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """EMAs of per-slot influence score (TAI) and per-slot social h-index (TASH).

    Split out of FeatureExtractor so that arles.tuning can re-evaluate it across an
    (alpha, delta) grid without re-reading the archive: the sorted buffer is built
    once and this is called per grid point.

    Every author seen so far is decayed at every slot boundary, including slots in
    which they were reshared zero times -- TASH_t = alpha*TASH_{t-1} + (1-alpha)*H_t
    with H_t = 0. The original implementation only updated authors active in the slot,
    which let a single early burst persist undecayed for the rest of the window.

    Args:
        ts: repost timestamps, ascending.
        content: content slot per repost.
        content_author: author slot per content slot.
        origin: POSIX timestamp the slot grid starts from.
    """
    tai = np.zeros(n_actors, dtype=np.float64)
    tash = np.zeros(n_actors, dtype=np.float64)
    if ts.shape[0] == 0:
        return tai, tash

    seeded = np.zeros(n_actors, dtype=bool)
    seen_author = np.zeros(n_actors, dtype=bool)
    slot_of = np.floor((ts - origin) / slot_seconds).astype(np.int64)

    def flush(slot_counts: Dict[int, Dict[int, int]]) -> None:
        slot_influence = np.zeros(n_actors, dtype=np.float64)
        slot_h = np.zeros(n_actors, dtype=np.float64)
        for a_slot, per_content in slot_counts.items():
            vals = list(per_content.values())
            slot_influence[a_slot] = float(sum(vals))
            slot_h[a_slot] = float(h_index(vals))
            seen_author[a_slot] = True

        active = seen_author
        fresh = active & ~seeded
        tai[fresh] = slot_influence[fresh]
        tash[fresh] = slot_h[fresh]
        seeded[fresh] = True

        old = active & ~fresh
        tai[old] = tai_alpha * tai[old] + (1 - tai_alpha) * slot_influence[old]
        tash[old] = tash_alpha * tash[old] + (1 - tash_alpha) * slot_h[old]

    current = slot_of[0]
    buffer: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(ts.shape[0]):
        s = slot_of[i]
        if s != current:
            flush(buffer)
            buffer = defaultdict(lambda: defaultdict(int))
            current = s
        a_slot = int(content_author[content[i]])
        buffer[a_slot][int(content[i])] += 1
    flush(buffer)

    return tai, tash


class FeatureExtractor:
    """Pass 2: build the per-user feature matrix for one window.

    Usage:
        index = WindowIndex.build(pass1_actions)
        fx = FeatureExtractor(index, window_start, window_end)
        for action in pass2_actions:
            fx.add(action)
        user_ids, X = fx.finish()
    """

    def __init__(
        self,
        index: WindowIndex,
        window_start: datetime,
        window_end: datetime,
        slot: timedelta = DEFAULT_SLOT,
        tash_alpha: float = DEFAULT_TASH_ALPHA,
        tai_alpha: float = DEFAULT_TAI_ALPHA,
        co_action_window: timedelta = DEFAULT_CO_ACTION_WINDOW,
        niche_threshold: int = DEFAULT_NICHE_THRESHOLD,
    ):
        self.index = index
        self.window_start = window_start
        self.window_end = window_end
        self.slot_seconds = slot.total_seconds()
        self.tash_alpha = tash_alpha
        self.tai_alpha = tai_alpha
        self.co_action_seconds = co_action_window.total_seconds()
        self.niche_threshold = niche_threshold

        # Interning: strings -> dense indices, so the pass-2 buffer is three int64
        # columns rather than three million Python tuples.
        self._actor_ids: List[str] = []
        self._actor_index: Dict[str, int] = {}
        self._content_index: Dict[str, int] = {}
        self._content_author_idx: List[int] = []
        self._content_n: List[int] = []

        # The pass-2 buffer, sorted by time in finish().
        self._ts: List[float] = []
        self._content: List[int] = []
        self._actor: List[int] = []

        # Activity ledger for the confidence score. Keyed by actor id rather than by
        # interned slot, deliberately: an account that only replies must contribute to
        # its own confidence without being added to the feature matrix, where it would
        # sit as an all-zero row it never earned.
        #
        # The features need only reposts; confidence additionally uses total activity,
        # which is what "how much of this account have we seen" means. It is close to
        # free, since the pipeline already streams every row and filters to reposts.
        self._activity_count: Dict[str, int] = defaultdict(int)
        self._activity_first: Dict[str, float] = {}
        self._activity_last: Dict[str, float] = {}

    # ------------------------------------------------------------------ interning

    def _actor_slot(self, actor_id: str) -> int:
        slot = self._actor_index.get(actor_id)
        if slot is None:
            slot = len(self._actor_ids)
            self._actor_index[actor_id] = slot
            self._actor_ids.append(actor_id)
        return slot

    def _content_slot(self, content_id: str, author_id: str) -> int:
        slot = self._content_index.get(content_id)
        if slot is None:
            slot = len(self._content_n)
            self._content_index[content_id] = slot
            self._content_n.append(self.index.content_reshares.get(content_id, 0))
            self._content_author_idx.append(self._actor_slot(author_id))
        return slot

    # ------------------------------------------------------------------ ingestion

    def add(self, action: CanonicalAction) -> None:
        """Record one action. Only reposts feed the features; all of them feed confidence."""
        ts = action.created_at.timestamp()
        actor = action.actor_id
        self._activity_count[actor] += 1
        if actor not in self._activity_first or ts < self._activity_first[actor]:
            self._activity_first[actor] = ts
        if actor not in self._activity_last or ts > self._activity_last[actor]:
            self._activity_last[actor] = ts

        if action.activity_type != "repost" or not action.parent_id:
            return
        if action.is_self_reshare or not action.parent_actor_id:
            return
        if action.parent_id not in self.index.content_reshares:
            return  # not counted in pass 1; keep the two passes consistent

        c = self._content_slot(action.parent_id, action.parent_actor_id)
        a = self._actor_slot(action.actor_id)
        self._ts.append(action.created_at.timestamp())
        self._content.append(c)
        self._actor.append(a)

    # ------------------------------------------------------------------ finishing

    def sorted_buffer(self):
        """(ts, content, actor, content_author, content_n), chronologically sorted.

        The archive is ingestion-ordered and created_at is client-supplied, so rows
        arrive out of chronological order by up to ~18.6 h. Co-action windows, reshare
        ranks and EMA slots all assume time order, so sorting happens once here and
        everything downstream consumes the result. arles.tuning reuses this buffer to
        sweep an (alpha, delta) grid without touching the disk again.
        """
        ts = np.asarray(self._ts, dtype=np.float64)
        content = np.asarray(self._content, dtype=np.int64)
        actor = np.asarray(self._actor, dtype=np.int64)
        order = np.argsort(ts, kind="stable")
        return (
            ts[order],
            content[order],
            actor[order],
            np.asarray(self._content_author_idx, dtype=np.int64),
            np.asarray(self._content_n, dtype=np.int64),
        )

    def confidence(self) -> Tuple[List[str], np.ndarray]:
        """(user_ids, confidence) in [0,1], aligned with finish()'s rows.

        How much of this account have we actually seen? That is data availability, and
        it is measured over the user's whole activity, not only the reposts the features
        happen to read. Two users with five reposts each are not equally well observed
        if one of them also made 500 replies: we have watched that account act 500 times
        and it chose not to reshare, so its low amplifier score is a measurement. The
        other may simply be new.

        Authors who are reshared but never act themselves have no activity of their own;
        the reshares they received are what we have seen of them, so those count.
        """
        n_actors = len(self._actor_ids)
        if n_actors == 0:
            return [], np.zeros(0, dtype=np.float64)

        window_start_ts = self.window_start.timestamp()
        counts = np.zeros(n_actors, dtype=np.float64)
        first = np.full(n_actors, np.inf)
        last = np.full(n_actors, -np.inf)

        for i, uid in enumerate(self._actor_ids):
            counts[i] = self._activity_count.get(uid, 0)
            if uid in self._activity_first:
                first[i] = self._activity_first[uid]
                last[i] = self._activity_last[uid]

        # A user in the matrix who never acted is there because they were reshared.
        # Those reshares are what we observed of them, so they are their activity.
        if self._ts:
            ts, content, actor, content_author, content_n = self.sorted_buffer()
            received = np.zeros(n_actors, dtype=np.float64)
            np.add.at(received, content_author, content_n)
            passive = counts == 0
            counts[passive] = received[passive]

            r_first = np.full(n_actors, np.inf)
            r_last = np.full(n_actors, -np.inf)
            np.minimum.at(r_first, content_author[content], ts)
            np.maximum.at(r_last, content_author[content], ts)
            first[passive] = r_first[passive]
            last[passive] = r_last[passive]

        unseen = ~np.isfinite(first)
        first[unseen] = window_start_ts
        last[unseen] = window_start_ts

        conf = confidence_scores(
            counts, first, last, self.window_start, self.window_end
        )
        return list(self._actor_ids), conf

    def save_buffer(self, path: str) -> None:
        """Persist the pass-2 buffer so a window never has to be re-read.

        Reading a window off the external archive costs minutes; re-scoring an
        in-memory buffer costs seconds. The first parameter sweep spent 4.5 hours, and
        almost all of it was I/O that produced the same six buffers over and over. With
        these cached, changing the grid is a minutes-long job, which is the difference
        between "we can check that" and "we'd rather not".
        """
        np.savez_compressed(
            path,
            ts=np.asarray(self._ts, dtype=np.float64),
            content=np.asarray(self._content, dtype=np.int64),
            actor=np.asarray(self._actor, dtype=np.int64),
            content_author=np.asarray(self._content_author_idx, dtype=np.int64),
            content_n=np.asarray(self._content_n, dtype=np.int64),
            actor_ids=np.asarray(self._actor_ids, dtype="U40"),
            window=np.asarray(
                [self.window_start.timestamp(), self.window_end.timestamp()],
                dtype=np.float64,
            ),
        )

    @classmethod
    def load_buffer(cls, path: str, index: "WindowIndex", **kwargs) -> "FeatureExtractor":
        """Rebuild an extractor from a cached buffer, skipping both archive passes."""
        from datetime import timezone

        blob = np.load(path, allow_pickle=False)
        start_ts, end_ts = blob["window"]
        fx = cls(
            index,
            datetime.fromtimestamp(float(start_ts), tz=timezone.utc),
            datetime.fromtimestamp(float(end_ts), tz=timezone.utc),
            **kwargs,
        )
        fx._ts = list(blob["ts"])
        fx._content = list(blob["content"])
        fx._actor = list(blob["actor"])
        fx._content_author_idx = list(blob["content_author"])
        fx._content_n = list(blob["content_n"])
        fx._actor_ids = [str(a) for a in blob["actor_ids"]]
        fx._actor_index = {a: i for i, a in enumerate(fx._actor_ids)}
        return fx

    def finish(self) -> Tuple[List[str], np.ndarray]:
        """Return (user_ids, X) with X of shape (n_users, 12)."""
        n_actors = len(self._actor_ids)
        if not self._ts or n_actors == 0:
            return [], np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)

        ts, content, actor, content_author, content_n = self.sorted_buffer()

        f = {name: np.zeros(n_actors, dtype=np.float64) for name in FEATURE_NAMES}

        # ---------------------------------------------------------- super-spreader
        # influence_score: reshares received. h_index: over the author's own content.
        received = np.zeros(n_actors, dtype=np.float64)
        by_author: Dict[int, List[int]] = defaultdict(list)
        for c_slot, n in enumerate(self._content_n):
            a_slot = self._content_author_idx[c_slot]
            received[a_slot] += n
            by_author[a_slot].append(n)
        f["influence_score"] = received
        for a_slot, counts in by_author.items():
            f["h_index"][a_slot] = h_index(counts)

        # tai_score / tash_index: EMAs over per-slot influence and per-slot h-index.
        f["tai_score"], f["tash_index"] = time_aware_scores(
            ts,
            content,
            content_author,
            n_actors,
            origin=self.window_start.timestamp(),
            slot_seconds=self.slot_seconds,
            tash_alpha=self.tash_alpha,
            tai_alpha=self.tai_alpha,
        )

        # ---------------------------------------------------------------- amplifier
        counts = np.bincount(actor, minlength=n_actors).astype(np.float64)
        f["repost_count"] = counts

        first_seen = np.full(n_actors, np.inf)
        last_seen = np.full(n_actors, -np.inf)
        np.minimum.at(first_seen, actor, ts)
        np.maximum.at(last_seen, actor, ts)
        span_days = np.where(
            np.isfinite(first_seen) & np.isfinite(last_seen),
            (last_seen - first_seen) / 86400.0,
            0.0,
        )
        window_days = max(
            (self.window_end - self.window_start).total_seconds() / 86400.0, 1e-9
        )
        # Rate over the window, not over a per-user deque of the last 100 events: one
        # float per user instead of ~600 MB of deques, and easier to write down.
        f["repost_rate"] = counts / window_days

        # ear_index: (1/|P_u|) * sum over reshared posts of (N_p - r_u(p) + 1),
        # exactly as defined in Verdolotti et al. r is the 1-based chronological rank
        # of this user's reshare within the post's cascade; N_p is its final size,
        # which is why pass 1 exists.
        rank_so_far = np.zeros(len(self._content_n), dtype=np.int64)
        ear_sum = np.zeros(n_actors, dtype=np.float64)
        ear_n = np.zeros(n_actors, dtype=np.float64)
        # amplification_breadth: distinct authors amplified.
        breadth: Dict[int, set] = defaultdict(set)

        for i in range(ts.shape[0]):
            c = content[i]
            a = actor[i]
            rank_so_far[c] += 1
            r = rank_so_far[c]
            n_p = content_n[c]
            ear_sum[a] += float(n_p - r + 1)
            ear_n[a] += 1.0
            breadth[a].add(int(content_author[c]))

        with np.errstate(invalid="ignore", divide="ignore"):
            f["ear_index"] = np.where(ear_n > 0, ear_sum / np.maximum(ear_n, 1), 0.0)
        for a_slot, authors in breadth.items():
            f["amplification_breadth"][a_slot] = len(authors)

        # -------------------------------------------------------------- coordinated
        self._accumulate_co_action(ts, content, actor, content_n, f, n_actors)

        X = np.column_stack([f[name] for name in FEATURE_NAMES])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return list(self._actor_ids), X

    # ------------------------------------------------------------------ internals

    def _accumulate_co_action(self, ts, content, actor, content_n, f, n_actors) -> None:
        """Co-action against a sliding window, keyed by content.

        Indexing by content is what keeps this cheap: each repost is compared only
        against other reshares of the same post inside delta_t, never against the whole
        buffer. State is O(reposts/sec * delta_t) -- a couple of thousand entries.
        """
        live: Dict[int, deque] = defaultdict(deque)  # content -> [(ts, actor)]
        expiry: deque = deque()  # (ts, content) in arrival order

        n_actions = np.zeros(n_actors, dtype=np.float64)
        n_with_coactor = np.zeros(n_actors, dtype=np.float64)
        coactor_total = np.zeros(n_actors, dtype=np.float64)
        latency_total = np.zeros(n_actors, dtype=np.float64)
        latency_n = np.zeros(n_actors, dtype=np.float64)
        niche_total = np.zeros(n_actors, dtype=np.float64)
        niche_n = np.zeros(n_actors, dtype=np.float64)

        for i in range(ts.shape[0]):
            now = ts[i]
            c = int(content[i])
            a = int(actor[i])

            cutoff = now - self.co_action_seconds
            while expiry and expiry[0][0] < cutoff:
                _, old_c = expiry.popleft()
                dq = live.get(old_c)
                if dq:
                    dq.popleft()
                    if not dq:
                        live.pop(old_c, None)

            peers = live.get(c)
            coactors = set()
            nearest = None
            if peers:
                for p_ts, p_actor in peers:
                    if p_actor == a:
                        continue
                    coactors.add(p_actor)
                    gap = now - p_ts
                    if nearest is None or gap < nearest:
                        nearest = gap

            n_actions[a] += 1
            if coactors:
                n_with_coactor[a] += 1
                coactor_total[a] += len(coactors)
                if nearest is not None:
                    latency_total[a] += nearest
                    latency_n[a] += 1
            # Swarming on content that never went viral is the coordination signal;
            # piling onto a popular post is not.
            if content_n[c] <= self.niche_threshold:
                niche_total[a] += len(coactors)
                niche_n[a] += 1

            live[c].append((now, a))
            expiry.append((now, c))

        with np.errstate(invalid="ignore", divide="ignore"):
            f["co_action_rate"] = np.where(
                n_actions > 0, n_with_coactor / np.maximum(n_actions, 1), 0.0
            )
            f["co_action_size"] = np.where(
                n_actions > 0, coactor_total / np.maximum(n_actions, 1), 0.0
            )
            # Inverted so that larger means tighter synchrony, keeping every feature
            # in the bucket positively oriented -- otherwise PC1 mixes signs.
            mean_latency = np.where(
                latency_n > 0, latency_total / np.maximum(latency_n, 1), np.inf
            )
            f["co_action_latency"] = np.where(
                np.isfinite(mean_latency), 1.0 / (1.0 + mean_latency), 0.0
            )
            f["niche_co_action"] = np.where(
                niche_n > 0, niche_total / np.maximum(niche_n, 1), 0.0
            )


def confidence_scores(
    activity: np.ndarray,
    first_seen: np.ndarray,
    last_seen: np.ndarray,
    window_start: datetime,
    window_end: datetime,
    reference_activity: int = 10,
) -> np.ndarray:
    """How much evidence supports each user's vector, in [0,1].

    volume (50%)   log-scaled total actions
    recency (30%)  time since last activity, measured to the window's end
    lifespan (20%) total days observed, as a fraction of the window

    This is data availability: how much of the account have we seen? It is deliberately
    not "evidence for the vector". A user with five reposts among 500 replies is far
    better observed than one with five reposts and nothing else -- we watched the first
    account act 500 times and it chose not to reshare, which makes its low amplifier
    score a measurement rather than an absence of data.

    Both time terms are measured against the window and normalised by its duration,
    never against the wall clock. Two bugs motivated that: using datetime.now() as the
    reference made a 2024 window score recency ~5e-10 for every user (so a >=0.5 gate
    rejected 100% of them, and the result depended on the date of the run); and
    normalising both terms by a hard-coded 30 days against a 5-day window squashed
    recency into [0.85, 1] and lifespan into [0, 0.167], so neither discriminated and
    the documented 30%/20% weights did nothing.
    """
    n = len(activity)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    window_days = max((window_end - window_start).total_seconds() / 86400.0, 1e-9)
    end_ts = window_end.timestamp()

    volume = np.minimum(
        1.0, np.log1p(activity) / math.log1p(2 * reference_activity)
    )
    days_since = np.maximum(0.0, (end_ts - last_seen) / 86400.0)
    recency = np.exp(-days_since / (window_days / 3.0))
    lifespan = np.minimum(
        1.0, np.maximum(0.0, last_seen - first_seen) / 86400.0 / window_days
    )

    conf = 0.5 * volume + 0.3 * recency + 0.2 * lifespan
    return np.clip(conf, 0.0, 1.0)
