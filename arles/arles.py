"""
ARLES: Archetype Learning System

Optimized online learning system for social media information spreader archetypes.
Learns three archetypes: Superspreaders, Amplifiers, and Coordinated actors.
"""

import csv
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .metrics import (
    ArchetypeMetric,
    get_default_amplifier_metrics,
    get_default_coordinated_metrics,
    get_default_superspreader_metrics,
)

warnings.filterwarnings("ignore")


class MalformedActionError(ValueError):
    """Raised when an action row cannot be parsed.

    Never swallow this silently: a row that cannot be parsed must be counted and
    reported, not replaced by a plausible-looking default. See parse_timestamp.
    """


_TIMESTAMP_RE = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<frac>\d+))?"
    r"(?:\s*(?P<tz>Z|z|[+-]\d{2}:?\d{2}))?$"
)

_PLC_URI_RE = re.compile(r"^at://did:plc:([^/]+)/")


def author_of_uri(uri: Optional[str]) -> Optional[str]:
    """Return the author DID of an AT-URI, or None if it is not a did:plc URI.

    An AT-Protocol record lives in its author's repository, so the DID embedded in the
    URI *is* the author:

        at://did:plc:vlpy6zuqqum5tumv7b6dw5fp/app.bsky.feed.post/3l2dszqkmqt25
                     ^^^^^^^^^^^^^^^^^^^^^^^^ the author

    This matters because the dataset is a sample: the post being reshared is present in
    the file for only 0.09% of reposts (470 of 509,844 in bluesky_sampled_clean_small.csv),
    so resolving a reshare's author by looking the original post up in the stream fails
    almost always. Parsing the URI resolves 100% of them without needing the original.

    The returned id has no "did:plc:" prefix, matching the author_user_id column exactly
    (verified on 400k rows: for post/repost/reply/quote the parsed DID equals
    author_user_id 100% of the time).

    did:web identities (e.g. at://did:web:genco.me/...) are rare and return None.
    """
    if not uri:
        return None
    m = _PLC_URI_RE.match(uri)
    return m.group(1) if m else None


def parse_timestamp(value: Any) -> datetime:
    """Parse an action timestamp into a timezone-aware UTC datetime.

    The dataset stores ISO-8601 with an explicit UTC offset, in three variants that
    all occur in practice:

        2024-08-23 00:00:00+00:00              (no fractional part)
        2024-08-23 00:03:48.226000+00:00       (microseconds, the common case)
        2024-10-03 01:00:07.649171700+00:00    (nanoseconds, written by pandas)

    datetime.fromisoformat handles only the first two before Python 3.11, so the
    fractional part is normalised to 6 digits before parsing.

    Why this is strict: the previous implementation tried "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S" and "%Y-%m-%d" with strptime and fell back to
    datetime.now() when all three failed. Every one of them rejects the "+00:00"
    offset, so *every* row took the fallback and the whole action stream was
    replaced by the wall-clock time of the run. A multi-day window collapsed into
    the seconds it took to read the file, which zeroed the TASH-index outright and
    corrupted every other time-dependent metric. A timestamp that cannot be parsed
    is now an error, never a guess.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        m = _TIMESTAMP_RE.match(value.strip())
        if m is None:
            raise MalformedActionError(f"unparseable timestamp: {value!r}")

        head = m.group("head")
        frac = m.group("frac") or ""
        tz = m.group("tz") or "+00:00"

        # Truncate (or pad) the fractional part to microsecond resolution.
        micros = (frac + "000000")[:6]
        if tz in ("Z", "z"):
            tz = "+00:00"
        elif ":" not in tz:  # "+0000" -> "+00:00"
            tz = tz[:3] + ":" + tz[3:]

        try:
            dt = datetime.fromisoformat(f"{head}.{micros}{tz}")
        except ValueError as exc:  # pragma: no cover - regex should prevent this
            raise MalformedActionError(f"unparseable timestamp: {value!r}") from exc
    else:
        raise MalformedActionError(f"unparseable timestamp: {value!r}")

    # Normalise to UTC so that .timestamp() is comparable across rows.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class Action:
    """Represents a single social media action."""

    action_id: str
    created_at: datetime
    author_user_id: str
    target_user_id: Optional[str]
    original_action_id: Optional[str]
    activity_type: str  # post, repost, reply, quote, follow, block, ...
    text: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create Action from dictionary, handling datetime parsing.

        Raises MalformedActionError if the row cannot be parsed.
        """
        created_at = parse_timestamp(data["created_at"])
        if not data.get("action_id") or not data.get("author_user_id"):
            raise MalformedActionError("missing action_id or author_user_id")

        return cls(
            action_id=data["action_id"],
            created_at=created_at,
            author_user_id=data["author_user_id"],
            target_user_id=data.get("target_user_id"),
            original_action_id=data.get("original_action_id"),
            activity_type=data["activity_type"],
            text=data.get("text"),
        )


class ArchetypeLearner:
    """
    Optimized online learning system with pluggable metrics.

    Key optimizations:
    - Numpy arrays for fast vectorized operations
    - Delayed PCA computation until get_archetypes() is called
    - Pluggable metric architecture
    - Efficient streaming with proper progress bars
    """

    def __init__(
        self,
        superspreader_metrics: Optional[List[ArchetypeMetric]] = None,
        amplifier_metrics: Optional[List[ArchetypeMetric]] = None,
        coordinated_metrics: Optional[List[ArchetypeMetric]] = None,
        min_actions_for_confidence: int = 10,
        initial_capacity: int = 1000,
    ):
        """
        Initialize the archetype learning system with pluggable metrics.

        Args:
            superspreader_metrics: List of metrics for superspreader archetype (None = defaults)
            amplifier_metrics: List of metrics for amplifier archetype (None = defaults)
            coordinated_metrics: List of metrics for coordinated archetype (None = defaults)
            min_actions_for_confidence: Minimum actions needed for reliable estimates
            initial_capacity: Initial capacity for numpy arrays
        """
        self.min_actions = min_actions_for_confidence
        self._capacity = initial_capacity

        # User ID mapping
        self.user_id_to_idx: Dict[str, int] = {}
        self.idx_to_user_id: Dict[int, str] = {}
        self.next_user_idx = 0

        # Temporal tracking (vectorized)
        self.first_seen = np.zeros(initial_capacity, dtype=np.float64)
        self.last_seen = np.zeros(initial_capacity, dtype=np.float64)
        self.action_count = np.zeros(initial_capacity, dtype=np.int32)

        # Shared state for metrics
        self.post_to_author: Dict[str, int] = {}
        self.post_reshares: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

        # Pluggable metrics (use defaults if None provided)
        self.superspreader_metrics = (
            superspreader_metrics or get_default_superspreader_metrics()
        )
        self.amplifier_metrics = amplifier_metrics or get_default_amplifier_metrics()
        self.coordinated_metrics = (
            coordinated_metrics or get_default_coordinated_metrics()
        )

        # Initialize all metrics
        for metric in (
            self.superspreader_metrics
            + self.amplifier_metrics
            + self.coordinated_metrics
        ):
            metric.initialize(initial_capacity)

        # Cached PCA models for dimensionality reduction
        self.pca_superspreader = PCA(n_components=1)
        self.pca_amplifier = PCA(n_components=1)
        self.pca_coordinated = PCA(n_components=1)
        self.pca_needs_update = True

        # Action counter
        self.total_actions_processed = 0

    def _get_or_create_user_idx(self, user_id: str) -> int:
        """Get user index or create new one, expanding arrays if needed."""
        if user_id in self.user_id_to_idx:
            return self.user_id_to_idx[user_id]

        if self.next_user_idx >= self._capacity:
            self._expand_arrays()

        idx = self.next_user_idx
        self.user_id_to_idx[user_id] = idx
        self.idx_to_user_id[idx] = user_id
        self.next_user_idx += 1

        return idx

    def _expand_arrays(self) -> None:
        """Double the capacity of all arrays."""
        new_capacity = self._capacity * 2

        self.first_seen = np.resize(self.first_seen, new_capacity)
        self.last_seen = np.resize(self.last_seen, new_capacity)
        self.action_count = np.resize(self.action_count, new_capacity)

        # Expand all metric arrays
        for metric in (
            self.superspreader_metrics
            + self.amplifier_metrics
            + self.coordinated_metrics
        ):
            metric.expand(new_capacity)

        self._capacity = new_capacity

    def process_action(self, action: Action) -> None:
        """Process a single action and update internal state."""
        user_idx = self._get_or_create_user_idx(action.author_user_id)
        timestamp = action.created_at.timestamp()

        # Update temporal information
        if self.first_seen[user_idx] == 0:
            self.first_seen[user_idx] = timestamp
        self.last_seen[user_idx] = timestamp
        self.action_count[user_idx] += 1

        # Update all metrics
        for metric in self.superspreader_metrics:
            metric.update(user_idx, action, timestamp, self)
        for metric in self.amplifier_metrics:
            metric.update(user_idx, action, timestamp, self)
        for metric in self.coordinated_metrics:
            metric.update(user_idx, action, timestamp, self)

        self.pca_needs_update = True
        self.total_actions_processed += 1

    def process_action_batch(
        self, actions: List[Action], show_progress: bool = True
    ) -> None:
        """Process a batch of actions."""
        iterator: Any
        if show_progress:
            iterator = tqdm(
                actions, desc="Processing actions", miniters=max(1, len(actions) // 100)
            )
        else:
            iterator = actions

        for action in iterator:
            self.process_action(action)

    def process_csv_stream(
        self,
        filepath: str,
        show_progress: bool = True,
        stream_mode: bool = False,
        max_skip_ratio: float = 0.01,
    ) -> Dict[str, int]:
        """
        Process actions from a CSV file, row by row, without loading it into memory.

        Malformed rows are counted and reported rather than silently dropped. If more
        than `max_skip_ratio` of rows fail to parse the run aborts: a systematically
        unparseable column (e.g. a timestamp format change) should stop the pipeline,
        not quietly degrade every downstream metric.

        Returns a dict with the processed/skipped counts.
        """

        with open(filepath, "r", encoding="utf-8") as f:

            total_records = None
            pbar = None

            reader = csv.DictReader(f)

            if not stream_mode:
                total_records = sum(1 for _ in reader)
                f.seek(0)  # Reset file pointer
                reader = csv.DictReader(f)  # Reset reader after counting

            if show_progress:

                pbar = tqdm(
                    total=total_records,
                    desc="Processing CSV",
                    unit=" actions",
                )

            n_read = 0
            n_skipped = 0
            first_error: Optional[str] = None

            for row in reader:
                n_read += 1

                try:

                    action = Action.from_dict(row)
                    self.process_action(action)

                    if pbar is not None:
                        pbar.update()

                except MalformedActionError as exc:
                    n_skipped += 1
                    if first_error is None:
                        first_error = str(exc)
                    if pbar is not None:
                        pbar.update()

            if pbar is not None:
                pbar.close()

        if n_read and (n_skipped / n_read) > max_skip_ratio:
            raise MalformedActionError(
                f"{n_skipped}/{n_read} rows ({100 * n_skipped / n_read:.1f}%) could not be "
                f"parsed in {filepath}, above the {100 * max_skip_ratio:.1f}% tolerance. "
                f"First error: {first_error}"
            )

        if n_skipped:
            print(f"  skipped {n_skipped}/{n_read} malformed rows (first: {first_error})")

        return {"read": n_read, "processed": n_read - n_skipped, "skipped": n_skipped}

    def _extract_archetype_vectors(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Extract feature matrices from all metrics."""
        n_users = self.next_user_idx
        user_ids = [self.idx_to_user_id[i] for i in range(n_users)]

        # Stack metrics for each archetype
        superspreader = np.column_stack(
            [metric.get_values(n_users) for metric in self.superspreader_metrics]
        )

        amplifier = np.column_stack(
            [metric.get_values(n_users) for metric in self.amplifier_metrics]
        )

        coordinated = np.column_stack(
            [metric.get_values(n_users) for metric in self.coordinated_metrics]
        )

        return superspreader, amplifier, coordinated, user_ids

    # def _apply_dimensionality_reduction(
    #     self, superspreader: np.ndarray, amplifier: np.ndarray, coordinated: np.ndarray
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """Apply PCA to reduce each archetype to 1D."""
    #     n_users = superspreader.shape[0]

    #     if self.pca_needs_update:
    #         self.pca_needs_update = False

    #     def safe_pca(data: np.ndarray, pca_model: PCA) -> np.ndarray:
    #         data_norm = data.copy()
    #         col_min = data.min(axis=0)
    #         col_max = data.max(axis=0)
    #         col_range = col_max - col_min

    #         mask = col_range > 1e-10
    #         data_norm[:, mask] = (data[:, mask] - col_min[mask]) / col_range[mask]

    #         if n_users > 1:
    #             reduced = pca_model.fit_transform(data_norm)
    #             reduced_min, reduced_max = reduced.min(), reduced.max()
    #             if reduced_max > reduced_min:
    #                 reduced = (reduced - reduced_min) / (reduced_max - reduced_min)
    #             return reduced.flatten()
    #         else:
    #             return np.array([0.5])

    #     ss_1d = safe_pca(superspreader, self.pca_superspreader)
    #     amp_1d = safe_pca(amplifier, self.pca_amplifier)
    #     coord_1d = safe_pca(coordinated, self.pca_coordinated)

    #     return ss_1d, amp_1d, coord_1d

    def _apply_dimensionality_reduction(
        self, superspreader: np.ndarray, amplifier: np.ndarray, coordinated: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply global feature scaling across [SS | AMP | COORD] per user, then
        apply PCA separately to each archetype chunk to produce three 1D scores.

        Returns: (ss_1d, amp_1d, coord_1d) each as 1D numpy arrays in [0,1].
        """
        n_users = superspreader.shape[0]
        # Handle empty case
        if n_users == 0:
            return np.array([]), np.array([]), np.array([])

        # Concatenate horizontally to build global feature matrix
        combined = np.hstack([superspreader, amplifier, coordinated])

        # Replace NaN/inf with finite numbers
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

        # Global scaling

        # Keep scaler as attribute so future calls can reuse if desired
        if not hasattr(self, "global_scaler") or self.pca_needs_update:
            # (re)fit scaler when PCA needs update or scaler not present
            self.global_scaler = StandardScaler()
            try:
                scaled_combined = self.global_scaler.fit_transform(combined)
            except Exception:
                # fallback robust handling if fit_transform fails for degenerate data
                scaled_combined = (combined - np.nanmean(combined, axis=0)) / (
                    np.nanstd(combined, axis=0) + 1e-8
                )
        else:
            scaled_combined = self.global_scaler.transform(combined)

        # Split scaled features back to the three chunks
        ss_cols = superspreader.shape[1]
        amp_cols = amplifier.shape[1]
        coord_cols = coordinated.shape[1]

        ss_scaled = (
            scaled_combined[:, :ss_cols] if ss_cols > 0 else np.zeros((n_users, 0))
        )
        amp_scaled = (
            scaled_combined[:, ss_cols : ss_cols + amp_cols]
            if amp_cols > 0
            else np.zeros((n_users, 0))
        )
        coord_scaled = (
            scaled_combined[:, ss_cols + amp_cols :]
            if coord_cols > 0
            else np.zeros((n_users, 0))
        )

        # Helper: safe PCA per chunk
        def safe_chunk_pca(data: np.ndarray, model_attr_name: str) -> np.ndarray:
            if data.size == 0 or data.shape[1] == 0:
                return np.full(n_users, 0.5, dtype=np.float32)
            if n_users < 2:
                return np.full(n_users, 0.5, dtype=np.float32)

            # Initialize or reuse PCA model
            if not hasattr(self, model_attr_name) or self.pca_needs_update:
                setattr(self, model_attr_name, PCA(n_components=1))

            model = getattr(self, model_attr_name)

            try:
                if self.pca_needs_update:
                    reduced = model.fit_transform(data).flatten()
                else:
                    reduced = model.transform(data).flatten()
            except Exception:
                reduced = data[:, 0].astype(np.float32).flatten()

            rmin, rmax = reduced.min(), reduced.max()
            if rmax > rmin:
                reduced = (reduced - rmin) / (rmax - rmin)
            else:
                reduced = np.full_like(reduced, 0.5)

            return reduced.astype(np.float32)

        ss_1d = safe_chunk_pca(ss_scaled, "pca_superspreader")
        amp_1d = safe_chunk_pca(amp_scaled, "pca_amplifier")
        coord_1d = safe_chunk_pca(coord_scaled, "pca_coordinated")

        # After computing, mark PCA up-to-date
        self.pca_needs_update = False

        return ss_1d, amp_1d, coord_1d

    def _compute_confidence_scores(
        self,
        n_users: int,
        reference_time: Optional[float] = None,
        window_days: Optional[float] = None,
    ) -> np.ndarray:
        """How much to trust a user's archetype vector, in [0,1].

        Three components, weighted 50/30/20 as in the paper:
          volume   -- log-scaled action count, saturating at 2 * min_actions
          recency  -- how close to the end of the window the user was last active
          lifespan -- how much of the window the user's activity spans

        Both time terms are measured against the *observed window*, never against the
        wall clock, and are normalised by the window's own duration.

        Two bugs are fixed here, and they interact:

        1. `datetime.now()` was used as the reference for recency. With timestamps
           correctly parsed (they previously all *were* now(), which masked this), a
           2024 window scored recency ~= exp(-644/30) ~= 5e-10 for every user, so
           confidence could not exceed ~0.53 and the >= 0.5 gate rejected everyone.
           Worse, the result depended on the date the script was run.

        2. Both time terms were normalised by a hard-coded 30 days while the analysis
           window is 5 days. Recency was therefore confined to [exp(-5/30), 1] = [0.85, 1]
           and lifespan to [0, 0.167]: both near-constant across users, which silently
           reduced confidence to the volume term alone and made the 30% and 20% weights
           decorative. Normalising by the window duration restores their range to [0,1]
           and makes the weights mean what they say.

        Args:
            reference_time: end of the window, as a POSIX timestamp. Defaults to the
                latest action observed.
            window_days: duration of the window in days. Defaults to the span of the
                observed data.
        """
        if n_users == 0:
            return np.zeros(0, dtype=np.float32)

        last_seen = self.last_seen[:n_users]
        first_seen = self.first_seen[:n_users]

        if reference_time is None:
            reference_time = float(last_seen.max())

        if window_days is None:
            seen = first_seen[first_seen > 0]
            earliest = float(seen.min()) if seen.size else reference_time
            window_days = (reference_time - earliest) / 86400
        # A window of zero duration (single instant) leaves the time terms undefined;
        # fall back to a nominal day so they stay finite rather than dividing by zero.
        window_days = max(float(window_days), 1e-9)

        volume_scores = np.minimum(
            1.0, np.log1p(self.action_count[:n_users]) / np.log1p(self.min_actions * 2)
        )

        days_since_last = np.maximum(0.0, (reference_time - last_seen) / 86400)
        # Decay over a third of the window: a user last seen a full window ago scores
        # exp(-3) ~= 0.05 rather than an indistinguishable ~0.85.
        recency_scores = np.exp(-days_since_last / (window_days / 3.0))

        span_days = (last_seen - first_seen) / 86400
        lifespan_scores = np.minimum(1.0, np.maximum(0.0, span_days) / window_days)

        confidence = 0.5 * volume_scores + 0.3 * recency_scores + 0.2 * lifespan_scores
        return np.clip(confidence, 0.0, 1.0).astype(np.float32)

    def get_archetypes(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Get archetype estimates for all users.
        PCA computed lazily here.
        """
        n_users = self.next_user_idx
        if n_users == 0:
            return {}

        superspreader, amplifier, coordinated, user_ids = (
            self._extract_archetype_vectors()
        )

        ss_1d, amp_1d, coord_1d = self._apply_dimensionality_reduction(
            superspreader, amplifier, coordinated
        )

        confidences = self._compute_confidence_scores(n_users)

        results = {}
        for i in range(n_users):
            uid = user_ids[i]
            archetypal_vector = np.array(
                [ss_1d[i], amp_1d[i], coord_1d[i]], dtype=np.float32
            )
            results[uid] = (archetypal_vector, float(confidences[i]))

        return results

    def get_user_archetype(self, user_id: str) -> Optional[Tuple[np.ndarray, float]]:
        """Get archetype estimate for a specific user."""
        if user_id not in self.user_id_to_idx:
            return None

        all_archetypes = self.get_archetypes()
        return all_archetypes.get(user_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_actions_processed": self.total_actions_processed,
            "total_users": self.next_user_idx,
            "total_posts_tracked": len(self.post_reshares),
            "array_capacity": self._capacity,
            "memory_efficiency": f"{self.next_user_idx}/{self._capacity} ({100*self.next_user_idx/self._capacity:.1f}%)",
            "superspreader_metrics": [m.get_name() for m in self.superspreader_metrics],
            "amplifier_metrics": [m.get_name() for m in self.amplifier_metrics],
            "coordinated_metrics": [m.get_name() for m in self.coordinated_metrics],
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    from .metrics import (
        AvgResharesPerPost,
        BotLikeBehavior,
        CascadeAmplification,
        CoRepostCount,
        PostingFrequency,
        RepostCount,
        TASHIndexMetric,
        TemporalBurstiness,
        WeightedRepostPosition,
    )

    # Example 1: Default metrics
    print("=" * 80)
    print("Example 1: Using default metrics")
    print("=" * 80)

    learner = ArchetypeLearner()

    actions = [
        Action(
            action_id="1",
            created_at=datetime(2024, 1, 1, 10, 0),
            author_user_id="user_a",
            target_user_id=None,
            original_action_id=None,
            activity_type="post",
            text="Hello world",
        ),
        Action(
            action_id="2",
            created_at=datetime(2024, 1, 1, 10, 5),
            author_user_id="user_b",
            target_user_id=None,
            original_action_id="1",
            activity_type="repost",
            text=None,
        ),
        Action(
            action_id="3",
            created_at=datetime(2024, 1, 1, 10, 6),
            author_user_id="user_c",
            target_user_id=None,
            original_action_id="1",
            activity_type="repost",
            text=None,
        ),
    ]

    for action in actions:
        learner.process_action(action)

    archetypes = learner.get_archetypes()
    print("\nArchetype Results:")
    for user_id, (vector, confidence) in archetypes.items():
        print(
            f"{user_id}: SS={vector[0]:.3f}, Amp={vector[1]:.3f}, "
            f"Coord={vector[2]:.3f}, Conf={confidence:.3f}"
        )

    print(f"\nStats: {learner.get_stats()}")

    # Example 2: Custom metrics
    print("\n" + "=" * 80)
    print("Example 2: Custom metric configuration")
    print("=" * 80)

    learner_custom = ArchetypeLearner(
        superspreader_metrics=[
            TASHIndexMetric(time_slot_size=5.0, alpha_smoothing=0.4),
            AvgResharesPerPost(),
            PostingFrequency(),
        ],
        amplifier_metrics=[
            RepostCount(),
            WeightedRepostPosition(),
            CascadeAmplification(),
        ],
        coordinated_metrics=[
            CoRepostCount(),
            TemporalBurstiness(),
            BotLikeBehavior(),
        ],
    )

    # Simulate more complex activity
    for i in range(10):
        learner_custom.process_action(
            Action(
                action_id=f"post_{i}",
                created_at=datetime(2024, 1, 1, 10, i),
                author_user_id="user_a",
                target_user_id=None,
                original_action_id=None,
                activity_type="post",
                text=f"Post {i}",
            )
        )

        # Multiple users repost
        for j in range(i):
            learner_custom.process_action(
                Action(
                    action_id=f"repost_{i}_{j}",
                    created_at=datetime(2024, 1, 1, 10, i, 30),
                    author_user_id=f"user_{j}",
                    target_user_id=None,
                    original_action_id=f"post_{i}",
                    activity_type="repost",
                    text=None,
                )
            )

    archetypes_custom = learner_custom.get_archetypes()
    print("\nArchetype Results with custom metrics:")
    for user_id, (vector, confidence) in sorted(
        archetypes_custom.items(), key=lambda x: x[1][0][0], reverse=True
    )[:5]:
        print(
            f"{user_id}: SS={vector[0]:.3f}, Amp={vector[1]:.3f}, "
            f"Coord={vector[2]:.3f}, Conf={confidence:.3f}"
        )

    print(f"\nStats: {learner_custom.get_stats()}")
