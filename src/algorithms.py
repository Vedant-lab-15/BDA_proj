"""
SentinelPay — Core Algorithm Engine
====================================
BDA Syllabus Coverage:
  Module 4 – Stream Mining  : BloomFilter, DGIMAlgorithm
  Module 5 – Distance Measures: cosine_distance, mahalanobis_distance

Reference Textbook: Mining of Massive Datasets (Leskovec, Rajaraman, Ullman)
  • Bloom Filter  — Chapter 4.3
  • DGIM          — Chapter 4.2
  • Cosine Dist.  — Chapter 3.5
  • Mahalanobis   — Chapter 3.5 / statistical extension
"""

from __future__ import annotations

import json
import math
import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import mmh3
    _HAS_MMH3 = True
except ImportError:          # graceful fallback — mmh3 preferred but not mandatory
    _HAS_MMH3 = False

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ░░  MODULE 4 – STREAM MINING  ░░
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
#  1. BLOOM FILTER
# ══════════════════════════════════════════════════════════════════════════════
class BloomFilter:
    """
    Probabilistic set-membership data structure using a bit-array and k
    independent hash functions.

    False-positive probability:
        P_fp ≈ (1 − e^(−kn/m))^k
    where:
        k = number of hash functions
        n = number of inserted elements
        m = size of bit array (bits)

    Space complexity : O(m) bits
    Query complexity : O(k) — constant with respect to n
    """

    def __init__(self, capacity: int = 10_000, fp_rate: float = 0.01):
        """
        Args:
            capacity : Expected number of distinct elements (n).
            fp_rate  : Desired false-positive probability (ε).
        """
        self.capacity = capacity
        self.fp_rate = fp_rate

        # Optimal bit-array size: m = −n ln ε / (ln 2)²
        self.m: int = self._optimal_m(capacity, fp_rate)
        # Optimal hash count:   k = (m/n) ln 2
        self.k: int = self._optimal_k(self.m, capacity)

        self._bit_array: bytearray = bytearray(math.ceil(self.m / 8))
        self._count: int = 0          # items inserted
        self._fp_history: List[float] = []

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _optimal_m(n: int, fp_rate: float) -> int:
        return max(1, int(-n * math.log(fp_rate) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_k(m: int, n: int) -> int:
        return max(1, int((m / n) * math.log(2)))

    def _hashes(self, item: str) -> List[int]:
        """
        Generate k independent hash positions in [0, m).

        Strategy: double hashing — h_i(x) = (h1(x) + i·h2(x)) mod m
        This avoids storing k full hash functions while maintaining
        statistical independence (Kirsch & Mitzenmacher, 2008).
        """
        positions: List[int] = []
        if _HAS_MMH3:
            h1 = mmh3.hash(item, seed=0, signed=False)
            h2 = mmh3.hash(item, seed=1, signed=False)
        else:
            raw1 = hashlib.md5(item.encode()).digest()
            raw2 = hashlib.sha1(item.encode()).digest()
            h1 = int.from_bytes(raw1[:8], "little")
            h2 = int.from_bytes(raw2[:8], "little")

        for i in range(self.k):
            positions.append((h1 + i * h2) % self.m)
        return positions

    def _set_bit(self, pos: int) -> None:
        byte_idx, bit_idx = divmod(pos, 8)
        self._bit_array[byte_idx] |= (1 << bit_idx)

    def _get_bit(self, pos: int) -> bool:
        byte_idx, bit_idx = divmod(pos, 8)
        return bool(self._bit_array[byte_idx] & (1 << bit_idx))

    # ── public interface ──────────────────────────────────────────────────────

    def add(self, item: str) -> None:
        """Insert an element — sets k bit positions."""
        for pos in self._hashes(item):
            self._set_bit(pos)
        self._count += 1

    def contains(self, item: str) -> bool:
        """
        Query membership.
        Returns True  → element MAY be in the set (possible false positive).
        Returns False → element is DEFINITELY NOT in the set.
        """
        return all(self._get_bit(pos) for pos in self._hashes(item))

    def bulk_add(self, items) -> None:
        for item in items:
            self.add(str(item))

    # ── analytics ─────────────────────────────────────────────────────────────

    @property
    def current_fp_rate(self) -> float:
        """
        Empirical false-positive estimate based on current fill level.
        Formula: P_fp ≈ (1 − e^(−kn/m))^k
        """
        if self._count == 0:
            return 0.0
        exponent = -self.k * self._count / self.m
        return (1 - math.exp(exponent)) ** self.k

    @property
    def fill_ratio(self) -> float:
        """Fraction of bits set to 1."""
        set_bits = sum(bin(b).count("1") for b in self._bit_array)
        return set_bits / self.m

    @property
    def bit_density_vector(self) -> np.ndarray:
        """
        Returns a 1-D numpy array of bit values for visualization.
        Sampled to max 1024 bits for display performance.
        """
        bits = []
        for byte in self._bit_array:
            for i in range(8):
                bits.append((byte >> i) & 1)
        arr = np.array(bits[:self.m], dtype=np.uint8)
        if len(arr) > 1024:
            idx = np.linspace(0, len(arr) - 1, 1024, dtype=int)
            arr = arr[idx]
        return arr

    def load_blacklist(self, json_path: str) -> int:
        """
        Populate the filter from a JSON blacklist file.
        Supports keys: 'blacklisted_ips' and 'blacklisted_accounts'.
        Returns number of items loaded.
        """
        path = Path(json_path)
        if not path.exists():
            logger.warning("Blacklist file not found: %s", json_path)
            return 0

        with path.open() as fh:
            data = json.load(fh)

        loaded = 0
        for key in ("blacklisted_ips", "blacklisted_accounts"):
            for entry in data.get(key, []):
                self.add(str(entry))
                loaded += 1
        logger.info("Bloom Filter loaded %d blacklist entries (m=%d, k=%d)", loaded, self.m, self.k)
        return loaded

    def stats(self) -> dict:
        return {
            "bit_array_size_m": self.m,
            "hash_functions_k": self.k,
            "elements_inserted_n": self._count,
            "fill_ratio": round(self.fill_ratio, 4),
            "estimated_fp_rate": round(self.current_fp_rate, 6),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  2. DGIM ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DGIMBucket:
    """A single DGIM bucket: stores timestamp and size (power of 2)."""
    timestamp: int   # stream position of the most recent '1' in this bucket
    size: int        # number of 1s represented: 1, 2, 4, 8, …

    def __repr__(self) -> str:
        return f"Bucket(t={self.timestamp}, size={self.size})"


class DGIMAlgorithm:
    """
    Datar-Ganti-Indyk-Motwani (DGIM) Algorithm for counting 1s in a
    sliding window stream in O(log² N) space.

    Rule: At most 2 buckets of each size 2^i may exist at any time.
    When a third bucket of size 2^i appears, merge the two oldest into
    one bucket of size 2^(i+1).

    Estimation error guarantee: ≤ 50% of the true count.
    Space: O(log² N) buckets where N is the window size.

    Here, a '1' represents a HIGH-VALUE transaction (amount > $5,000).
    """

    HIGH_VALUE_THRESHOLD: float = 5_000.0

    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Rolling window N (number of most recent transactions).
        """
        self.window_size = window_size
        self._stream_pos: int = 0          # global position counter
        self._true_count: int = 0          # exact count (for comparison)
        # Buckets ordered newest→oldest, grouped by size
        self._buckets: deque[DGIMBucket] = deque()
        self._surge_threshold: int = 5    # DGIM estimate threshold for surge alert

    # ── internal ──────────────────────────────────────────────────────────────

    def _expire_old_buckets(self) -> None:
        """Remove buckets whose timestamp falls outside the current window."""
        cutoff = self._stream_pos - self.window_size
        # Buckets are newest-first; remove from the right (oldest)
        while self._buckets and self._buckets[-1].timestamp <= cutoff:
            self._buckets.pop()

    def _merge_if_needed(self) -> None:
        """
        Enforce the DGIM invariant: at most 2 buckets per size.
        Scan from smallest to largest, merging when 3+ buckets of same size exist.
        """
        changed = True
        while changed:
            changed = False
            size_map: dict[int, List[int]] = {}  # size → [indices]

            bucket_list = list(self._buckets)
            for idx, b in enumerate(bucket_list):
                size_map.setdefault(b.size, []).append(idx)

            for size, indices in sorted(size_map.items()):
                if len(indices) >= 3:
                    # Merge the two OLDEST (highest index = rightmost in deque = oldest)
                    i1, i2 = indices[-2], indices[-1]   # second-oldest, oldest
                    older_ts = bucket_list[i2].timestamp
                    newer_ts = bucket_list[i1].timestamp
                    new_bucket = DGIMBucket(
                        timestamp=max(older_ts, newer_ts),  # keep newer timestamp
                        size=size * 2
                    )
                    # Rebuild deque without the two merged buckets
                    new_list = [b for i, b in enumerate(bucket_list) if i not in (i1, i2)]
                    # Insert new_bucket in correct timestamp order (newest first)
                    insert_at = 0
                    for i, b in enumerate(new_list):
                        if b.timestamp < new_bucket.timestamp:
                            insert_at = i
                            break
                        insert_at = i + 1
                    new_list.insert(insert_at, new_bucket)
                    self._buckets = deque(new_list)
                    changed = True
                    break

    # ── public interface ──────────────────────────────────────────────────────

    def add_transaction(self, amount: float) -> bool:
        """
        Process one incoming transaction.

        Args:
            amount: Transaction amount in USD.
        Returns:
            True if this transaction is a high-value event (bit=1), else False.
        """
        self._stream_pos += 1
        is_high_value = amount > self.HIGH_VALUE_THRESHOLD

        if is_high_value:
            self._true_count += 1
            # Create a new size-1 bucket at the current timestamp
            new_bucket = DGIMBucket(timestamp=self._stream_pos, size=1)
            self._buckets.appendleft(new_bucket)   # newest at front

        self._expire_old_buckets()
        self._merge_if_needed()
        return is_high_value

    def estimate_count(self) -> int:
        """
        DGIM count estimate: sum all bucket sizes, but halve the oldest bucket.

        Rationale: The oldest bucket may be partially outside the window.
        Estimate = (sum of all bucket sizes) − floor(oldest_size / 2)
        """
        if not self._buckets:
            return 0
        bucket_list = list(self._buckets)
        total = sum(b.size for b in bucket_list)
        oldest_size = bucket_list[-1].size
        return max(0, total - oldest_size // 2)

    def true_count_in_window(self) -> int:
        """Exact count of high-value events in the current window (for comparison)."""
        cutoff = self._stream_pos - self.window_size
        return sum(1 for b in self._buckets if b.timestamp > cutoff and b.size == 1)

    @property
    def is_surge(self) -> bool:
        """True when DGIM estimate exceeds surge threshold."""
        return self.estimate_count() >= self._surge_threshold

    def get_bucket_tree(self) -> List[dict]:
        """Return bucket data for visualization."""
        return [
            {"timestamp": b.timestamp, "size": b.size, "log2_size": int(math.log2(b.size))}
            for b in self._buckets
        ]

    def stats(self) -> dict:
        return {
            "stream_position": self._stream_pos,
            "window_size": self.window_size,
            "bucket_count": len(self._buckets),
            "dgim_estimate": self.estimate_count(),
            "true_window_count": self.true_count_in_window(),
            "is_surge": self.is_surge,
            "buckets": self.get_bucket_tree(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ░░  MODULE 5 – DISTANCE MEASURES  ░░
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
#  3. COSINE DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """
    Cosine Distance between two feature vectors.

    Formula:
        cosine_distance(u, v) = 1 − (u · v) / (‖u‖₂ · ‖v‖₂)

    Range: [0, 2]
        0  → identical direction (no anomaly)
        1  → orthogonal (moderate anomaly)
        2  → opposite direction (extreme anomaly)

    Application here:
        u = current transaction feature vector
        v = baseline spending profile (historical mean vector)

    Args:
        u: Current transaction vector [amount, oldBalance, newBalance, …]
        v: Baseline/reference vector of the same dimensionality.
    Returns:
        Scalar float in [0, 2].
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        return 0.0   # zero-vector edge case: treat as identical

    cosine_similarity = np.dot(u, v) / (norm_u * norm_v)
    # Clamp to [-1, 1] to guard against floating-point drift
    cosine_similarity = float(np.clip(cosine_similarity, -1.0, 1.0))
    return 1.0 - cosine_similarity


# ══════════════════════════════════════════════════════════════════════════════
#  4. MAHALANOBIS DISTANCE
# ══════════════════════════════════════════════════════════════════════════════

def mahalanobis_distance(
    x: np.ndarray,
    mu: np.ndarray,
    cov: Optional[np.ndarray] = None,
    inv_cov: Optional[np.ndarray] = None,
) -> float:
    """
    Mahalanobis Distance — scale-invariant, correlation-aware anomaly metric.

    Formula:
        D_M(x) = √[ (x − μ)ᵀ Σ⁻¹ (x − μ) ]

    Where:
        x   = observation vector
        μ   = mean vector of the reference distribution
        Σ   = covariance matrix of the reference distribution
        Σ⁻¹ = precision matrix (inverse covariance)

    Key property: accounts for feature correlations via Σ⁻¹, unlike
    Euclidean distance which treats all features as independent.

    Under a multivariate normal distribution, D_M² ~ χ²(d),
    so a 3-sigma boundary corresponds to D_M ≈ √χ²_{0.997}(d).

    Args:
        x       : Observation vector (d-dimensional).
        mu      : Mean vector of the reference population.
        cov     : Covariance matrix Σ. Provide either cov or inv_cov.
        inv_cov : Pre-computed Σ⁻¹ (faster when called repeatedly).
    Returns:
        Scalar float ≥ 0.
    """
    x   = np.asarray(x,  dtype=float)
    mu  = np.asarray(mu, dtype=float)
    diff = x - mu

    if inv_cov is None:
        if cov is None:
            raise ValueError("Provide either cov or inv_cov.")
        cov = np.asarray(cov, dtype=float)
        # Add regularisation λI to ensure invertibility (handles near-singular Σ)
        reg = 1e-6 * np.eye(cov.shape[0])
        inv_cov = np.linalg.inv(cov + reg)

    maha_sq = float(diff @ inv_cov @ diff)
    return math.sqrt(max(0.0, maha_sq))


# ══════════════════════════════════════════════════════════════════════════════
#  5. BASELINE PROFILE BUILDER (utility for app.py)
# ══════════════════════════════════════════════════════════════════════════════

class BaselineProfile:
    """
    Maintains a rolling Welford online mean/covariance estimator for
    computing Cosine and Mahalanobis distances against a streaming baseline.

    Uses Welford's numerically stable online algorithm to update μ and Σ
    in O(d²) per sample without storing the full history.
    """

    def __init__(self, n_features: int = 3):
        self.n_features = n_features
        self.n_samples  = 0
        self._mean      = np.zeros(n_features)
        self._M2        = np.zeros((n_features, n_features))   # sum of squared deviations

    def update(self, x: np.ndarray) -> None:
        """Welford online update for mean and covariance."""
        x = np.asarray(x, dtype=float)
        self.n_samples += 1
        delta  = x - self._mean
        self._mean += delta / self.n_samples
        delta2 = x - self._mean
        self._M2 += np.outer(delta, delta2)

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        if self.n_samples < 2:
            return np.eye(self.n_features)
        return self._M2 / (self.n_samples - 1)

    @property
    def inv_covariance(self) -> np.ndarray:
        cov = self.covariance
        reg = 1e-6 * np.eye(self.n_features)
        try:
            return np.linalg.inv(cov + reg)
        except np.linalg.LinAlgError:
            return np.eye(self.n_features)

    def cosine_dist(self, x: np.ndarray) -> float:
        return cosine_distance(x, self._mean)

    def mahalanobis_dist(self, x: np.ndarray) -> float:
        return mahalanobis_distance(x, self._mean, inv_cov=self.inv_covariance)

    def ready(self, min_samples: int = 30) -> bool:
        return self.n_samples >= min_samples
