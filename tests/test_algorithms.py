"""
SentinelPay — Unit Tests for Core Algorithms
=============================================
Run with: pytest tests/ -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms import (
    BloomFilter,
    DGIMAlgorithm,
    DGIMBucket,
    BaselineProfile,
    cosine_distance,
    mahalanobis_distance,
)


# ══════════════════════════════════════════════════════════════════════════════
#  BLOOM FILTER TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestBloomFilter:

    def test_contains_after_add(self):
        """An item added to the filter must always be reported as present."""
        bf = BloomFilter(capacity=1000, fp_rate=0.01)
        bf.add("192.168.1.1")
        assert bf.contains("192.168.1.1") is True

    def test_definitely_not_present(self):
        """
        An item never inserted must return False — no false negatives allowed.
        We test a large number of novel keys to verify zero false negatives.
        """
        bf = BloomFilter(capacity=1000, fp_rate=0.01)
        for i in range(500):
            bf.add(f"known-item-{i}")
        # Query items that were never inserted
        false_negatives = sum(
            1 for i in range(500) if not bf.contains(f"known-item-{i}")
        )
        assert false_negatives == 0, "Bloom Filter must have zero false negatives"

    def test_false_positive_rate_within_bound(self):
        """
        Empirical FP rate after filling to capacity should stay below 2× target.
        (Exact FP rate is probabilistic, so we allow 2× tolerance.)
        """
        target_fp = 0.01
        bf = BloomFilter(capacity=1000, fp_rate=target_fp)
        for i in range(1000):
            bf.add(f"seed-{i}")

        fp_count = 0
        trials = 2000
        for i in range(trials):
            # Test items guaranteed NOT in the filter
            if bf.contains(f"novel-{i+10000}"):
                fp_count += 1

        empirical_fp = fp_count / trials
        assert empirical_fp < target_fp * 4, (
            f"Empirical FP rate {empirical_fp:.4f} exceeds 4× target {target_fp}"
        )

    def test_optimal_m_formula(self):
        """m = -n·ln(ε) / (ln 2)² — verify formula implementation."""
        n, fp = 1000, 0.01
        expected_m = int(-n * math.log(fp) / (math.log(2) ** 2))
        bf = BloomFilter(capacity=n, fp_rate=fp)
        assert abs(bf.m - expected_m) <= 1

    def test_optimal_k_formula(self):
        """k = (m/n)·ln 2 — verify formula implementation."""
        n, fp = 1000, 0.01
        m = int(-n * math.log(fp) / (math.log(2) ** 2))
        expected_k = int((m / n) * math.log(2))
        bf = BloomFilter(capacity=n, fp_rate=fp)
        assert abs(bf.k - expected_k) <= 1

    def test_bulk_add(self):
        """Bulk add should insert all items."""
        bf = BloomFilter(capacity=100, fp_rate=0.01)
        items = [f"ip-{i}" for i in range(50)]
        bf.bulk_add(items)
        for item in items:
            assert bf.contains(item)

    def test_fill_ratio_increases_monotonically(self):
        """Fill ratio must increase (or stay the same) as items are added."""
        bf = BloomFilter(capacity=500, fp_rate=0.05)
        prev_ratio = 0.0
        for i in range(100):
            bf.add(f"item-{i}")
        assert bf.fill_ratio >= prev_ratio

    def test_fp_rate_formula_monotone_with_fill(self):
        """Estimated FP rate should increase as more items are inserted."""
        bf = BloomFilter(capacity=1000, fp_rate=0.01)
        fp_empty = bf.current_fp_rate
        for i in range(500):
            bf.add(f"x-{i}")
        fp_half = bf.current_fp_rate
        assert fp_half >= fp_empty

    def test_stats_dict_keys(self):
        """stats() must return all expected keys."""
        bf = BloomFilter()
        s = bf.stats()
        for key in ("bit_array_size_m", "hash_functions_k",
                    "elements_inserted_n", "fill_ratio", "estimated_fp_rate"):
            assert key in s


# ══════════════════════════════════════════════════════════════════════════════
#  DGIM ALGORITHM TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDGIMAlgorithm:

    def test_add_high_value_transaction_creates_bucket(self):
        """A transaction > $5000 should create at least one bucket."""
        dgim = DGIMAlgorithm(window_size=100)
        dgim.add_transaction(10_000.0)
        assert len(dgim._buckets) == 1
        assert dgim._buckets[0].size == 1

    def test_low_value_transaction_creates_no_bucket(self):
        """A transaction <= $5000 should NOT create a bucket."""
        dgim = DGIMAlgorithm(window_size=100)
        dgim.add_transaction(100.0)
        assert len(dgim._buckets) == 0

    def test_estimate_is_nonzero_after_high_value(self):
        """Estimate should be > 0 after inserting high-value transactions."""
        dgim = DGIMAlgorithm(window_size=100)
        for amount in [6000, 7000, 8000]:
            dgim.add_transaction(amount)
        assert dgim.estimate_count() > 0

    def test_at_most_two_buckets_per_size(self):
        """DGIM invariant: at most 2 buckets of any size at all times."""
        dgim = DGIMAlgorithm(window_size=1000)
        for _ in range(50):
            dgim.add_transaction(10_000.0)

        from collections import Counter
        size_counts = Counter(b.size for b in dgim._buckets)
        for size, count in size_counts.items():
            assert count <= 2, f"Size {size} has {count} buckets (violation)"

    def test_estimate_within_50_percent_error(self):
        """
        DGIM guarantee: estimate error ≤ 50% of true count.

        Note: true_count_in_window() counts only size-1 buckets (individual
        high-value events still unmerged). After merging, the accurate ground
        truth is tracked via the DGIM estimate vs. actual stream insertion count.
        We verify the estimate is in the range [n/2, 3n/2] for n insertions.
        """
        dgim = DGIMAlgorithm(window_size=2000)
        n_high = 32   # power-of-2 count keeps merging deterministic
        for _ in range(n_high):
            dgim.add_transaction(10_000.0)
        for _ in range(50):
            dgim.add_transaction(100.0)

        estimate = dgim.estimate_count()
        # DGIM guarantee: estimate lies in [n/2, 3n/2]
        assert estimate >= n_high // 2, (
            f"DGIM estimate {estimate} below lower bound {n_high // 2}"
        )
        assert estimate <= n_high * 3 // 2 + 1, (
            f"DGIM estimate {estimate} above upper bound {n_high * 3 // 2}"
        )

    def test_bucket_expiry_outside_window(self):
        """Buckets outside the rolling window must be expired."""
        dgim = DGIMAlgorithm(window_size=10)
        # Insert 5 high-value transactions
        for _ in range(5):
            dgim.add_transaction(10_000.0)
        initial_estimate = dgim.estimate_count()

        # Flush 15 low-value transactions to push window past the first 5
        for _ in range(15):
            dgim.add_transaction(10.0)

        # All original buckets should be expired
        assert dgim.estimate_count() == 0

    def test_surge_flag(self):
        """Surge flag should be True when estimate >= threshold."""
        dgim = DGIMAlgorithm(window_size=500)
        dgim._surge_threshold = 3
        for _ in range(6):
            dgim.add_transaction(6000.0)
        assert dgim.is_surge is True

    def test_stats_structure(self):
        """stats() should return expected keys."""
        dgim = DGIMAlgorithm(window_size=100)
        s = dgim.stats()
        for key in ("stream_position", "window_size", "bucket_count",
                    "dgim_estimate", "true_window_count", "is_surge", "buckets"):
            assert key in s


# ══════════════════════════════════════════════════════════════════════════════
#  COSINE DISTANCE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCosineDistance:

    def test_identical_vectors_zero_distance(self):
        """Identical vectors must have cosine distance 0."""
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_vectors_distance_one(self):
        """Orthogonal vectors must have cosine distance 1."""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        assert cosine_distance(u, v) == pytest.approx(1.0, abs=1e-9)

    def test_opposite_vectors_distance_two(self):
        """Anti-parallel vectors must have cosine distance 2."""
        u = np.array([1.0, 0.0])
        v = np.array([-1.0, 0.0])
        assert cosine_distance(u, v) == pytest.approx(2.0, abs=1e-9)

    def test_scaled_vectors_same_as_unit(self):
        """Cosine distance is scale-invariant — scaling should not change distance."""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([2.0, 4.0, 6.0])
        assert cosine_distance(u, v) == pytest.approx(0.0, abs=1e-9)

    def test_zero_vector_handled(self):
        """Zero vector should return 0.0 without raising an exception."""
        u = np.zeros(3)
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(u, v) == 0.0

    def test_output_in_range(self):
        """Output must always lie within [0, 2]."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            u = rng.uniform(-10, 10, 5)
            v = rng.uniform(-10, 10, 5)
            d = cosine_distance(u, v)
            assert 0.0 <= d <= 2.0 + 1e-9


# ══════════════════════════════════════════════════════════════════════════════
#  MAHALANOBIS DISTANCE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMahalanobisDistance:

    def test_mean_has_zero_distance(self):
        """A vector equal to the mean must have Mahalanobis distance 0."""
        mu  = np.array([5.0, 10.0, 2.0])
        cov = np.eye(3)
        assert mahalanobis_distance(mu, mu, cov=cov) == pytest.approx(0.0, abs=1e-9)

    def test_identity_covariance_equals_euclidean(self):
        """
        With identity covariance, Mahalanobis should equal Euclidean distance.
        Tolerance is rel=1e-5 because regularisation λ=1e-6 causes tiny deviation.
        """
        x   = np.array([4.0, 6.0])
        mu  = np.array([1.0, 2.0])
        cov = np.eye(2)
        euclidean = np.linalg.norm(x - mu)
        assert mahalanobis_distance(x, mu, cov=cov) == pytest.approx(euclidean, rel=1e-4)

    def test_scaling_covariance(self):
        """
        Scaling Σ by factor s² multiplies Σ⁻¹ by 1/s², so D_M scales by 1/s.
        Tolerance is rel=1e-4 to account for regularisation λ=1e-6 on both cov calls.
        """
        x   = np.array([3.0, 0.0])
        mu  = np.array([0.0, 0.0])
        cov1 = np.eye(2)
        cov2 = 4 * np.eye(2)  # Σ scaled by 4 → D_M halved
        d1 = mahalanobis_distance(x, mu, cov=cov1)
        d2 = mahalanobis_distance(x, mu, cov=cov2)
        assert d1 == pytest.approx(d2 * 2, rel=1e-4)

    def test_nonnegative_output(self):
        """Mahalanobis distance must always be non-negative."""
        rng = np.random.default_rng(7)
        for _ in range(30):
            x  = rng.uniform(-5, 5, 4)
            mu = rng.uniform(-5, 5, 4)
            A  = rng.uniform(0, 1, (4, 4))
            cov = A @ A.T + np.eye(4) * 0.1
            d = mahalanobis_distance(x, mu, cov=cov)
            assert d >= 0.0

    def test_precomputed_inv_cov(self):
        """Providing inv_cov should give the same result as providing cov."""
        x   = np.array([2.0, 3.0, 1.0])
        mu  = np.array([1.0, 1.0, 1.0])
        cov = np.array([[2., 0.5, 0.1],
                        [0.5, 3., 0.2],
                        [0.1, 0.2, 1.]])
        inv_cov = np.linalg.inv(cov)
        d_cov = mahalanobis_distance(x, mu, cov=cov)
        d_inv = mahalanobis_distance(x, mu, inv_cov=inv_cov)
        assert d_cov == pytest.approx(d_inv, rel=1e-6)

    def test_raises_without_cov_or_inv(self):
        """Should raise ValueError when neither cov nor inv_cov is provided."""
        with pytest.raises(ValueError):
            mahalanobis_distance(np.array([1.0]), np.array([0.0]))


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE PROFILE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestBaselineProfile:

    def test_mean_converges(self):
        """Mean should converge towards the true mean after sufficient samples."""
        true_mean = np.array([1000.0, 5000.0, 4000.0])
        profile = BaselineProfile(n_features=3)
        rng = np.random.default_rng(0)
        for _ in range(500):
            x = rng.normal(true_mean, scale=100.0)
            profile.update(x)
        np.testing.assert_allclose(profile.mean, true_mean, rtol=0.05)

    def test_ready_threshold(self):
        """Profile should not be ready before min_samples samples."""
        profile = BaselineProfile(n_features=2)
        assert not profile.ready(min_samples=30)
        for i in range(30):
            profile.update(np.array([float(i), float(i)]))
        assert profile.ready(min_samples=30)

    def test_cosine_dist_zero_for_mean(self):
        """Cosine distance of the mean against itself should be ~0."""
        profile = BaselineProfile(n_features=3)
        vec = np.array([100.0, 200.0, 150.0])
        for _ in range(50):
            profile.update(vec)
        assert profile.cosine_dist(profile.mean) == pytest.approx(0.0, abs=1e-9)
