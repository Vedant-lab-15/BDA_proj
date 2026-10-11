# 🛡️ SentinelPay — Examiner Quick-Reference Guide
**University of Mumbai · BDA Semester 7 · Project Viva Cheat Sheet**

---

## 1. Code Navigation Table

| BDA Syllabus Topic | File | Class / Function | Line Range | Module |
|---|---|---|---|---|
| **Bloom Filter** — bit-array, hash functions, FP rate formula | `src/algorithms.py` | `class BloomFilter` | 37 – 167 | Module 4 |
| Bloom Filter — `add()` method (set k bits) | `src/algorithms.py` | `BloomFilter.add` | 107 – 110 | Module 4 |
| Bloom Filter — `contains()` method (query) | `src/algorithms.py` | `BloomFilter.contains` | 112 – 118 | Module 4 |
| Bloom Filter — FP rate formula `P_fp ≈ (1−e^(−kn/m))^k` | `src/algorithms.py` | `BloomFilter.current_fp_rate` | 121 – 129 | Module 4 |
| Bloom Filter — double hashing `h_i = (h1 + i·h2) mod m` | `src/algorithms.py` | `BloomFilter._hashes` | 79 – 100 | Module 4 |
| Bloom Filter — optimal m & k formulae | `src/algorithms.py` | `BloomFilter._optimal_m`, `_optimal_k` | 68 – 72 | Module 4 |
| **DGIM Algorithm** — bucket management, merge rule | `src/algorithms.py` | `class DGIMAlgorithm` | 172 – 284 | Module 4 |
| DGIM — `add_transaction()` (bit insertion, bucket creation) | `src/algorithms.py` | `DGIMAlgorithm.add_transaction` | 230 – 247 | Module 4 |
| DGIM — merge invariant enforcement (≤ 2 per size) | `src/algorithms.py` | `DGIMAlgorithm._merge_if_needed` | 210 – 228 | Module 4 |
| DGIM — `estimate_count()` (half-oldest-bucket rule) | `src/algorithms.py` | `DGIMAlgorithm.estimate_count` | 249 – 260 | Module 4 |
| DGIM — bucket expiry (sliding window) | `src/algorithms.py` | `DGIMAlgorithm._expire_old_buckets` | 204 – 208 | Module 4 |
| **Cosine Distance** — formula, implementation | `src/algorithms.py` | `cosine_distance()` | 292 – 328 | Module 5 |
| **Mahalanobis Distance** — Σ⁻¹, regularisation | `src/algorithms.py` | `mahalanobis_distance()` | 331 – 393 | Module 5 |
| Baseline Profile — Welford online mean/covariance | `src/algorithms.py` | `class BaselineProfile` | 396 – 444 | Module 5 |
| Data Ingestion — chunked CSV streaming | `src/pipeline.py` | `class TransactionStream` | 62 – 145 | Infra |
| Feature vector extraction [amount, oldBal, newBal] | `src/pipeline.py` | `extract_feature_vector()` | 148 – 158 | Infra |
| Fraud Simulation — synthetic transaction builder | `src/pipeline.py` | `build_synthetic_transaction()` | 162 – 192 | Infra |
| Transaction processing (all 4 algorithms in one pass) | `app.py` | `process_transaction()` | ~195 – 240 | App |
| Bloom Filter unit tests | `tests/test_algorithms.py` | `class TestBloomFilter` | 25 – 100 | Tests |
| DGIM unit tests | `tests/test_algorithms.py` | `class TestDGIMAlgorithm` | 103 – 175 | Tests |
| Cosine Distance unit tests | `tests/test_algorithms.py` | `class TestCosineDistance` | 178 – 220 | Tests |
| Mahalanobis unit tests | `tests/test_algorithms.py` | `class TestMahalanobisDistance` | 223 – 280 | Tests |

---

## 2. Core Algorithm Summaries

### 🔵 Bloom Filter *(Module 4 — Stream Mining)*

A **Bloom Filter** is a space-efficient, probabilistic set-membership data structure using a bit-array of size *m* and *k* independent hash functions. When an element is inserted, all *k* bit positions (computed via double-hashing `hᵢ(x) = (h₁(x) + i·h₂(x)) mod m`) are set to 1. A query returns **True** only if all *k* bits are 1 — this guarantees **zero false negatives** but allows false positives with probability `P_fp ≈ (1 − e^(−kn/m))^k`. Optimal parameters are derived analytically: `m = −n·ln(ε) / (ln 2)²` and `k = (m/n)·ln 2`.

**In SentinelPay:** Used to check whether an incoming transaction's IP address or account ID appears on the blacklist. The filter is pre-seeded from `data/blacklisted_ips.json`. A match triggers the `⛔ BLACKLISTED` alert tag.

---

### 🟡 DGIM Algorithm *(Module 4 — Stream Mining)*

The **Datar-Ganti-Indyk-Motwani (DGIM)** algorithm counts the number of 1-bits (high-value transactions > $5,000) within a sliding window of *N* most recent events, using only **O(log² N) space**. Timestamps are stored in buckets whose sizes are powers of 2 (1, 2, 4, 8, …). The invariant is: **at most 2 buckets of the same size may coexist** — when a third appears, the two oldest are merged into one bucket of double the size. The count estimate subtracts half the oldest bucket to correct for partial-window overlap, guaranteeing an error of **≤ 50%**.

**In SentinelPay:** Tracks high-value transaction frequency in a configurable rolling window. A DGIM estimate exceeding the surge threshold triggers the `⚠️ DGIM SURGE` alert and KPI badge.

---

### 🟢 Cosine Distance *(Module 5 — Distance Measures)*

**Cosine Distance** measures the angular divergence between two vectors in feature space:
`d(u, v) = 1 − (u · v) / (‖u‖₂ · ‖v‖₂)`.
It ranges from **0** (identical direction) to **2** (opposite direction) and is **scale-invariant** — a transaction ten times larger in all features is not penalised if its *pattern* matches the baseline. The baseline profile vector `v` is maintained using Welford's online algorithm, which updates the mean in O(1) per sample without storing history.

**In SentinelPay:** Each transaction is represented as a 3-D vector `[amount, oldBalance, newBalance]` and compared against the running historical mean. Cosine distance > τ (default 0.35) triggers the `🚨 ANOMALY` flag.

---

### 🔴 Mahalanobis Distance *(Module 5 — Distance Measures)*

**Mahalanobis Distance** is a correlation-aware, scale-invariant anomaly metric:
`D_M(x) = √[ (x − μ)ᵀ Σ⁻¹ (x − μ) ]`.
Unlike Euclidean distance, it accounts for feature correlations through the **precision matrix Σ⁻¹** (inverse covariance). Under a multivariate normal distribution, `D_M² ~ χ²(d)`, so a 3-sigma decision boundary corresponds to `D_M ≈ 3.7` for `d = 3` features. A regularisation term `λI` (λ = 1e-6) is added to Σ before inversion to handle near-singular covariance matrices.

**In SentinelPay:** Computed against the same Welford online covariance matrix as Cosine Distance. Mahalanobis distance > τ (default 4.5) triggers the `🚨 ANOMALY` flag. The scatter plot in Tab 2 shows both metrics with the decision boundary overlaid.

---

## 3. Top Examiner Questions & Direct Answers

### Q1: What is the false-positive rate formula for a Bloom Filter, and what does each variable represent?

**Answer:**
`P_fp ≈ (1 − e^(−kn/m))^k`

| Symbol | Meaning |
|--------|---------|
| `k` | Number of hash functions |
| `n` | Number of elements inserted |
| `m` | Size of the bit array (in bits) |
| `e` | Euler's number (~2.718) |

As `n` grows (more elements inserted), more bits are set to 1, so `P_fp` increases. The optimal `k` that minimises `P_fp` for given `m` and `n` is `k = (m/n) ln 2`. In our implementation with `capacity=10,000` and `fp_rate=0.01`, this gives `m ≈ 95,851 bits` and `k = 6 hash functions`.

---

### Q2: Explain the DGIM bucket merging rule with a concrete mathematical example.

**Answer:**
The invariant is: **at most 2 buckets of any size 2^i** may exist simultaneously.

**Example:** Suppose we have buckets of size `[1, 1, 1]` (three size-1 buckets). The two **oldest** ones merge:
- Two size-1 buckets → one size-2 bucket
- New state: `[1, 2]` ✓

If we now add another size-1 bucket: `[1, 1, 2]` — no merge needed.
Add yet another size-1 bucket: `[1, 1, 1, 2]` → merge two oldest size-1s:
`[1, 2, 2]` — now two size-2s, check again → merge: `[1, 4]` ✓

The merged bucket inherits the **newer** of the two timestamps (conservative estimate). The estimate formula is:
`Estimate = (Σ all bucket sizes) − floor(oldest_size / 2)`

The **−floor(oldest_size/2)** correction accounts for the oldest bucket potentially being half outside the window, bounding the error to ≤ 50% of the true count.

---

### Q3: Why does Mahalanobis Distance outperform Euclidean Distance for fraud detection?

**Answer:**
Euclidean distance treats all features as **independent and equally scaled**. In financial data, `amount`, `oldBalance`, and `newBalance` are strongly correlated (a large transfer reduces `oldBalance` by roughly the `amount`). Euclidean distance would flag normal large-balance transfers as anomalies simply due to magnitude.

Mahalanobis distance **normalises by the covariance matrix Σ**, effectively rotating and scaling the feature space so that the natural spread of data maps to unit variance in every direction. This means only transactions that deviate from the *correlated pattern* — not just from the scale — are flagged.

Formally: `D_Euclidean(x, μ) = √[(x−μ)ᵀ I (x−μ)]` vs `D_M(x, μ) = √[(x−μ)ᵀ Σ⁻¹ (x−μ)]`. When `Σ = I` (identity), they are identical — Euclidean is a special case of Mahalanobis.

---

### Q4: Can a Bloom Filter produce false negatives? Why or why not?

**Answer:**
**No.** A Bloom Filter can **never produce false negatives**.

When an element is inserted, all *k* hash positions are set to **1**. When the same element is queried, the same *k* positions are checked — they will always be 1 because bits are **never cleared**. Therefore, any element that was added will always return True on query.

False *positives* occur because hash collisions can incidentally set a queried element's bit positions to 1 even without it being inserted. This is the fundamental asymmetry: **False Negatives = 0**, **False Positives > 0** with probability `P_fp`.

---

### Q5: What is the space complexity of DGIM, and why is O(log² N) better than O(N)?

**Answer:**
- **Naïve approach:** Store all N bits in the window → **O(N)** space. For N = 1,000,000 stream positions, that's ~1 MB just for the window.
- **DGIM:** Stores at most `2 × log₂(N)` buckets (at most 2 per power-of-2 size, and sizes range from 1 to N) → **O(log N)** buckets, each storing a timestamp of O(log N) bits → **O(log² N)** total bits.

For N = 10^6: log₂(10^6) ≈ 20 → at most 40 buckets vs 10^6 bits. That is a **25,000× reduction** in space with only ≤50% error in the count estimate.

---

### Q6: What is Cosine Similarity vs Cosine Distance, and when would distance = 2?

**Answer:**
- **Cosine Similarity:** `sim(u, v) = (u · v) / (‖u‖ · ‖v‖)` — ranges from **−1** to **+1**
- **Cosine Distance:** `d(u, v) = 1 − sim(u, v)` — ranges from **0** to **2**

Distance = 2 occurs when `sim = −1`, i.e., the vectors point in **exactly opposite directions** (anti-parallel). In the fraud context, this would mean the current transaction vector is the mirror-opposite of the typical spending pattern — e.g., an account that normally has high balances and small amounts suddenly shows zero balance and a massive outgoing transfer.

---

### Q7: How does the Welford online algorithm update mean and covariance without storing all past data?

**Answer:**
Welford's algorithm maintains running statistics with **O(1) memory per update**:

```
For each new sample x (n-th sample):
  delta  = x − mean_prev
  mean   += delta / n
  delta2 = x − mean_new          # note: uses updated mean
  M2     += outer(delta, delta2)  # sum of squared deviations
  
Covariance Σ = M2 / (n − 1)      # unbiased sample covariance
```

This is numerically stable (unlike the naïve sum-of-squares formula which suffers from catastrophic cancellation) and streams perfectly — no historical data is retained after each update.

---

### Q8: What is the PaySim dataset, and why is it appropriate for this project?

**Answer:**
**PaySim** is a synthetic financial transaction dataset generated by an agent-based simulation model, published on Kaggle. It contains **6.3 million transactions** over 30 days of simulated mobile money operations with 5 transaction types: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN. The dataset includes ground-truth fraud labels (`isFraud`) making it ideal for:
1. **Streaming simulation** — the `step` column represents sequential time steps
2. **Fraud rate realism** — fraud rate is ~0.13%, matching real-world class imbalance
3. **Algorithm validation** — known fraud labels allow measuring Bloom/Mahalanobis effectiveness

In SentinelPay, the CSV is read using `pandas.read_csv(chunksize=500)` to simulate a real-time stream without loading the entire 500MB file into memory.

---

## 4. Algorithm Complexity Summary

| Algorithm | Time (query) | Time (insert) | Space | Error Guarantee |
|---|---|---|---|---|
| Bloom Filter | O(k) | O(k) | O(m) bits | FP: `(1−e^(−kn/m))^k` · FN: 0 |
| DGIM | O(log N) | O(log² N) | O(log² N) bits | Count error ≤ 50% |
| Cosine Distance | O(d) | — | O(d) | Exact (floating point) |
| Mahalanobis Distance | O(d²) | — | O(d²) for Σ⁻¹ | Exact (floating point) |

*d = feature dimensionality (3 in this project), N = window size, k = hash functions, m = bit array size*

---

## 5. Project File Structure

```
BDA project/
├── app.py                    ← Streamlit frontend (main entry point)
├── run.sh                    ← One-click launcher (bash)
├── requirements.txt          ← Python dependencies
├── guide.md                  ← This file
├── src/
│   ├── __init__.py
│   ├── algorithms.py         ← Bloom Filter, DGIM, Cosine, Mahalanobis
│   └── pipeline.py           ← CSV streaming, feature extraction
├── tests/
│   ├── __init__.py
│   └── test_algorithms.py    ← 25+ unit tests (pytest)
└── data/
    ├── PS_20174392719_...csv  ← PaySim dataset (6.3M transactions)
    └── blacklisted_ips.json  ← Bloom Filter seed data
```

---

## 6. Quick Launch Reference

```bash
# Linux / macOS / Git Bash (Windows)
chmod +x run.sh && ./run.sh

# Windows (PowerShell — manual steps)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest tests/ -v
streamlit run app.py

# Run tests only
pytest tests/ -v --tb=short

# Run app directly
streamlit run app.py --server.port 8501
```

---

*SentinelPay · University of Mumbai BDA Semester 7 · Algorithms: Bloom Filter · DGIM · Cosine Distance · Mahalanobis Distance*
