# 🛡️ SentinelPay — Real-Time Financial Fraud & Anomaly Intelligence Platform

> **University of Mumbai · Big Data Analytics · Semester 7 Project**
> Stream mining algorithms and statistical distance metrics operating on live financial transaction logs.

---

## 📸 What It Looks Like

A dark-mode, enterprise-grade Security Operations Portal built with Streamlit — inspired by Datadog, Stripe Radar, and Palantir Foundry. Every algorithm from the BDA syllabus runs in real time as transactions stream from the PaySim dataset.

---

## 🚀 Quick Start

### Windows (PowerShell)
```powershell
cd "BDA project"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### Linux / macOS / Git Bash
```bash
chmod +x run.sh && ./run.sh
```

Then open **http://localhost:8501** in your browser.

---

## 🗂️ Project Structure

```
BDA project/
├── app.py                         ← Streamlit dashboard (main entry)
├── run.sh                         ← One-click bash launcher
├── requirements.txt               ← Python dependencies
├── README.md                      ← This file
├── guide.md                       ← Examiner viva cheat sheet
├── .streamlit/
│   └── config.toml                ← Dark theme + server config
├── src/
│   ├── __init__.py
│   ├── algorithms.py              ← Bloom Filter, DGIM, Cosine, Mahalanobis
│   └── pipeline.py                ← CSV streaming & feature extraction
├── tests/
│   ├── __init__.py
│   └── test_algorithms.py         ← 32 unit tests (pytest)
└── data/
    ├── PS_20174392719_...csv      ← PaySim dataset (6.3M transactions)
    └── blacklisted_ips.json       ← Bloom Filter seed blacklist
```

---

## 🧠 Algorithms Implemented

| Algorithm | BDA Module | Purpose in SentinelPay |
|---|---|---|
| **Bloom Filter** | Module 4 — Stream Mining | Check if source IP / account is blacklisted in O(k) time with sub-MB memory |
| **DGIM Algorithm** | Module 4 — Stream Mining | Count high-value transactions (>$5,000) in a rolling window using O(log² N) space |
| **Cosine Distance** | Module 5 — Distance Measures | Detect angular deviation from historical spending pattern |
| **Mahalanobis Distance** | Module 5 — Distance Measures | Correlation-aware multivariate anomaly scoring |

### Bloom Filter
```
P_fp ≈ (1 − e^(−kn/m))^k
Optimal m = −n·ln(ε) / (ln 2)²
Optimal k = (m/n)·ln 2
```
A bit-array of size **m** with **k** hash functions. Guarantees **zero false negatives** — any item
not inserted will always return False. Used here to check incoming IPs and account IDs against
a pre-loaded blacklist in constant O(k) time.

### DGIM Algorithm
```
Space: O(log² N)    Error: ≤ 50%
Rule:  ≤ 2 buckets per size 2^i
```
Tracks the count of high-value transactions (Amount > $5,000) in the last N transactions without
storing the full window. Buckets of size 1, 2, 4, 8… merge when a third bucket of the same size appears.

### Cosine Distance
```
d(u, v) = 1 − (u · v) / (‖u‖₂ · ‖v‖₂)     Range: [0, 2]
```
Compares the **direction** of the current transaction vector `[amount, oldBalance, newBalance]`
against the running historical mean. Scale-invariant — a large but normal transaction won't be flagged.

### Mahalanobis Distance
```
D_M(x) = √[ (x − μ)ᵀ Σ⁻¹ (x − μ) ]
```
Accounts for correlations between features via the precision matrix Σ⁻¹. Unlike Euclidean distance,
it isn't fooled by naturally correlated financial features. Under MVN, D_M² ~ χ²(d).

---

## 🖥️ Dashboard Features

### Header Bar
Live clock, throughput (tx/s), buffer size, and LIVE/IDLE badge.

### KPI Metrics Ribbon (6 cards)
| Card | Metric |
|---|---|
| Total Streamed | Running count of processed transactions |
| Flagged Frauds | Count + percentage flagged by any algorithm |
| Blacklist Size | Entries loaded into Bloom Filter |
| DGIM HV Estimate | Rolling count of high-value transactions |
| Bloom Fill Ratio | Fraction of bits set (+ bit-array size m) |
| DGIM Buckets | Active bucket count + surge indicator |

### Live Stream Table
Last 60 transactions with Time, Type, Amount, Account, IP, Cosine Δ, Mahal Δ, and Status tag.

### Deep-Dive Analytics Tabs

**Tab 1 — Algorithm Inspection**
- Bloom Filter bit-array density bar chart
- DGIM bucket tree scatter plot (bubble size = log₂ of bucket size)
- Metric cards for both algorithms

**Tab 2 — Distance Scatter Plot**
- 2D scatter: Cosine Distance (x) vs Mahalanobis Distance (y)
- Green = normal, Red = labelled fraud
- Amber dashed lines show the configurable decision boundaries

**Tab 3 — Fraud Simulation Panel**
- Enter any transaction parameters manually
- Inject it live through all four algorithms
- Per-algorithm verdict cards + overall FLAGGED / CLEARED banner

### Sidebar Controls
- Max rows to stream (500–10,000)
- Batch size per refresh (10–200)
- DGIM window size N (100–5,000)
- Cosine distance threshold τ
- Mahalanobis distance threshold τ
- Init / Stream / Pause / Reset buttons

---

## 💼 Real-Life Applications

### 1. Payment Processor Fraud Screening (Stripe / Razorpay / PayU)
Every card transaction can be scored in < 1 ms using the Bloom Filter (IP blacklist check) and
Mahalanobis distance (deviation from cardholder's historical pattern). No database round-trip needed
for the blacklist check — the Bloom Filter lives entirely in memory at ~96 KB for 10,000 entries.

### 2. Anti-Money Laundering (AML) in Core Banking
DGIM tracks the frequency of large-value transfers in a rolling 24-hour window without storing every
transaction. Compliance teams get a real-time surge alert when high-value activity spikes above
normal levels — catching structuring (breaking a large transfer into smaller ones) patterns early.

### 3. Real-Time ATM / POS Terminal Monitoring
Cosine distance flags when a card's spending *pattern* suddenly shifts direction — e.g., an account
that normally shows small grocery payments suddenly initiating large international transfers. The
scale-invariance of Cosine distance means legitimate high-value customers don't generate false alerts.

### 4. Cryptocurrency Exchange Compliance (CeFi)
Wallet addresses and transaction originator IPs are pre-loaded into the Bloom Filter from OFAC /
FinCEN sanction lists. Every deposit/withdrawal is checked in O(k) = O(6) hash computations —
sub-microsecond — enabling compliance at exchange-scale throughput (millions of tx/day).

### 5. Insurance Claim Fraud Detection
Mahalanobis distance on claim features (amount, days-since-policy, claim-frequency) identifies
multivariate outliers that bypass simple threshold rules. Because it accounts for feature
correlations (large claims are expected on expensive policies), it has far fewer false positives
than univariate anomaly rules.

### 6. Telecom Revenue Assurance
DGIM can track high-volume call events (potential toll fraud / SIM boxing) in a sliding window
of the last N call-detail records, using O(log² N) memory regardless of call volume — critical
for telcos processing billions of CDRs per day.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | ≥1.32 | Dashboard UI |
| pandas | ≥2.0 | CSV chunked streaming |
| numpy | ≥1.24 | Numerical computations |
| scipy | ≥1.11 | Statistical utilities |
| plotly | ≥5.18 | Interactive charts |
| mmh3 | ≥4.0 | MurmurHash3 for Bloom Filter |
| pytest | ≥7.4 | Unit test runner |

---

## 🧪 Running Tests

```bash
# Run all 32 tests
.venv\Scripts\python -m pytest tests/ -v

# Run with coverage summary
.venv\Scripts\python -m pytest tests/ -v --tb=short
```

Expected output: **32 passed** in < 1 second.

---

## 📖 References

- Leskovec, Rajaraman, Ullman — *Mining of Massive Datasets*, 3rd Ed.
  - Chapter 4.2: DGIM Algorithm
  - Chapter 4.3: Bloom Filters
  - Chapter 3.5: Distance Measures
- Lopez-Rojas et al. — *PaySim: A Financial Mobile Money Simulator* (2016)
- Kirsch & Mitzenmacher — *Less Hashing, Same Performance* (2008) — double hashing for Bloom Filters
- Welford, B.P. — *Note on a method for calculating corrected sums of squares* (1962)

---

*SentinelPay · University of Mumbai BDA Semester 7 · PaySim Dataset · Python 3.13*
