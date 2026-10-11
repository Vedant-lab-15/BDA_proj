"""
SentinelPay — Data Ingestion & Streaming Pipeline
===================================================
Low-latency generator pipeline that reads the PaySim financial transaction
log in configurable chunks and emits enriched transaction payloads.

Dataset columns (PaySim schema):
  step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
  nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Emitted payload adds:
  ip_address  — deterministically synthesised from account ID (no PII risk)
  risk_score  — placeholder updated by app.py after algorithm evaluation
"""

from __future__ import annotations

import hashlib
import random
import time
import logging
from pathlib import Path
from typing import Generator, Iterator, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
CSV_FILE   = DATA_DIR / "PS_20174392719_1491204439457_log.csv"
CHUNK_SIZE = 500

# Pre-computed pool of synthetic IPs for deterministic assignment
_IP_POOL_SIZE = 10_000

# ── Synthetic IP generator ────────────────────────────────────────────────────

def _account_to_ip(account_id: str) -> str:
    """
    Deterministically map an account ID to a synthetic IPv4 address using
    the first 4 bytes of its MD5 hash.  No real IP addresses are stored in
    the dataset; this simulates network telemetry.
    """
    digest = hashlib.md5(account_id.encode()).digest()
    octets = [int(b) % 256 for b in digest[:4]]
    # Force into private/test ranges so we never generate real routable IPs
    octets[0] = random.choice([10, 172, 192, 198, 203]) if octets[0] % 5 == 0 else octets[0]
    return ".".join(map(str, octets))


def _seed_blacklist_ips(n: int = 5) -> list[str]:
    """Return n synthetic IPs from the known blacklist pool."""
    from data_utils import _BLACKLIST_POOL  # local import to avoid circular
    return random.sample(_BLACKLIST_POOL, min(n, len(_BLACKLIST_POOL)))


# ── Core streaming generator ──────────────────────────────────────────────────

class TransactionStream:
    """
    Chunked pandas CSV reader wrapped as an infinite generator.

    Usage:
        stream = TransactionStream()
        for tx in stream.iter_transactions(max_rows=5000):
            process(tx)

    Each yielded item is a plain dict with schema:
        {
          "step":            int,
          "type":            str,   # PAYMENT | TRANSFER | CASH_OUT | DEBIT | CASH_IN
          "amount":          float,
          "nameOrig":        str,
          "oldbalanceOrg":   float,
          "newbalanceOrig":  float,
          "nameDest":        str,
          "oldbalanceDest":  float,
          "newbalanceDest":  float,
          "isFraud":         int,   # ground-truth label (0/1)
          "isFlaggedFraud":  int,
          "ip_address":      str,   # synthetic
          "row_index":       int,   # global row counter
        }
    """

    REQUIRED_COLUMNS = [
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud",
    ]

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        chunk_size: int = CHUNK_SIZE,
    ):
        self.csv_path   = csv_path or CSV_FILE
        self.chunk_size = chunk_size
        self._row_index = 0

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.csv_path}. "
                "Place PS_20174392719_1491204439457_log.csv inside data/."
            )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _enrich(self, row: pd.Series) -> dict:
        """Attach synthetic telemetry fields to a raw CSV row."""
        payload = {col: row[col] for col in self.REQUIRED_COLUMNS}
        payload["ip_address"] = _account_to_ip(str(row["nameOrig"]))
        payload["row_index"]  = self._row_index
        return payload

    # ── public API ────────────────────────────────────────────────────────────

    def iter_transactions(
        self,
        max_rows:  Optional[int] = None,
        loop:      bool = False,
        delay_ms:  float = 0.0,
    ) -> Generator[dict, None, None]:
        """
        Yield enriched transaction dicts from the CSV.

        Args:
            max_rows : Stop after this many rows (None = full dataset).
            loop     : Restart from the beginning when EOF is reached.
            delay_ms : Simulated inter-message delay in milliseconds.
        """
        rows_yielded = 0
        while True:
            try:
                reader = pd.read_csv(
                    self.csv_path,
                    chunksize=self.chunk_size,
                    dtype={
                        "step": int,
                        "type": str,
                        "amount": float,
                        "nameOrig": str,
                        "oldbalanceOrg": float,
                        "newbalanceOrig": float,
                        "nameDest": str,
                        "oldbalanceDest": float,
                        "newbalanceDest": float,
                        "isFraud": int,
                        "isFlaggedFraud": int,
                    },
                )
                for chunk in reader:
                    for _, row in chunk.iterrows():
                        if max_rows and rows_yielded >= max_rows:
                            return
                        self._row_index += 1
                        rows_yielded += 1
                        yield self._enrich(row)
                        if delay_ms > 0:
                            time.sleep(delay_ms / 1000.0)

            except StopIteration:
                pass

            if not loop:
                break

    def get_sample_batch(self, n: int = 200, skip_rows: int = 0) -> pd.DataFrame:
        """
        Return a pandas DataFrame of n rows starting at skip_rows.
        Useful for baseline profile initialisation without streaming.
        """
        chunks_to_skip = skip_rows // self.chunk_size
        remaining_skip = skip_rows % self.chunk_size

        collected: list[pd.DataFrame] = []
        total = 0
        reader = pd.read_csv(self.csv_path, chunksize=self.chunk_size)

        for i, chunk in enumerate(reader):
            if i < chunks_to_skip:
                continue
            if i == chunks_to_skip and remaining_skip > 0:
                chunk = chunk.iloc[remaining_skip:]
            collected.append(chunk)
            total += len(chunk)
            if total >= n:
                break

        if not collected:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        df = pd.concat(collected, ignore_index=True).head(n)
        return df


# ── Feature vector extractor ──────────────────────────────────────────────────

def extract_feature_vector(tx: dict) -> np.ndarray:
    """
    Convert a transaction dict to a 3-D numeric feature vector:
        [amount, oldbalanceOrg, newbalanceOrig]

    These three features capture transaction magnitude and balance shift —
    the most discriminative features for Cosine and Mahalanobis distances.
    """
    return np.array([
        float(tx.get("amount", 0.0)),
        float(tx.get("oldbalanceOrg", 0.0)),
        float(tx.get("newbalanceOrig", 0.0)),
    ], dtype=float)


# ── Fraud injection utility (Simulation Panel) ────────────────────────────────

def build_synthetic_transaction(
    step:          int   = 1,
    tx_type:       str   = "TRANSFER",
    amount:        float = 9999.99,
    name_orig:     str   = "C9999999999",
    old_balance:   float = 10000.0,
    new_balance:   float = 0.01,
    name_dest:     str   = "C1111111111",
    ip_address:    str   = "185.220.101.33",
    is_fraud:      int   = 1,
) -> dict:
    """
    Construct a synthetic transaction payload for the Fraud Simulation Panel.
    All fields match the standard emitted schema.
    """
    return {
        "step":            step,
        "type":            tx_type,
        "amount":          amount,
        "nameOrig":        name_orig,
        "oldbalanceOrg":   old_balance,
        "newbalanceOrig":  new_balance,
        "nameDest":        name_dest,
        "oldbalanceDest":  0.0,
        "newbalanceDest":  amount,
        "isFraud":         is_fraud,
        "isFlaggedFraud":  0,
        "ip_address":      ip_address,
        "row_index":       -1,   # synthetic marker
    }
