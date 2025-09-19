import numpy as np
import pandas as pd
import pytest

from utils_cg import ensure_unique_key, reindex_5m_ffill_limit
from qa_p1_5m import evaluate_qa


def test_expected_count_80d():
    idx = pd.date_range("2024-01-01", periods=80*24*12, freq="5min", tz="UTC")
    assert len(idx) == 23040


def test_right_closed_grid():
    idx = pd.date_range("2024-01-01", periods=1000, freq="5min", tz="UTC")
    # Right-closed 5m grid aligns at minutes % 5 == 0
    assert all(ts.minute % 5 == 0 for ts in idx)


def test_ffill_limits():
    base = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "ts": [base[0], base[4], base[8]],
        "x": [1.0, np.nan, 2.0],
    })
    out = reindex_5m_ffill_limit(df.rename(columns={"x": "funding_now"}), "funding_now", limit=3)
    flags = out["funding_now_imputed"].astype(int).values
    # First block: indices 1..3 should be ffilled (flags 1), index 4 is original NaN → remains NaN (flag 0)
    assert (flags[1:4] == 1).all()
    assert flags[4] == 0


def test_unique_key():
    idx = pd.to_datetime(["2024-01-01T00:05:00Z", "2024-01-01T00:05:00Z"])  # duplicate ts
    df = pd.DataFrame({"symbol": ["BTCUSDT", "BTCUSDT"], "ts": idx})
    with pytest.raises(ValueError):
        ensure_unique_key(df, ["symbol", "ts"])


def test_acceptance_gate():
    # Build a perfect 1-day dataset (288 bars) to pass gate for testing logic
    idx = pd.date_range("2024-01-01", periods=288, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "ts": idx,
        "symbol": ["BTCUSDT"] * len(idx),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1.0,
        "funding_now": 0.0,
        "oi_now": 1000.0,
        "_imputed_funding_now": 0,
        "_imputed_oi_now": 0,
    })
    report, fail = evaluate_qa(df, horizon_days=1)
    assert report["expected_bars_80d"] == 1 * 24 * 12
    assert fail is False
