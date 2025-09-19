import time

import numpy as np
import pandas as pd

from transforms import winsorize_causal, zscore_causal, ffill_with_limit_and_flag
from features_p2 import build_features


def test_causal_zscore_no_leakage():
    idx = pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC")
    s = pd.Series(np.arange(100, dtype=float), index=idx)
    z = zscore_causal(s, window=10)

    # For t=20, compute expected using only past values [t-10, t-1]
    t = 20
    past = s.iloc[t-10:t]
    mean = past.mean()
    std = past.std(ddof=0)
    exp = (s.iloc[t-1] - mean) / (std if std != 0 else 1e-12)
    assert np.isclose(z.iloc[t], exp, atol=1e-9)


def test_winsorize_causal():
    idx = pd.date_range("2024-01-01", periods=50, freq="5min", tz="UTC")
    s = pd.Series(np.linspace(0, 100, 50), index=idx)
    w = winsorize_causal(s, window=10, low=0.1, high=0.9)
    # Current point should be clamped using quantiles computed on previous points
    t = 20
    past = s.iloc[t-10:t]
    ql, qh = np.quantile(past.values, 0.1), np.quantile(past.values, 0.9)
    assert w.iloc[t] >= ql - 1e-9 and w.iloc[t] <= qh + 1e-9


def test_flags_imputed():
    idx = pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC")
    vals = [1.0, np.nan, np.nan, np.nan, np.nan, 2.0] + [np.nan] * 14
    s = pd.Series(vals, index=idx)
    res = ffill_with_limit_and_flag(s, limit=3)
    # Only first three NaNs after 1.0 are imputed (flags=1), the 4th remains NaN (flag=0)
    assert res.flags.iloc[1] == 1 and res.flags.iloc[2] == 1 and res.flags.iloc[3] == 1
    assert pd.isna(res.values.iloc[4]) and res.flags.iloc[4] == 0


def test_no_nan_after_drop():
    # Minimal dataset
    idx = pd.date_range("2024-01-01", periods=200, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "ts": idx,
        "symbol": ["BTCUSDT"] * len(idx),
        "open": 100 + np.random.rand(len(idx)),
        "high": 100 + np.random.rand(len(idx)) + 0.5,
        "low": 100 + np.random.rand(len(idx)) - 0.5,
        "close": 100 + np.random.rand(len(idx)),
        "volume": np.random.randint(100, 1000, len(idx)).astype(float),
        "funding_now": np.random.randn(len(idx)),
        "oi_now": np.abs(np.random.randn(len(idx))) + 1000,
        "cvd_perp_5m": np.random.randn(len(idx)).cumsum(),
        "perp_share_60m": np.random.rand(len(idx)),
        "liq_notional_60m": np.random.rand(len(idx)) * 1e6,
        "rv_5m": np.random.rand(len(idx)),
        "funding_pctile_30d": np.random.rand(len(idx)),
        "oi_pctile_30d": np.random.rand(len(idx)),
    })
    feat = build_features(df, warmup=50)
    assert not feat.isna().any().any()


def test_ms_per_bar_benchmark():
    idx = pd.date_range("2024-01-01", periods=200_000, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "ts": idx,
        "symbol": ["BTCUSDT"] * len(idx),
        "open": 100 + np.random.rand(len(idx)),
        "high": 100 + np.random.rand(len(idx)) + 0.5,
        "low": 100 + np.random.rand(len(idx)) - 0.5,
        "close": 100 + np.random.rand(len(idx)),
        "volume": np.random.randint(100, 1000, len(idx)).astype(float),
        "funding_now": np.random.randn(len(idx)),
        "oi_now": np.abs(np.random.randn(len(idx))) + 1000,
        "cvd_perp_5m": np.random.randn(len(idx)).cumsum(),
        "perp_share_60m": np.random.rand(len(idx)),
        "liq_notional_60m": np.random.rand(len(idx)) * 1e6,
        "rv_5m": np.random.rand(len(idx)),
        "funding_pctile_30d": np.random.rand(len(idx)),
        "oi_pctile_30d": np.random.rand(len(idx)),
    })
    t0 = time.perf_counter()
    feat = build_features(df, warmup=864)
    dt = time.perf_counter() - t0
    ms_per_bar = dt * 1000 / max(1, len(feat))
    assert ms_per_bar <= 50.0

