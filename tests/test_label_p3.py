import numpy as np
import pandas as pd
import pytest

from label_p3 import _compute_atr_pct, _label_symbol, TBParams


def _make_series(n=200, start="2024-01-01", freq="5min"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    price = 100 + np.cumsum(np.random.randn(n) * 0.1)
    high = price + np.random.rand(n) * 0.2
    low = price - np.random.rand(n) * 0.2
    open_ = price + np.random.randn(n) * 0.02
    close = price
    df = pd.DataFrame({
        "ts": idx,
        "symbol": ["BTCUSDT"] * n,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    })
    return df


def test_causal_atr():
    df = _make_series(100)
    g = df.sort_values("ts").reset_index(drop=True)
    atrpct = _compute_atr_pct(g, 14)
    # Check causal: ATR at t uses only TR up to t-1 (shifted). Compare slice by recomputation
    t = 50
    close = g["close"].astype(float)
    prev_close = close.shift(1)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    exp = tr.iloc[:t].tail(14).mean() / prev_close.iloc[t]
    assert np.isfinite(atrpct.iloc[t])
    assert np.isclose(float(atrpct.iloc[t]), float(exp), atol=1e-6)


def test_tie_case_wait():
    # Construct OHLC so first future bar hits both up and down
    n = 60
    df = _make_series(n)
    g = df.copy().sort_values("ts").reset_index(drop=True)
    params = TBParams(k=1.2, H=3, atr_window=3)
    atr = _compute_atr_pct(g, params.atr_window)
    i = 10
    close_i = g.loc[i, "close"]
    up = close_i * (1 + params.k * atr.iloc[i])
    dn = close_i * (1 - params.k * atr.iloc[i])
    # Force bar i+1 to straddle both barriers
    g.loc[i + 1, "open"] = close_i
    g.loc[i + 1, "high"] = up * 1.01
    g.loc[i + 1, "low"] = dn * 0.99
    labels = _label_symbol(g, params)
    lab_i = labels.set_index("ts").loc[g.loc[i, "ts"], "label"]
    assert lab_i == "WAIT"


def test_gap_case_long():
    n = 60
    df = _make_series(n)
    g = df.copy().sort_values("ts").reset_index(drop=True)
    params = TBParams(k=1.2, H=3, atr_window=3)
    atr = _compute_atr_pct(g, params.atr_window)
    i = 10
    close_i = g.loc[i, "close"]
    up = close_i * (1 + params.k * atr.iloc[i])
    g.loc[i + 1, "open"] = up * 1.01  # gap above
    labels = _label_symbol(g, params)
    lab_i = labels.set_index("ts").loc[g.loc[i, "ts"], "label"]
    assert lab_i == "LONG"


def test_timeout_wait():
    n = 60
    df = _make_series(n)
    g = df.copy().sort_values("ts").reset_index(drop=True)
    params = TBParams(k=10.0, H=5, atr_window=3)  # huge k to avoid crosses
    labels = _label_symbol(g, params)
    i = 10
    lab_i = labels.set_index("ts").loc[g.loc[i, "ts"], "label"]
    assert lab_i == "WAIT"


def test_join_keys_unique(tmp_path):
    # Create small dataset and label, ensuring unique keys
    df = _make_series(50)
    params = TBParams(k=1.2, H=3, atr_window=3)
    labels = _label_symbol(df, params)
    # Unique by (ts,symbol)
    dup = labels.duplicated(subset=["symbol", "ts"]).sum()
    assert dup == 0


def test_cli_runs(tmp_path, monkeypatch):
    # Write features to daily Parquet, run CLI, and validate
    import pyarrow as pa
    import pyarrow.parquet as pq
    from label_p3 import app as cli
    from typer.testing import CliRunner

    runner = CliRunner()
    # Prepare features
    df = _make_series(300)
    root = tmp_path / "data" / "features" / "5m" / "BTCUSDT" / "y=2024" / "m=01" / "d=01"
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), root / "part-20240101.parquet")
    # Run labeling
    out_root = tmp_path / "data" / "labels" / "5m" / "BTCUSDT"
    r = runner.invoke(cli, [
        "triple-barrier",
        "--features", str(tmp_path / "data" / "features" / "5m" / "BTCUSDT" / "y=2024" / "m=01" / "d=01" / "part-20240101.parquet"),
        "--out", str(out_root),
        "--tf", "5m",
        "--k", "1.2",
        "--H", "10",
        "--atr_window", "3",
    ])
    assert r.exit_code == 0, r.output

