import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from label_p3 import TBParams, _label_symbol
from label_validate import app as validate_cli


def _make_ohlc(n=500, start="2024-01-01", freq="5min"):
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


def test_schema_and_keys(tmp_path):
    # Build synthetic features and labels
    feat = _make_ohlc(600)
    # labels from our generator with small H
    params = TBParams(k=1.2, H=10, atr_window=5)
    labels = _label_symbol(feat, params)
    # Persist
    froot = tmp_path / "data" / "features" / "5m" / "BTCUSDT" / "y=2024" / "m=01" / "d=01"
    froot.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(feat, preserve_index=False), froot / "part-20240101.parquet")
    lroot = tmp_path / "data" / "labels" / "5m" / "BTCUSDT" / "y=2024" / "m=01" / "d=01"
    lroot.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(labels, preserve_index=False), lroot / "part-20240101.parquet")

    runner = CliRunner()
    out = tmp_path / "reports" / "p3_validate.json"
    r = runner.invoke(
        validate_cli,
        [
            "run",
            "--labels",
            str(lroot / "part-20240101.parquet"),
            "--features",
            str(froot / "part-20240101.parquet"),
            "--k",
            "1.2",
            "--H",
            "10",
            "--atr-window",
            "5",
            "--days",
            "80",
            "--out-json",
            str(out),
        ],
    )
    # Gaps may be large due to 80d assumption on small data, but validator still runs.
    assert r.exit_code in (0, 1)
    assert out.exists()
    with open(out) as f:
        rep = json.load(f)
    # Required fields present
    for k in [
        "expected_usable_bars_50d",
        "present_labels",
        "gap_ratio_after_warmup",
        "dup_key_count",
        "class_histogram",
        "sample_rulecheck_mismatch_ratio",
        "join_count",
        "pass",
        "violations",
    ]:
        assert k in rep

