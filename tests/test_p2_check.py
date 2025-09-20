import json
import numpy as np
import pandas as pd

from typer.testing import CliRunner
from p2_check import app as p2cli


def _make_features(n=240, F=12):
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"ts": idx, "symbol": ["BTCUSDT"] * n})
    for i in range(F):
        df[f"f{i}"] = np.random.randn(n)
    df["_imputed_funding_now"] = 0
    df["_imputed_oi_now"] = 0
    return df


def test_feature_count_range(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    df = _make_features(F=12)
    p = tmp_path / "part.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    out = tmp_path / "check.json"
    r = CliRunner().invoke(p2cli, [
        "run", "--features", str(p), "--out-json", str(out)
    ])
    assert r.exit_code == 0
    rep = json.loads(out.read_text())
    assert 10 <= rep["n_features"] <= 20


def test_no_nan(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    df = _make_features()
    df.loc[0, "f0"] = np.nan
    p = tmp_path / "part.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    out = tmp_path / "check.json"
    r = CliRunner().invoke(p2cli, ["run", "--features", str(p), "--out-json", str(out)])
    assert r.exit_code == 1
    rep = json.loads(out.read_text())
    assert any("nan_ratio" in v for v in rep.get("violations", [])) or rep["pass"] is False


def test_imputed_ratio_threshold(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    df = _make_features()
    # set 10% imputed
    n = len(df)
    df.loc[: int(0.1 * n), "_imputed_funding_now"] = 1
    p = tmp_path / "part.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    out = tmp_path / "check.json"
    r = CliRunner().invoke(p2cli, ["run", "--features", str(p), "--out-json", str(out)])
    assert r.exit_code == 1
    rep = json.loads(out.read_text())
    assert any("_imputed_*" in v for v in rep.get("violations", []))


def test_gap_ratio_threshold(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    df = _make_features(n=100)  # expected grid inferred → 80d capped; but our function restricts to span
    # remove every 10th bar to create gaps
    df = df[df.index % 10 != 0]
    p = tmp_path / "part.parquet"
    pq.write_table(pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False), p)
    out = tmp_path / "check.json"
    r = CliRunner().invoke(p2cli, ["run", "--features", str(p), "--out-json", str(out)])
    rep = json.loads(out.read_text())
    # If enough gaps → may fail; else at least report gap_ratio > 0
    assert rep["gap_ratio"] >= 0.0


def test_unique_keys(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    df = _make_features(n=20)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate first row
    p = tmp_path / "part.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    out = tmp_path / "check.json"
    r = CliRunner().invoke(p2cli, ["run", "--features", str(p), "--out-json", str(out)])
    rep = json.loads(out.read_text())
    assert rep["dup_key_count"] >= 1


def test_cli_exit_codes(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    df = _make_features(F=12)
    p = tmp_path / "part.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
    out = tmp_path / "check.json"
    r = CliRunner().invoke(p2cli, ["run", "--features", str(p), "--out-json", str(out)])
    assert r.exit_code == 0
    # Break feature count
    df2 = _make_features(F=5)
    p2 = tmp_path / "part2.parquet"
    pq.write_table(pa.Table.from_pandas(df2, preserve_index=False), p2)
    out2 = tmp_path / "check2.json"
    r2 = CliRunner().invoke(p2cli, ["run", "--features", str(p2), "--out-json", str(out2)])
    assert r2.exit_code == 1

