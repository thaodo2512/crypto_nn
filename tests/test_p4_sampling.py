import json
import numpy as np
import pandas as pd

from folds import make_purged_folds
from iforest_gate import fit_if_rolling
from smote_train import smote_long_short_only, apply_per_fold
from windows import build_sequence_windows
from cli_p4 import app as cli
from typer.testing import CliRunner


def _mk_feats_labels(n=600, start="2024-01-01T00:00:00Z"):
    idx = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ts": idx,
        "symbol": ["BTCUSDT"] * n,
        "ret_5m": rng.standard_normal(n),
        "vol_z": rng.standard_normal(n),
        "oi_z": rng.standard_normal(n),
        "fund_now_z": rng.standard_normal(n),
        "cvd_diff_z": rng.standard_normal(n),
        "liq60_z": rng.standard_normal(n),
        "rv_5m_z": rng.standard_normal(n),
        "perp_share_60m": rng.random(n),
        # OHLC for ATR proxy if needed
        "open": 100 + rng.standard_normal(n),
        "high": 100 + rng.standard_normal(n) + 0.5,
        "low": 100 + rng.standard_normal(n) - 0.5,
        "close": 100 + rng.standard_normal(n),
    })
    # Make labels skewed towards WAIT
    y = np.array(["WAIT"] * n)
    y[50:70] = "LONG"
    y[120:140] = "SHORT"
    lab = pd.DataFrame({"ts": idx, "symbol": ["BTCUSDT"] * n, "label": y})
    return df, lab


def test_purged_folds_embargo():
    idx = pd.date_range("2024-01-01", periods=600, freq="5min", tz="UTC")
    folds = make_purged_folds(idx, n_folds=5, embargo="1D")
    for f in folds:
        tr_last = idx[f["train_idx"]].max() if len(f["train_idx"]) else idx.min()
        vl_first = idx[f["val_idx"]].min()
        assert (vl_first - tr_last) >= pd.Timedelta(days=1)


def test_if_mask_rate():
    df, lab = _mk_feats_labels()
    gate = fit_if_rolling(df, q=0.9, rolling_days=3, seed=42)
    keep_ratio = gate["keep"].mean()
    assert 0.75 <= keep_ratio <= 0.95


def test_smote_no_wait():
    X = np.random.randn(100, 10)
    y = np.array(["WAIT"] * 80 + ["LONG"] * 10 + ["SHORT"] * 10)
    Xa, ya = smote_long_short_only(X, y, seed=42)
    # WAIT count unchanged
    assert (ya == "WAIT").sum() == (y == "WAIT").sum()


def test_smote_only_train(tmp_path):
    df, lab = _mk_feats_labels()
    X, y, meta = build_sequence_windows(df, lab, W=12)
    folds = make_purged_folds(meta["ts"], n_folds=3, embargo="1D")
    out = tmp_path / "aug"
    counts = apply_per_fold(X, y, meta, folds, out_root=str(out), seed=42)
    # Ensure files per fold exist; VAL/OOS untouched (we only write TRAIN)
    for f in folds:
        assert (out / str(f["fold_id"]) / "train.parquet").exists()


def test_acceptance_wait_share(tmp_path):
    df, lab = _mk_feats_labels()
    X, y, meta = build_sequence_windows(df, lab, W=12)
    folds = make_purged_folds(meta["ts"], n_folds=3, embargo="1D")
    counts = apply_per_fold(X, y, meta, folds, out_root=str(tmp_path / "aug"), seed=42)
    # Check WAIT share ≤ 0.60 after augmentation
    for fid, d in counts.items():
        post_total = d.get("post_LONG", 0) + d.get("post_SHORT", 0) + d.get("post_WAIT", 0)
        if post_total:
            assert d.get("post_WAIT", 0) / post_total <= 0.60


def test_cli_pipeline(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    runner = CliRunner()
    df, lab = _mk_feats_labels(400)
    feat_root = tmp_path / "data" / "features" / "5m" / "BTCUSDT" / "y=2024" / "m=01" / "d=01"
    lab_root = tmp_path / "data" / "labels" / "5m" / "BTCUSDT" / "y=2024" / "m=01" / "d=01"
    feat_root.mkdir(parents=True, exist_ok=True)
    lab_root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), feat_root / "part-20240101.parquet")
    pq.write_table(pa.Table.from_pandas(lab, preserve_index=False), lab_root / "part-20240101.parquet")

    # IF gate
    r1 = runner.invoke(cli, [
        "iforest-train",
        "--features", str(feat_root / "part-20240101.parquet"),
        "--labels", str(lab_root / "part-20240101.parquet"),
        "--out", str(tmp_path / "mask.parquet"),
        "--q", "0.9", "--rolling-days", "3", "--seed", "42", "--folds", "3",
    ])
    assert r1.exit_code == 0, r1.output
    # SMOTE windows
    r2 = runner.invoke(cli, [
        "smote-windows",
        "--features", str(feat_root / "part-20240101.parquet"),
        "--labels", str(lab_root / "part-20240101.parquet"),
        "--mask", str(tmp_path / "mask.parquet"),
        "--W", "12",
        "--out", str(tmp_path / "aug"),
        "--seed", "42", "--folds", "3",
    ])
    assert r2.exit_code == 0, r2.output
    # Report
    (tmp_path / "data" / "train").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "data" / "train" / "counts.json", "w") as f:
        json.dump({"0": {"post_WAIT": 10, "post_LONG": 10, "post_SHORT": 10}}, f)
    r3 = runner.invoke(cli, [
        "report-classmix",
        "--pre", str(tmp_path / "data" / "train"),
        "--post", str(tmp_path / "aug"),
        "--out", str(tmp_path / "reports" / "p4_classmix.json"),
    ])
    assert r3.exit_code == 0, r3.output

