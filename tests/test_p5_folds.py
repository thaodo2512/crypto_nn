import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from meta.cv_splits import emit_folds_json, BARS_PER_DAY
from cli_p5 import GRUClassifier, build_windows
from p5_export_oos import run as export_oos


def _mk_feats_labels(n=288*10, start="2025-01-01T00:00:00Z"):
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
    })
    y = np.array(["WAIT"] * n)
    y[200:260] = "LONG"
    y[300:360] = "SHORT"
    lab = pd.DataFrame({"ts": idx, "symbol": ["BTCUSDT"] * n, "label": y})
    return df, lab


def test_emit_and_export_oos(tmp_path):
    # Prepare synthetic features/labels
    feat, lab = _mk_feats_labels(n=288*10)  # 10 days
    froot = tmp_path / "data" / "features" / "5m" / "BTCUSDT" / "y=2025" / "m=01" / "d=01"
    lroot = tmp_path / "data" / "labels" / "5m" / "BTCUSDT" / "y=2025" / "m=01" / "d=01"
    froot.mkdir(parents=True, exist_ok=True)
    lroot.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(feat, preserve_index=False), froot / "part-20250101.parquet")
    pq.write_table(pa.Table.from_pandas(lab, preserve_index=False), lroot / "part-20250101.parquet")

    # Emit folds.json with smaller settings to fit 10 days
    folds_path = tmp_path / "artifacts" / "folds.json"
    ts_all = feat["ts"]
    emit_folds_json(ts_all, folds_path, tf="5m", symbol="BTCUSDT", window=144, horizon=36, embargo_bars=BARS_PER_DAY, n_folds=2, min_oos=144, val_bars=144)
    obj = json.loads(folds_path.read_text())
    assert len(obj["folds"]) == 2
    assert all(f["counts"]["oos"] > 0 for f in obj["folds"])

    # Create dummy checkpoints for 2 folds
    for fid in (0, 1):
        ck_dir = tmp_path / "models" / "gru_5m" / str(fid)
        ck_dir.mkdir(parents=True, exist_ok=True)
        net = GRUClassifier(input_dim=9)
        torch.save({"state_dict": net.state_dict()}, ck_dir / "best.pt")

    # Export OOS using folds.json
    out_dir = tmp_path / "artifacts" / "p5_oos_probs"
    export_oos(
        features=str(froot / "part-20250101.parquet"),
        labels=str(lroot / "part-20250101.parquet"),
        models_root=str(tmp_path / "models" / "gru_5m"),
        out=str(out_dir),
        embargo="1D",
        folds=2,
        window=144,
        folds_json=str(folds_path),
    )
    # Check outputs
    for fid in (0, 1):
        fp = out_dir / f"fold{fid}.parquet"
        assert fp.exists()
        df = pd.read_parquet(fp)
        assert len(df) > 0
        for c in ["ts", "symbol", "p_long", "p_short", "p_wait", "y", "fold_id", "split"]:
            assert c in df.columns

