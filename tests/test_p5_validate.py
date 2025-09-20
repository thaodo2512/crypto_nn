import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from typer.testing import CliRunner

from cli_p5 import GRUClassifier
from p5_validate import app as p5v_cli


def _mk_feats_labels(n=500, start="2024-01-01T00:00:00Z"):
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
    y[50:70] = "LONG"
    y[120:140] = "SHORT"
    lab = pd.DataFrame({"ts": idx, "symbol": ["BTCUSDT"] * n, "label": y})
    return df, lab


def test_cli_exit_codes(tmp_path):
    # Synthetic features/labels
    feat, lab = _mk_feats_labels(500)
    froot = tmp_path / "data" / "features" / "5m" / "BTCUSDT" / "y=2024" / "m=01" / "d=01"
    lroot = tmp_path / "data" / "labels" / "5m" / "BTCUSDT" / "y=2024" / "m=01" / "d=01"
    froot.mkdir(parents=True, exist_ok=True)
    lroot.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(feat, preserve_index=False), froot / "part-20240101.parquet")
    pq.write_table(pa.Table.from_pandas(lab, preserve_index=False), lroot / "part-20240101.parquet")

    # Folds: split by halves
    ts_sorted = feat.sort_values(["symbol", "ts"])['ts']
    half = len(ts_sorted) // 2
    folds = {
        "folds": [
            {
                "fold_id": 0,
                "train": [str(t) for t in ts_sorted.iloc[:half-288].tolist()],
                "val": [str(t) for t in ts_sorted.iloc[half-288:half].tolist()],
                "oos": [str(t) for t in ts_sorted.iloc[half:].tolist()],
            }
        ]
    }
    folds_path = tmp_path / "artifacts" / "folds.json"
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    folds_path.write_text(json.dumps(folds))

    # Checkpoint
    ck_dir = tmp_path / "models" / "gru_5m" / "fold0"
    ck_dir.mkdir(parents=True, exist_ok=True)
    net = GRUClassifier(input_dim=9)
    torch.save({"state_dict": net.state_dict()}, ck_dir / "best.pt")

    # OOS probs
    probs = pd.DataFrame({
        "ts": ts_sorted.iloc[half:half+10].values,
        "symbol": ["BTCUSDT"] * 10,
        "p_long": np.full(10, 1/3),
        "p_short": np.full(10, 1/3),
        "p_wait": np.full(10, 1/3),
    })
    pdir = tmp_path / "artifacts" / "p5_oos_probs"
    pdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(probs, preserve_index=False), pdir / "fold0.parquet")

    # Metrics
    metrics = {"0": {"val_loss": 1.0, "train_loss": 1.2}}
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "p5_cv_metrics.json").write_text(json.dumps(metrics))

    # Log with hints
    log = """fold=0 train_loss=1.2 val_loss=1.1\nloss: CrossEntropyLoss time-decay lambda=0.98\n"""
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "p5_train.log").write_text(log)

    runner = CliRunner()
    out_json = tmp_path / "reports" / "p5_validate.json"
    r = runner.invoke(
        p5v_cli,
        [
            "run",
            "--features", str(froot / "part-20240101.parquet"),
            "--labels", str(lroot / "part-20240101.parquet"),
            "--folds", str(folds_path),
            "--models", str(tmp_path / "models" / "gru_5m" / "fold*/best.pt"),
            "--oos-probs", str(pdir / "fold*.parquet"),
            "--train-log", str(tmp_path / "logs" / "p5_train.log"),
            "--metrics", str(tmp_path / "reports" / "p5_cv_metrics.json"),
            "--out-json", str(out_json),
        ],
    )
    assert r.exit_code in (0, 1)
    assert out_json.exists()

