from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import typer

from cli_p5 import GRUClassifier, build_windows
from folds import make_purged_folds


app = typer.Typer(help="P5 – Export OOS probabilities per fold from trained models")


def _read_parquet(glob: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


@app.command("run")
def run(
    features: str = typer.Option(..., "--features"),
    labels: str = typer.Option(..., "--labels"),
    models_root: str = typer.Option("models/gru_5m", "--models-root"),
    out: str = typer.Option("artifacts/p5_oos_probs", "--out"),
    embargo: str = typer.Option("1D", "--embargo"),
    folds_n: int = typer.Option(5, "--folds"),
    window: int = typer.Option(144, "--window"),
) -> None:
    feat = _read_parquet(features)
    lab = _read_parquet(labels)
    X, y, meta = build_windows(feat, lab, W=window)
    if X.shape[0] == 0:
        raise typer.BadParameter("No windows built; check inputs or window length")
    ts_sorted = meta.sort_values("ts")["ts"]
    folds = make_purged_folds(ts_sorted, n_folds=folds_n, embargo=embargo)

    # Standardize per fold using TRAIN stats, then emit OOS probs
    Path(out).mkdir(parents=True, exist_ok=True)
    for f in folds:
        fid = int(f["fold_id"])
        # Map fold indices to meta ordering
        meta_sorted = meta.sort_values("ts").reset_index(drop=True)
        m_tr_ts = meta_sorted.iloc[f["train_idx"]]["ts"]
        m_oo_ts = meta_sorted.iloc[f["oos_idx"]]["ts"]
        sel_tr = meta["ts"].isin(set(m_tr_ts))
        sel_oo = meta["ts"].isin(set(m_oo_ts))
        tr_ids = np.where(sel_tr.to_numpy())[0]
        oo_ids = np.where(sel_oo.to_numpy())[0]
        if tr_ids.size == 0 or oo_ids.size == 0:
            continue
        mu = X[tr_ids].mean(axis=(0, 1))
        std = X[tr_ids].std(axis=(0, 1))
        std[std == 0] = 1.0
        Xn = (X - mu) / std
        # Load checkpoint
        ckpt_path = Path(models_root) / str(fid) / "best.pt"
        if not ckpt_path.exists():
            continue
        state = torch.load(str(ckpt_path), map_location="cpu")
        input_dim = X.shape[2]
        net = GRUClassifier(input_dim=input_dim)
        net.load_state_dict(state.get("state_dict", state))
        net.eval()
        probs_rows: List[Dict] = []
        with torch.no_grad():
            bs = 256
            for start in range(0, len(oo_ids), bs):
                sl = oo_ids[start : start + bs]
                xb = torch.tensor(Xn[sl], dtype=torch.float32)
                logits = net(xb)
                p = F.softmax(logits, dim=1).cpu().numpy()
                for k, idx in enumerate(sl):
                    probs_rows.append(
                        {
                            "ts": meta.iloc[idx]["ts"],
                            "symbol": meta.iloc[idx]["symbol"],
                            "p_wait": float(p[k, 0]),
                            "p_long": float(p[k, 1]),
                            "p_short": float(p[k, 2]),
                            "y": int(1 if str(lab.iloc[idx]["label"]).upper() == "LONG" else 2 if str(lab.iloc[idx]["label"]).upper() == "SHORT" else 0),
                            "fold_id": fid,
                            "split": "oos",
                        }
                    )
        if probs_rows:
            df = pd.DataFrame(probs_rows)
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), Path(out) / f"fold{fid}.parquet")
    typer.echo(f"OOS probs written under {out}")


if __name__ == "__main__":
    app()

