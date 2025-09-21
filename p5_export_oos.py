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
from meta.cv_splits import emit_folds_json
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
    folds_json: str = typer.Option("artifacts/folds.json", "--folds-json"),
    val_bars: int = typer.Option(144, "--val-bars", help="Fallback validation bars if folds.json has empty VAL"),
) -> None:
    feat = _read_parquet(features)
    lab = _read_parquet(labels)
    X, y, meta = build_windows(feat, lab, W=window)
    if X.shape[0] == 0:
        raise typer.BadParameter("No windows built; check inputs or window length")
    # Load canonical folds if provided; else build one (but don't skip folds)
    folds_spec = None
    try:
        import json
        if folds_json and Path(folds_json).exists():
            folds_spec = json.load(open(folds_json, "r"))
    except Exception:
        folds_spec = None
    if folds_spec is None:
        ts_sorted = meta.sort_values("ts")["ts"]
        folds = make_purged_folds(ts_sorted, n_folds=folds_n, embargo=embargo)
        # Convert to spec-like dict using index windows
        ts_list = list(pd.to_datetime(ts_sorted, utc=True))
        folds_spec = {
            "meta": {"window": window, "horizon": 36, "embargo_bars": 288, "n_folds": folds_n},
            "folds": [
                {
                    "fold_id": int(f["fold_id"]),
                    "train": [[ts_list[f["train_idx"][0]].isoformat(), ts_list[f["train_idx"][-1]].isoformat()]] if len(f["train_idx"]) else [],
                    "val": [[ts_list[f["val_idx"][0]].isoformat(), ts_list[f["val_idx"][-1]].isoformat()]] if len(f["val_idx"]) else [],
                    "oos": [[ts_list[f["oos_idx"][0]].isoformat(), ts_list[f["oos_idx"][-1]].isoformat()]] if len(f["oos_idx"]) else [],
                }
                for f in folds
            ],
        }

    # Standardize per fold using TRAIN stats, then emit OOS probs
    Path(out).mkdir(parents=True, exist_ok=True)
    for f in folds_spec["folds"]:
        fid = int(f["fold_id"])
        # Build boolean masks by time spans
        def mask_from_spans(spans):
            if not spans:
                return pd.Series(False, index=meta.index)
            m = pd.Series(False, index=meta.index)
            for s, e in spans:
                sdt = pd.to_datetime(s, utc=True)
                edt = pd.to_datetime(e, utc=True)
                m = m | ((meta["ts"] >= sdt) & (meta["ts"] <= edt))
            return m

        sel_tr = mask_from_spans(f.get("train", []))
        sel_oo = mask_from_spans(f.get("oos", []))
        sel_val = mask_from_spans(f.get("val", []))
        tr_ids = np.where(sel_tr.to_numpy())[0]
        oo_ids = np.where(sel_oo.to_numpy())[0]
        if tr_ids.size == 0 or oo_ids.size == 0:
            raise typer.BadParameter(f"Empty TRAIN/OOS for fold {fid} – check folds.json eligibility")
        # If VAL selection is empty, synthesize a small validation region just before OOS with an embargo gap
        if not sel_val.to_numpy().any():
            try:
                emb_td = pd.to_timedelta(embargo)
            except Exception:
                emb_td = pd.to_timedelta("1D")
            oos_start_ts = meta.iloc[oo_ids[0]]["ts"]
            cutoff = oos_start_ts - emb_td
            # take up to val_bars decision rows before cutoff
            window_minutes = 5 * int(val_bars)
            val_mask = (meta["ts"] <= cutoff) & (meta["ts"] > cutoff - pd.Timedelta(minutes=window_minutes))
            if not val_mask.any() and tr_ids.size:
                # fall back: last up to val_bars from train indices
                keep = int(min(int(val_bars), tr_ids.size))
                vidx = tr_ids[-keep:]
            else:
                vidx = np.where(val_mask.to_numpy())[0]
        else:
            vidx = np.where(sel_val.to_numpy())[0]
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
        # Validation split (ensure present for calibration)
        if vidx.size > 0:
            with torch.no_grad():
                bs = 256
                for start in range(0, len(vidx), bs):
                    sl = vidx[start : start + bs]
                    xb = torch.tensor(Xn[sl], dtype=torch.float32)
                    logits = net(xb)
                    p = F.softmax(logits, dim=1).cpu().numpy()
                    log_np = logits.cpu().numpy()
                    for k, idx in enumerate(sl):
                        probs_rows.append(
                            {
                                "ts": meta.iloc[idx]["ts"],
                                "symbol": meta.iloc[idx]["symbol"],
                                "p_wait": float(p[k, 0]),
                                "p_long": float(p[k, 1]),
                                "p_short": float(p[k, 2]),
                                "logits_0": float(log_np[k, 0]),
                                "logits_1": float(log_np[k, 1]),
                                "logits_2": float(log_np[k, 2]),
                                "y": int(1 if str(lab.iloc[idx]["label"]).upper() == "LONG" else 2 if str(lab.iloc[idx]["label"]).upper() == "SHORT" else 0),
                                "fold_id": fid,
                                "split": "val",
                            }
                        )
        with torch.no_grad():
            bs = 256
            for start in range(0, len(oo_ids), bs):
                sl = oo_ids[start : start + bs]
                xb = torch.tensor(Xn[sl], dtype=torch.float32)
                logits = net(xb)
                p = F.softmax(logits, dim=1).cpu().numpy()
                log_np = logits.cpu().numpy()
                for k, idx in enumerate(sl):
                    probs_rows.append(
                        {
                            "ts": meta.iloc[idx]["ts"],
                            "symbol": meta.iloc[idx]["symbol"],
                            "p_wait": float(p[k, 0]),
                            "p_long": float(p[k, 1]),
                            "p_short": float(p[k, 2]),
                            "logits_0": float(log_np[k, 0]),
                            "logits_1": float(log_np[k, 1]),
                            "logits_2": float(log_np[k, 2]),
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
        typer.echo(f"OOS probs written under {out} (fold={fid} val_rows={int(sel_val.sum())} oos_rows={len(oo_ids)})")


@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    features: str = typer.Option(None, "--features"),
    labels: str = typer.Option(None, "--labels"),
    models_root: str = typer.Option("models/gru_5m", "--models-root"),
    out: str = typer.Option("artifacts/p5_oos_probs", "--out"),
    embargo: str = typer.Option("1D", "--embargo"),
    folds_n: int = typer.Option(5, "--folds"),
    window: int = typer.Option(144, "--window"),
    folds_json: str = typer.Option("artifacts/folds.json", "--folds-json"),
) -> None:
    # Allow calling without subcommand as a convenience in Docker Compose
    if ctx.invoked_subcommand is None:
        if not features or not labels:
            raise typer.BadParameter("Missing required options --features/--labels; use --help for usage.")
        run(features=features, labels=labels, models_root=models_root, out=out, embargo=embargo, folds_n=folds_n, window=window, folds_json=folds_json)

if __name__ == "__main__":
    app()
