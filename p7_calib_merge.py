from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import duckdb
import numpy as np
import pandas as pd
import typer


app = typer.Typer(help="P7 – Merge per-fold probs with calibration + ensemble into a single calibrated parquet")


def _read_parquet(glob: str, sel: str = "*") -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT {sel} FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    if z.size == 0:
        return z
    z = z / max(T, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _calibrate_probs(df: pd.DataFrame, cal: Dict[str, Dict[str, float]]) -> np.ndarray:
    # Accept schemas: logits_*, p_* or p_wait/p_long/p_short
    if any(c.startswith("logits_") for c in df.columns):
        logits = df[[c for c in df.columns if c.startswith("logits_")]].to_numpy()
        T = float(cal.get(str(int(df["fold_id"].iloc[0])), {}).get("temperature", 1.0))
        return _softmax(logits, T=T)
    elif any(c.startswith("p_") for c in df.columns):
        p = df[[c for c in df.columns if c.startswith("p_")]].to_numpy()
    elif set(["p_wait", "p_long", "p_short"]).issubset(set(df.columns)):
        p = df[["p_wait", "p_long", "p_short"]].to_numpy()
    else:
        raise typer.BadParameter("Missing probabilities/logits columns in input probs")
    # Convert to logits via log and apply temperature
    eps = 1e-8
    logits = np.log(np.clip(p, eps, 1.0))
    T = float(cal.get(str(int(df["fold_id"].iloc[0])), {}).get("temperature", 1.0))
    return _softmax(logits, T=T)


@app.command()
def run(
    probs_glob: str = typer.Option("artifacts/p5_oos_probs/fold*.parquet", "--probs-glob"),
    calib_json: str = typer.Option("models/calib.json", "--calib"),
    ensemble_json: str = typer.Option("models/ensemble_5m.json", "--ensemble"),
    out: str = typer.Option("artifacts/p6_calibrated.parquet", "--out"),
    split: str = typer.Option("oos", "--split", help="Which split to use: val|oos"),
) -> None:
    df = _read_parquet(probs_glob)
    if df.empty:
        raise typer.BadParameter("No input probabilities found")
    df = df[df.get("split", split) == split].copy()
    if df.empty:
        raise typer.BadParameter(f"No rows with split={split}")
    if "fold_id" not in df.columns:
        raise typer.BadParameter("Input probs must include fold_id column")

    # Load calibration and ensemble weights
    with open(calib_json, "r") as f:
        cal = json.load(f)
    with open(ensemble_json, "r") as f:
        ens = json.load(f)
    weights = {int(k): float(v) for k, v in ens.get("weights", {}).items()}

    parts: List[pd.DataFrame] = []
    for fid, sub in df.groupby("fold_id"):
        p_cal = _calibrate_probs(sub, cal)  # Nx3
        # Apply ensemble weight for this fold
        w = float(weights.get(int(fid), 1.0))
        # Build output frame
        out_part = pd.DataFrame({
            "ts": sub["ts"].to_numpy(),
            "symbol": sub.get("symbol", pd.Series(["BTCUSDT"] * len(sub))).to_numpy(),
            "p_0": p_cal[:, 0] * w,
            "p_1": p_cal[:, 1] * w,
            "p_2": p_cal[:, 2] * w,
        })
        parts.append(out_part)
    merged = pd.concat(parts, ignore_index=True)
    # If multiple folds yield the same ts (shouldn't for OOS), combine by weighted sum then renormalize
    g = merged.groupby(["ts", "symbol"], as_index=False)[["p_0", "p_1", "p_2"]].sum()
    s = (g[["p_0", "p_1", "p_2"]].sum(axis=1).values.reshape(-1, 1) + 1e-12)
    g[["p_0", "p_1", "p_2"]] = g[["p_0", "p_1", "p_2"]].values / s

    # Write parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(g, preserve_index=False), out)
    typer.echo(f"Calibrated probs written to {out} with {len(g)} rows")


if __name__ == "__main__":
    app()

