from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import duckdb
import numpy as np
import pandas as pd
import typer

from folds import make_purged_folds
from iforest_gate import export_mask_per_fold
from windows import build_sequence_windows
from smote_train import apply_per_fold
from reporting_p4 import classmix_report


app = typer.Typer(help="P4 – Sampling: IF gate + SMOTE per fold (5m, 80d)")


def _read_parquet(glob: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _setup_log() -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("p4")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/p4_sampling.log")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger


@app.command("iforest-train")
def iforest_train(
    features: str = typer.Option(..., "--features"),
    labels: str = typer.Option(..., "--labels"),
    out: str = typer.Option("data/masks/ifgate_5m.parquet", "--out"),
    q: float = typer.Option(0.995, "--q"),
    rolling_days: int = typer.Option(30, "--rolling-days"),
    seed: int = typer.Option(42, "--seed"),
    folds_n: int = typer.Option(5, "--folds"),
) -> None:
    logger = _setup_log()
    feat = _read_parquet(features)
    lab = _read_parquet(labels)
    typer.echo(f"[P4] IF gate: features rows={len(feat):,}, labels rows={len(lab):,}")
    # Make folds on features' ts
    ts = feat.sort_values(["symbol", "ts"])['ts']
    folds = make_purged_folds(ts, n_folds=folds_n, embargo='1D')
    logger.info(f"Starting IF gate q={q} rolling_days={rolling_days} folds={folds_n}")
    mask_df = export_mask_per_fold(feat, lab, folds, q=q, rolling_days=rolling_days, seed=seed)
    # Persist mask as Parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(mask_df, preserve_index=False), out)
    logger.info(f"IF mask written to {out} rows={len(mask_df):,}")
    typer.echo("IF gate completed.")


@app.command("smote-windows")
def smote_windows(
    features: str = typer.Option(..., "--features"),
    labels: str = typer.Option(..., "--labels"),
    mask: str = typer.Option(..., "--mask"),
    W: int = typer.Option(144, "--W"),
    out: str = typer.Option("data/aug/train_smote", "--out"),
    seed: int = typer.Option(42, "--seed"),
    folds_n: int = typer.Option(5, "--folds"),
) -> None:
    logger = _setup_log()
    feat = _read_parquet(features)
    lab = _read_parquet(labels)
    mask_df = _read_parquet(mask)
    typer.echo(f"[P4] SMOTE: features rows={len(feat):,}, labels rows={len(lab):,}, mask rows={len(mask_df):,}")
    # Join mask: keep only (ts,symbol) with keep=1 for their fold
    # Build folds on feats to get consistent train indices
    ts = feat.sort_values(["symbol", "ts"])['ts']
    folds = make_purged_folds(ts, n_folds=folds_n, embargo='1D')

    # Filter labels to mask keep=1 overall
    lab = lab.merge(mask_df[mask_df["keep"] == 1][["ts", "symbol"]].drop_duplicates(), on=["ts", "symbol"], how="inner")
    X, y, meta = build_sequence_windows(feat, lab, W=W)
    typer.echo(f"[P4] Windows built: X={X.shape} y={len(y)} meta={len(meta)}; applying SMOTE per fold...")
    if X.size == 0:
        raise typer.BadParameter("No windows created; consider reducing W")
    counts = apply_per_fold(X, y, meta, folds, out_root=out, seed=seed, ts_for_folds=ts)
    # Write counts.json to generic path and per-symbol namespaced path (if single symbol)
    pre_dir_generic = Path("data/train")
    pre_dir_generic.mkdir(parents=True, exist_ok=True)
    with open(pre_dir_generic / "counts.json", "w") as f:
        json.dump(counts, f)
    # Also write to data/train/<symbol>/counts.json when a single symbol is present
    syms = sorted(pd.Series(meta["symbol"].unique(), dtype=str).tolist()) if not meta.empty else []
    if len(syms) == 1:
        pre_dir_sym = Path("data/train") / syms[0]
        pre_dir_sym.mkdir(parents=True, exist_ok=True)
        with open(pre_dir_sym / "counts.json", "w") as f:
            json.dump(counts, f)
    logger.info(f"SMOTE applied per fold to {out}")
    typer.echo("SMOTE windows completed.")


@app.command("report-classmix")
def report_classmix(
    pre: str = typer.Option("data/train", "--pre"),
    post: str = typer.Option("data/aug/train_smote", "--post"),
    out: str = typer.Option("reports/p4_classmix.json", "--out"),
) -> None:
    # Load per-fold counts produced by apply_per_fold
    with open(Path(pre) / "counts.json") as f:
        counts = json.load(f)
    classmix_report({int(k): v for k, v in counts.items()}, out)
    typer.echo("Class mix report written; acceptance checks passed.")


if __name__ == "__main__":
    app()
