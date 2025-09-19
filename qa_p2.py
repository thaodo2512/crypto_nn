from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd
import typer


app = typer.Typer(help="P2 QA – validate 5m features acceptance gate")


def read_parquet_glob(glob: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


@app.command()
def validate(
    glob: str = typer.Option(..., "--glob"),
    qa: str = typer.Option(..., "--qa"),
    horizon_days: int = typer.Option(180, "--horizon_days"),
) -> None:
    df = read_parquet_glob(glob)
    if df.empty:
        typer.echo("No features found; failing QA.")
        Path(qa).parent.mkdir(parents=True, exist_ok=True)
        with open(qa, "w") as f:
            json.dump({"error": "no_data"}, f, indent=2)
        raise typer.Exit(code=1)

    ts_end = pd.to_datetime(df["ts"].max(), utc=True)
    ts_start = ts_end - pd.Timedelta(days=horizon_days)
    expected_idx = pd.date_range(ts_start, ts_end, freq="5min")
    window = df[(df["ts"] >= expected_idx[0]) & (df["ts"] <= expected_idx[-1])].copy()

    dup_key_count = int(window.duplicated(subset=["symbol", "ts"]).sum())
    present_bars = int(window["ts"].nunique())
    expected_bars = int(len(expected_idx))
    gap_ratio = float(1 - present_bars / expected_bars) if expected_bars else 1.0

    feature_cols = [
        "ret_5m",
        "ret_1h",
        "ret_4h",
        "hl_range",
        "co_ret",
        "vol_z",
        "oi_z",
        "fund_now_z",
        "funding_pctile_30d",
        "cvd_diff_z",
        "perp_share_60m",
        "liq60_z",
        "rv_5m_z",
        "hour_of_week_sin",
        "hour_of_week_cos",
        "_imputed_funding_now",
        "_imputed_oi_now",
    ]
    per_column: Dict[str, Dict[str, float]] = {}
    for c in feature_cols:
        if c not in window.columns:
            continue
        col = window[c]
        nan_ratio = float(col.isna().mean())
        if c in ("fund_now_z", "funding_pctile_30d"):
            imputed_ratio = float(window.get("_imputed_funding_now", pd.Series(0, index=window.index)).mean())
        elif c in ("oi_z",):
            imputed_ratio = float(window.get("_imputed_oi_now", pd.Series(0, index=window.index)).mean())
        else:
            imputed_ratio = 0.0
        per_column[c] = {
            "nan_ratio": nan_ratio,
            "imputed_ratio": float(imputed_ratio),
            "min": float(col.min(skipna=True)) if col.dtype.kind in {"f", "i"} else 0.0,
            "max": float(col.max(skipna=True)) if col.dtype.kind in {"f", "i"} else 0.0,
        }

    report = {
        "expected_bars": expected_bars,
        "present_bars": present_bars,
        "gap_ratio": gap_ratio,
        "dup_key_count": dup_key_count,
        "per_column": per_column,
    }
    Path(qa).parent.mkdir(parents=True, exist_ok=True)
    with open(qa, "w") as f:
        json.dump(report, f, indent=2)

    fail = False
    if gap_ratio > 0.005:
        fail = True
    for stats in per_column.values():
        if stats.get("nan_ratio", 0.0) > 0.0:
            fail = True
        if stats.get("imputed_ratio", 0.0) > 0.05:
            fail = True
    if fail:
        typer.echo("QA failed: gaps>0.5% or NaNs>0 or impute>5% present.")
        raise typer.Exit(code=1)
    typer.echo("QA passed.")


if __name__ == "__main__":
    app()

