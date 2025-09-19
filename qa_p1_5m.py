from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import duckdb
import pandas as pd
import typer


app = typer.Typer(help="P1 5m – Acceptance QA for raw 5m lake")


def read_parquet_glob(glob: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def evaluate_qa(df: pd.DataFrame, horizon_days: int = 80) -> Tuple[Dict, bool]:
    ts_end = pd.to_datetime(df["ts"].max(), utc=True)
    ts_start = ts_end - pd.Timedelta(days=horizon_days)
    expected_idx = pd.date_range(ts_start, ts_end, freq="5min")
    window = df[(df["ts"] >= expected_idx[0]) & (df["ts"] <= expected_idx[-1])].copy()

    # duplicate keys
    dup_key_count = int(window.duplicated(subset=["symbol", "ts"]).sum())
    present_bars = int(window["ts"].nunique())
    expected_bars = int(len(expected_idx))
    gap_ratio = float(1 - present_bars / expected_bars) if expected_bars else 1.0

    columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_now",
        "oi_now",
        "taker_buy_usd",
        "taker_sell_usd",
        "spot_taker_buy_usd",
        "spot_taker_sell_usd",
        "liq_notional_raw",
        "_imputed_funding_now",
        "_imputed_oi_now",
    ]
    per_column: Dict[str, Dict[str, float]] = {}
    for c in columns:
        if c not in window.columns:
            continue
        col = window[c]
        nan_ratio = float(col.isna().mean())
        if c == "funding_now":
            imputed_ratio = float(window.get("_imputed_funding_now", pd.Series(0, index=window.index)).mean())
        elif c == "oi_now":
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
        "expected_bars_80d": horizon_days * 24 * 12,
        "present_bars": present_bars,
        "gap_ratio": gap_ratio,
        "dup_key_count": dup_key_count,
        "per_column": per_column,
    }
    # gate
    fail = gap_ratio > 0.005
    if any(stats.get("nan_ratio", 0.0) > 0.0 for stats in per_column.values()):
        fail = True
    if any(stats.get("imputed_ratio", 0.0) > 0.05 for stats in per_column.values()):
        fail = True
    return report, fail


@app.command("qa")
def qa(
    glob: str = typer.Option(..., "--glob"),
    out: str = typer.Option("reports/p1_qa_core_5m_80d.json", "--out"),
    horizon_days: int = typer.Option(80, "--days", help="Horizon (days) for QA window; default 80"),
) -> None:
    df = read_parquet_glob(glob)
    if df.empty:
        typer.echo("No data found; failing QA")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"error": "no_data"}, f, indent=2)
        raise typer.Exit(code=1)
    report, fail = evaluate_qa(df, horizon_days=horizon_days)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    if fail:
        typer.echo("FAIL gaps>0.5% or NaN>0 or impute>5%")
        raise typer.Exit(code=1)
    typer.echo("PASS")


if __name__ == "__main__":
    app()
