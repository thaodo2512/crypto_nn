import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd
import typer


app = typer.Typer(help="Quality assurance for 15m Parquet dataset and DuckDB helpers")


def _read_parquet_glob(glob: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    sel = "*" if not cols else ",".join(cols)
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT {sel} FROM read_parquet('{glob}')").df()
        # Ensure ts is tz-aware UTC
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df
    finally:
        con.close()


def _expected_index_180d(ts_end: pd.Timestamp) -> pd.DatetimeIndex:
    ts_end = pd.to_datetime(ts_end, utc=True)
    # Snap to 15m grid (right edge)
    minute = ts_end.minute - (ts_end.minute % 15) + 15
    if minute == 60:
        ts_end = (ts_end.floor("H") + pd.Timedelta(hours=1))
    else:
        ts_end = ts_end.floor("H") + pd.Timedelta(minutes=minute)
    ts_start = ts_end - pd.Timedelta(days=180)
    return pd.date_range(ts_start, ts_end, freq="15min")


@app.command("qa")
def qa_core(
    glob: str = typer.Option(..., "--glob", help="Parquet glob (e.g., data/parquet/15m/BTCUSDT/**/*.parquet)"),
    out: str = typer.Option(..., "--out", help="JSON report output path"),
    conf: Optional[str] = typer.Option(None, "--conf", help="Optional config file for QA thresholds"),
) -> None:
    df = _read_parquet_glob(glob)
    if df.empty:
        typer.echo("No data found in glob; failing QA.")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"error": "no_data"}, f, indent=2)
        raise typer.Exit(code=1)

    # Work on last 180d window
    ts_end = pd.to_datetime(df["ts"].max(), utc=True)
    expected_idx = _expected_index_180d(ts_end)
    mask = (df["ts"] >= expected_idx[0]) & (df["ts"] <= expected_idx[-1])
    win = df.loc[mask].copy()

    # Duplicates
    dup_count = int(win.duplicated(subset=["symbol", "ts"]).sum())

    # Present vs expected
    present = int(win["ts"].nunique())
    expected = int(len(expected_idx))
    gaps_ratio = float(1 - present / expected) if expected else 1.0

    # Align to expected grid for column-wise missing ratios
    idx_df = pd.DataFrame({"ts": expected_idx})
    aligned = pd.merge(idx_df, win, on="ts", how="left")
    data_cols = [c for c in aligned.columns if c not in {"ts", "y", "m", "d"}]
    missing_ratios: Dict[str, float] = {}
    for c in data_cols:
        missing_ratios[c] = float(aligned[c].isna().mean())

    # Imputation ratios (only funding/OI)
    funding_impute_ratio = float(
        aligned.get("funding_now_imputed", pd.Series([False] * len(aligned))).mean()
    )
    oi_impute_ratio = float(
        aligned.get("oi_now_imputed", pd.Series([False] * len(aligned))).mean()
    )
    max_impute_ratio = max(funding_impute_ratio, oi_impute_ratio)

    # Feature NaN checks (core transforms)
    feature_cols = [
        c for c in [
            "rv_15m",
            "cvd_perp_15m",
            "perp_share_60m",
            "oi_pctile_30d",
            "funding_pctile_30d",
        ] if c in aligned.columns
    ]
    nan_counts = {c: int(aligned[c].isna().sum()) for c in feature_cols}

    # Thresholds (defaults, optionally from conf)
    missing_ratio_max = 0.005
    impute_ratio_max = 0.05
    no_nan_after_transform = True
    if conf:
        try:
            import yaml

            cfg = yaml.safe_load(Path(conf).read_text()) or {}
            qa = cfg.get("qa_targets", {})
            missing_ratio_max = float(qa.get("missing_ratio_max", missing_ratio_max))
            impute_ratio_max = float(qa.get("impute_ratio_max", impute_ratio_max))
            no_nan_after_transform = bool(qa.get("no_nan_after_transform", no_nan_after_transform))
        except Exception:
            pass

    report = {
        "window": {
            "start": expected_idx[0].isoformat(),
            "end": expected_idx[-1].isoformat(),
            "expected_bars": expected,
            "present_bars": present,
            "gaps_ratio": gaps_ratio,
            "duplicates": dup_count,
        },
        "missing_ratios": missing_ratios,
        "impute_ratios": {
            "funding": funding_impute_ratio,
            "oi": oi_impute_ratio,
        },
        "nan_counts": nan_counts,
    }

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    # Fail on thresholds
    fail = (gaps_ratio > missing_ratio_max) or (max_impute_ratio > impute_ratio_max)
    if no_nan_after_transform and any(v > 0 for v in nan_counts.values()):
        fail = True
    if fail:
        typer.echo("QA failed: gaps or imputation above thresholds.")
        raise typer.Exit(code=1)
    typer.echo("QA passed.")


@app.command("duckdb-view")
def duckdb_view(
    glob: str = typer.Option(..., "--glob", help="Parquet glob to create a DuckDB view for"),
    view: str = typer.Option("bars_15m", "--view", help="View name to create"),
    db: Optional[str] = typer.Option(None, "--db", help="Optional DuckDB file to persist the view (e.g., meta/duckdb/p1.duckdb)"),
) -> None:
    # Use in-memory if db is None; else persist in file
    con = duckdb.connect(database=db or ":memory:")
    try:
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{glob}')")
        # Demonstrate the view exists by counting rows quickly
        cnt = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        loc = db or "<memory>"
        typer.echo(f"Created DuckDB view '{view}' in {loc} with ~{cnt} rows.")
    finally:
        con.close()


if __name__ == "__main__":
    app()
