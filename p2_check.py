from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import typer


app = typer.Typer(help="P2 Checker – validate 5m features for 80 days (BTCUSDT)")


EXPECTED_BARS_80D = 80 * 24 * 12  # 23040


def _read_parquet(glob: str, sel: str = "*") -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT {sel} FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _detect_feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    exclude = {"ts", "symbol"}
    numeric_cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in {"f", "i"}]
    flag_cols = [c for c in numeric_cols if c.startswith("_imputed_")]
    # Count features excluding imputed flags
    feature_cols = [c for c in numeric_cols if c not in flag_cols]
    return feature_cols, flag_cols


def _grid_from_raw(raw: Optional[pd.DataFrame], feats: pd.DataFrame) -> pd.DatetimeIndex:
    if raw is not None and not raw.empty:
        # Use observed raw timestamps as the expected grid (robust for tests and partial windows)
        return pd.DatetimeIndex(sorted(pd.to_datetime(raw["ts"].unique(), utc=True)))
    # Else infer from features: start at min ts, span 80 days
    start = pd.to_datetime(feats["ts"].min(), utc=True)
    start = start.floor("5min")
    return pd.date_range(start, periods=EXPECTED_BARS_80D, freq="5min")


def _bench_ms(bench_csv: Optional[str]) -> Optional[float]:
    if not bench_csv or not Path(bench_csv).exists():
        return None
    try:
        with open(bench_csv, "r") as f:
            rdr = csv.DictReader(f)
            row = next(rdr)
            v = float(row.get("ms_per_bar", "nan"))
            return v
    except Exception:
        return None


@app.command("run")
def run(
    features: str = typer.Option(..., "--features", help="P2 features glob"),
    out_json: str = typer.Option("reports/p2_check_5m_80d.json", "--out-json"),
    raw: Optional[str] = typer.Option(None, "--raw", help="Optional P1 raw glob to cross-check gaps"),
    bench_csv: Optional[str] = typer.Option(None, "--bench-csv", help="Optional bench CSV (ms_per_bar)"),
    symbol: str = typer.Option("BTCUSDT", "--symbol"),
    tz: str = typer.Option("UTC", "--tz"),
) -> None:
    # Load data
    feats = _read_parquet(features)
    if feats.empty:
        typer.echo("FAIL: no features rows found")
        raise typer.Exit(code=1)
    if "ts" not in feats or "symbol" not in feats:
        typer.echo("FAIL: features missing required columns (ts,symbol)")
        raise typer.Exit(code=1)
    feats["ts"] = pd.to_datetime(feats["ts"], utc=True)
    feats = feats.sort_values(["symbol", "ts"]).reset_index(drop=True)

    # DuckDB helper for operator visibility
    con = duckdb.connect()
    try:
        cnt, min_ts, max_ts = con.execute(f"SELECT COUNT(*), MIN(ts), MAX(ts) FROM read_parquet('{features}')").fetchone()
        typer.echo(f"Features: rows={cnt}, min_ts={min_ts}, max_ts={max_ts}")
    finally:
        con.close()

    # Feature columns & flags
    feature_cols, flag_cols = _detect_feature_cols(feats)
    n_features = len(feature_cols)

    # NaN ratios and stats per feature
    features_report: List[Dict[str, object]] = []
    nan_violation = False
    for c in feature_cols:
        col = feats[c]
        nan_ratio = float(col.isna().mean())
        if nan_ratio > 0:
            nan_violation = True
        features_report.append(
            {
                "name": c,
                "nan_ratio": nan_ratio,
                "imputed_ratio": 0.0,
                "dtype": str(col.dtype),
                "min": float(col.min(skipna=True)) if col.dtype.kind in {"f", "i"} else 0.0,
                "max": float(col.max(skipna=True)) if col.dtype.kind in {"f", "i"} else 0.0,
            }
        )

    # Imputed flags
    impute_violation = False
    for c in flag_cols:
        ratio = float(feats[c].astype(float).mean())
        features_report.append({"name": c, "nan_ratio": float(feats[c].isna().mean()), "imputed_ratio": ratio, "dtype": str(feats[c].dtype), "min": float(feats[c].min(skipna=True) if hasattr(feats[c], 'min') else 0.0), "max": float(feats[c].max(skipna=True) if hasattr(feats[c], 'max') else 0.0)})
        if ratio > 0.05:
            impute_violation = True

    # Unique keys
    dup_key_count = int(feats.duplicated(subset=["symbol", "ts"]).sum())

    # Gap ratio (5m grid)
    raw_df = _read_parquet(raw) if raw else None
    expected_idx = _grid_from_raw(raw_df, feats)
    # If grid much larger than feats window, restrict to [min_ts, max_ts] + cap to 80d
    min_ts = feats["ts"].min()
    max_ts = feats["ts"].max()
    expected_idx = expected_idx[(expected_idx >= min_ts.floor("5min")) & (expected_idx <= max_ts.ceil("5min"))]
    present = int(feats["ts"].nunique())
    expected = int(len(expected_idx)) if len(expected_idx) > 0 else EXPECTED_BARS_80D
    gap_ratio = float(1 - present / expected) if expected else 1.0

    # Bench CSV
    ms_per_bar = _bench_ms(bench_csv)

    # Violations
    violations: List[str] = []
    if not (10 <= n_features <= 20):
        violations.append(f"n_features={n_features} outside [10,20]")
    if nan_violation:
        violations.append("nan_ratio>0 on at least one feature")
    if impute_violation:
        violations.append("_imputed_* ratio>5%")
    if gap_ratio > 0.005:
        violations.append(f"gap_ratio={gap_ratio:.6f} > 0.005")
    if ms_per_bar is not None and ms_per_bar > 50.0:
        violations.append(f"bench ms_per_bar={ms_per_bar:.3f} > 50")

    passed = len(violations) == 0

    # Report
    out = {
        "expected_bars_80d": EXPECTED_BARS_80D,
        "present_bars": present,
        "gap_ratio": gap_ratio,
        "dup_key_count": dup_key_count,
        "n_features": n_features,
        "features": features_report,
        "bench_ms_per_bar": ms_per_bar,
        "pass": passed,
        "violations": violations,
    }
    Path(Path(out_json).parent).mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # Optional wide CSV table per column
    table_path = str(Path(out_json).with_name(Path(out_json).stem + "_table.csv"))
    try:
        pd.DataFrame(features_report).to_csv(table_path, index=False)
    except Exception:
        pass

    # STDOUT summary and exit code
    if passed:
        typer.echo("PASS")
        raise typer.Exit(code=0)
    else:
        typer.echo("FAIL: " + "; ".join(violations))
        raise typer.Exit(code=1)


@app.callback()
def main(
    ctx: typer.Context,
    features: Optional[str] = typer.Option(None, "--features"),
    out_json: str = typer.Option("reports/p2_check_5m_80d.json", "--out-json"),
    raw: Optional[str] = typer.Option(None, "--raw"),
    bench_csv: Optional[str] = typer.Option(None, "--bench-csv"),
    symbol: str = typer.Option("BTCUSDT", "--symbol"),
    tz: str = typer.Option("UTC", "--tz"),
) -> None:
    # Allow calling without subcommand: python p2_check.py --features ...
    if ctx.invoked_subcommand is None:
        if not features:
            typer.echo("Missing --features; use --help for usage.")
            raise typer.Exit(code=2)
        run(features=features, out_json=out_json, raw=raw, bench_csv=bench_csv, symbol=symbol, tz=tz)


if __name__ == "__main__":
    app()
