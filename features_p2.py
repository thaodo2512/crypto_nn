from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from transforms import (
    ImputeResult,
    diff_series,
    encode_hour_of_week,
    ffill_with_limit_and_flag,
    safe_log1p,
    safe_log_ratio,
    winsorize_causal,
    zscore_causal,
)


app = typer.Typer(help="P2 – 5m feature builder: build, validate, bench")


def read_parquet_glob(glob: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    return d


def ensure_unique_key(df: pd.DataFrame, key: List[str]) -> None:
    dup = df.duplicated(subset=key)
    if dup.any():
        raise ValueError(f"Duplicate keys for {dup.sum()} rows: {df.loc[dup, key].head(5).to_dict(orient='records')}")


def _write_manifest_success(day_dir: Path, parquet_file: Path, rows: int, cols: int) -> None:
    import hashlib

    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = day_dir / "MANIFEST.tsv"
    digest = _sha256_file(parquet_file)
    with open(manifest, "w") as f:
        f.write(f"{parquet_file.name}\t{digest}\t{rows}\t{cols}\n")
    (day_dir / "_SUCCESS").touch()


def write_parquet_daily(df: pd.DataFrame, out_root: str, row_group_target_mb: int = 64) -> None:
    """Write partitioned Parquet: one file per day, ZSTD(3), dictionary on.

    Uses an approximate row_group_size to target ~row_group_target_mb per group.
    """
    if df.empty:
        return
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d["y"] = d["ts"].dt.year.astype("int16")
    d["m"] = d["ts"].dt.month.astype("int8")
    d["d"] = d["ts"].dt.day.astype("int8")
    # Approximate rows per group
    approx_cols = max(1, len(d.columns))
    bytes_per_row = approx_cols * 8  # rough heuristic
    rows_per_group = int((row_group_target_mb * 1024 * 1024) / max(1, bytes_per_row))
    rows_per_group = max(50_000, min(rows_per_group, 1_000_000))

    for (y, m, day), chunk in d.groupby(["y", "m", "d"], sort=True):
        day_dir = Path(out_root) / f"y={int(y):04d}" / f"m={int(m):02d}" / f"d={int(day):02d}"
        day_dir.mkdir(parents=True, exist_ok=True)

        # clean
        for p in day_dir.glob("part-*.parquet"):
            p.unlink(missing_ok=True)  # type: ignore
        (day_dir / "MANIFEST.tsv").unlink(missing_ok=True)  # type: ignore
        (day_dir / "_SUCCESS").unlink(missing_ok=True)  # type: ignore

        date_str = f"{int(y):04d}{int(m):02d}{int(day):02d}"
        out_path = day_dir / f"part-{date_str}.parquet"
        table = pa.Table.from_pandas(chunk.drop(columns=["y", "m", "d"]), preserve_index=False)
        pq.write_table(
            table,
            out_path,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
            row_group_size=rows_per_group,
        )
        _write_manifest_success(day_dir, out_path, table.num_rows, table.num_columns)


def build_features(
    df: pd.DataFrame,
    warmup: int = 8640,
    winsor_low: float = 0.01,
    winsor_high: float = 0.99,
) -> pd.DataFrame:
    """Build 5m features from P1 bars (vectorized, causal). Returns a clean DataFrame ready to write.

    Input df must contain columns: ts, symbol, and any available raw fields.
    """
    if df.empty:
        return pd.DataFrame()

    # Ensure required columns are present for stable schema
    df = _ensure_columns(
        df,
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "funding_now",
            "funding_pctile_30d",
            "oi_now",
            "oi_pctile_30d",
            "cvd_perp_5m",
            "perp_share_60m",
            "liq_notional_60m",
            "rv_5m",
        ],
    )

    # Sort and set index per symbol for causal windows
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)

    feats: List[pd.DataFrame] = []
    for sym, grp in df.groupby("symbol", sort=False):
        g = grp.copy()
        g = g.set_index("ts").sort_index()

        # Returns
        close = g["close"].astype(float)
        open_ = g["open"].astype(float)
        ret_5m = pd.Series(
            np.log(np.where((close > 0) & (close.shift(1) > 0), close / close.shift(1), 1.0)),
            index=g.index,
            dtype=float,
        )
        ret_1h = ret_5m.rolling(12, min_periods=1).sum()
        ret_4h = ret_5m.rolling(48, min_periods=1).sum()

        # Ranges/ratios
        hl_range = (g["high"].astype(float) - g["low"].astype(float)) / g["close"].astype(float)
        hl_range = hl_range.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        co_ret = safe_log_ratio(close, open_)

        # Exogenous limited ffill + flags
        fund_imp = ffill_with_limit_and_flag(g["funding_now"], limit=3)
        oi_imp = ffill_with_limit_and_flag(g["oi_now"], limit=3)

        # Causal winsorize + z-scores
        vol_z = zscore_causal(winsorize_causal(safe_log1p(g["volume"]), warmup, winsor_low, winsor_high), warmup)
        oi_z = zscore_causal(winsorize_causal(safe_log1p(oi_imp.values), warmup, winsor_low, winsor_high), warmup)
        fund_now_z = zscore_causal(winsorize_causal(fund_imp.values, warmup, winsor_low, winsor_high), warmup)

        # CVD and diffs
        if "cvd_perp_5m" in g:
            cvd_perp_5m = g["cvd_perp_5m"].astype(float)
        else:
            buy = g.get("taker_buy_usd", pd.Series(0.0, index=g.index))
            sell = g.get("taker_sell_usd", pd.Series(0.0, index=g.index))
            cvd_perp_5m = (buy - sell).fillna(0.0).cumsum()
        cvd_diff = diff_series(cvd_perp_5m)
        cvd_diff_z = zscore_causal(winsorize_causal(cvd_diff, warmup, winsor_low, winsor_high), warmup)
        # Liquidations 60m roll from raw if needed
        if "liq_notional_60m" in g:
            liq60_base = g["liq_notional_60m"].astype(float)
        else:
            liq60_base = g.get("liq_notional_raw", pd.Series(0.0, index=g.index)).astype(float).rolling("60min", min_periods=1).sum()
        liq60_z = zscore_causal(winsorize_causal(safe_log1p(liq60_base), warmup, winsor_low, winsor_high), warmup)
        rv_5m_z = zscore_causal(winsorize_causal(g["rv_5m"], warmup, winsor_low, winsor_high), warmup)

        # Perp share pass-through (ensure finite)
        perp_share_60m = g["perp_share_60m"].astype(float).replace([np.inf, -np.inf], 0.0)

        # Time encodings
        sin_how, cos_how = encode_hour_of_week(g.index)

        out = pd.DataFrame({
            "symbol": sym,
            "ret_5m": ret_5m,
            "ret_1h": ret_1h,
            "ret_4h": ret_4h,
            "hl_range": hl_range,
            "co_ret": co_ret,
            "vol_z": vol_z,
            "oi_z": oi_z,
            "fund_now_z": fund_now_z,
            "funding_pctile_30d": g["funding_pctile_30d"].astype(float),
            "cvd_perp_5m": cvd_perp_5m,
            "cvd_diff_z": cvd_diff_z,
            "perp_share_60m": perp_share_60m,
            "liq60_z": liq60_z,
            "rv_5m_z": rv_5m_z,
            "hour_of_week_sin": pd.Series(sin_how, index=g.index, dtype=float),
            "hour_of_week_cos": pd.Series(cos_how, index=g.index, dtype=float),
            "_imputed_funding_now": fund_imp.flags.reindex(g.index).fillna(0).astype(int),
            "_imputed_oi_now": oi_imp.flags.reindex(g.index).fillna(0).astype(int),
        }, index=g.index)

        # Drop warmup period per symbol
        mask = (np.arange(len(out)) >= warmup)
        out = out.loc[out.index[mask]]
        out = out.reset_index().rename(columns={"ts": "ts"})
        feats.append(out)

    feat = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()

    # Stable schema and cleanup
    # Replace NaNs with 0 for numeric columns except imputed flags already 0/1
    for col in feat.columns:
        if col in {"ts", "symbol"}:
            continue
        if feat[col].dtype.kind in {"f", "i"}:
            feat[col] = feat[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Ensure no NaN rows
    feat = feat.dropna(axis=0, how="any")

    # Enforce unique key and sort
    ensure_unique_key(feat, ["symbol", "ts"])
    feat = feat.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return feat


@app.command("build")
def build(
    glob: str = typer.Option(..., "--glob", help="Input 5m P1 Parquet glob"),
    out: str = typer.Option(..., "--out", help="Output features root directory"),
    warmup: int = typer.Option(8640, "--warmup", help="Warmup bars (default 30d of 5m = 8640)"),
    winsor_low: float = typer.Option(0.01, "--winsor_low", help="Causal winsorize lower quantile"),
    winsor_high: float = typer.Option(0.99, "--winsor_high", help="Causal winsorize upper quantile"),
) -> None:
    t0 = time.perf_counter()
    raw = read_parquet_glob(glob)
    if raw.empty:
        typer.echo("No input rows found; aborting.")
        raise typer.Exit(code=1)
    feat = build_features(raw, warmup=warmup, winsor_low=winsor_low, winsor_high=winsor_high)
    if feat.empty:
        typer.echo("No feature rows produced after warmup and cleaning; aborting.")
        raise typer.Exit(code=1)
    write_parquet_daily(feat, out)
    dt = time.perf_counter() - t0
    typer.echo(f"Built and wrote {len(feat):,} rows in {dt:.2f}s ({dt*1e3/len(feat):.3f} ms/bar)")


@app.command("validate")
def validate(
    glob: str = typer.Option(..., "--glob", help="Features Parquet glob"),
    qa: str = typer.Option("reports/p2_qa_5m_80d.json", "--qa", help="QA report JSON path"),
    schema: str = typer.Option("reports/p2_feature_schema.json", "--schema", help="Output schema JSON path"),
    horizon_days: int = typer.Option(80, "--horizon_days", help="Window (days) to validate; default 80"),
    warmup: int = typer.Option(8640, "--warmup", help="Warmup bars to ignore at head; default 8640"),
) -> None:
    df = read_parquet_glob(glob)
    if df.empty:
        typer.echo("No features found; failing QA.")
        Path(qa).parent.mkdir(parents=True, exist_ok=True)
        with open(qa, "w") as f:
            json.dump({"error": "no_data"}, f, indent=2)
        raise typer.Exit(code=1)

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    ts_end = df["ts"].max()
    # Use last (horizon_days - 30) days for acceptance since warmup=30d
    usable_days = max(0, horizon_days - 30)
    start_usable = ts_end - pd.Timedelta(days=usable_days)
    expected_idx = pd.date_range(start_usable, ts_end, freq="5min")
    window = df[(df["ts"] >= expected_idx[0]) & (df["ts"] <= expected_idx[-1])].copy()

    # Keys and basic counts
    dup_key_count = int(window.duplicated(subset=["symbol", "ts"]).sum())
    present_bars = int(window["ts"].nunique())
    expected_bars = int(len(expected_idx))
    gap_ratio = float(1 - present_bars / expected_bars) if expected_bars else 1.0

    # Per-column stats
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
        "usable_expected_bars_50d": int(50 * 24 * 12) if horizon_days == 80 else expected_bars,
        "present_bars_after_warmup": present_bars,
        "gap_ratio_after_warmup": gap_ratio,
        "dup_key_count": dup_key_count,
        "per_column": per_column,
    }
    Path(qa).parent.mkdir(parents=True, exist_ok=True)
    with open(qa, "w") as f:
        json.dump(report, f, indent=2)

    # Emit feature schema
    schema_dict = {c: str(df[c].dtype) for c in sorted(df.columns)}
    Path(schema).parent.mkdir(parents=True, exist_ok=True)
    with open(schema, "w") as f:
        json.dump(schema_dict, f, indent=2)

    # Acceptance gate
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


@app.command("bench")
def bench(
    glob: str = typer.Option(..., "--glob", help="Features Parquet glob"),
    report: str = typer.Option("reports/p2_bench_5m_80d.csv", "--report", help="CSV path for benchmark results"),
) -> None:
    t0 = time.perf_counter()
    df = read_parquet_glob(glob)
    dt = time.perf_counter() - t0
    rows = int(len(df))
    ms_per_bar = (dt * 1000.0 / rows) if rows else 0.0
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w") as f:
        f.write("rows,seconds,ms_per_bar\n")
        f.write(f"{rows},{dt:.6f},{ms_per_bar:.6f}\n")
    if ms_per_bar > 50.0:
        typer.echo(f"Benchmark failed: {ms_per_bar:.3f} ms/bar > 50")
        raise typer.Exit(code=1)
    typer.echo(f"Benchmark: {rows} rows in {dt:.3f}s ⇒ {ms_per_bar:.3f} ms/bar")


if __name__ == "__main__":
    app()
