from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd
import typer

from transforms import winsorize_causal
from utils_cg import ensure_unique_key, write_parquet_daily_files


app = typer.Typer(help="P3 – Triple-Barrier labeling for 5m BTCUSDT")


def _read_parquet_glob(glob: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    sel = "*" if not cols else ",".join(cols)
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT {sel} FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _compute_atr_pct(g: pd.DataFrame, atr_window: int) -> pd.Series:
    # Ensure numeric
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    close = g["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.shift(1).rolling(atr_window, min_periods=atr_window).mean()
    atrpct = (atr / prev_close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    # Causal winsorization to stabilize tails; use same window
    atrpct = winsorize_causal(atrpct.fillna(0.0), window=atr_window, low=0.01, high=0.99)
    return atrpct


@dataclass
class TBParams:
    k: float = 1.2
    H: int = 36
    atr_window: int = 14


def _label_symbol(g: pd.DataFrame, params: TBParams) -> pd.DataFrame:
    g = g.sort_values("ts").reset_index(drop=True)
    atrpct = _compute_atr_pct(g, params.atr_window)
    close = g["close"].astype(float)
    open_ = g["open"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)

    up_px = close * (1.0 + params.k * atrpct)
    dn_px = close * (1.0 - params.k * atrpct)

    n = len(g)
    labels: List[str] = ["WAIT"] * n
    tp_px: List[float] = [np.nan] * n
    sl_px: List[float] = [np.nan] * n
    timeout_ts: List[pd.Timestamp] = [pd.NaT] * n

    # Iterate over decision bars where we have future horizon
    for i in range(n - params.H):
        # Require ATR available at i (not NaN) and prev close exists
        if not np.isfinite(up_px.iat[i]) or not np.isfinite(dn_px.iat[i]):
            continue
        up = float(up_px.iat[i])
        dn = float(dn_px.iat[i])
        tp_px[i] = up
        sl_px[i] = dn
        label = "WAIT"
        for j in range(1, params.H + 1):
            oj = float(open_.iat[i + j])
            hj = float(high.iat[i + j])
            lj = float(low.iat[i + j])
            up_cross = (oj >= up) or (hj >= up)
            dn_cross = (oj <= dn) or (lj <= dn)
            if up_cross or dn_cross:
                if up_cross and dn_cross:
                    label = "WAIT"
                elif up_cross:
                    label = "LONG"
                else:
                    label = "SHORT"
                break
        labels[i] = label
        timeout_ts[i] = g["ts"].iat[i + params.H]

    out = pd.DataFrame(
        {
            "ts": g["ts"],
            "symbol": g["symbol"],
            "label": labels,
            "tp_px": tp_px,
            "sl_px": sl_px,
            "timeout_ts": timeout_ts,
        }
    )
    # Drop rows without barriers (NaN tp/sl or timeout)
    out = out.dropna(subset=["tp_px", "sl_px", "timeout_ts"])
    # No NaNs in outputs
    out = out.fillna({"label": "WAIT"})
    return out


@app.command("triple-barrier")
def triple_barrier(
    features: str = typer.Option(..., "--features", help="Features Parquet glob (will be joined with raw if OHLC missing)"),
    out: str = typer.Option(..., "--out", help="Output labels root"),
    tf: str = typer.Option("5m", "--tf", help="Timeframe; must be 5m"),
    k: float = typer.Option(1.2, "--k"),
    H: int = typer.Option(36, "--H", help="Horizon in bars"),
    atr_window: int = typer.Option(14, "--atr_window"),
    raw: Optional[str] = typer.Option(None, "--raw", help="Optional raw P1 Parquet glob to supply OHLC if features lack them"),
) -> None:
    if tf.lower() != "5m":
        raise typer.BadParameter("Only 5m timeframe is supported")
    typer.echo(f"[P3:label] features={features} raw={raw or 'None'} out={out} tf={tf} k={k} H={H} atr_window={atr_window}")
    cols_req = ["ts", "symbol", "open", "high", "low", "close"]
    df = _read_parquet_glob(features)
    # If OHLC missing and raw provided, join OHLC from raw lake
    need_join = any(c not in df.columns for c in ["open", "high", "low", "close"])
    if need_join:
        if not raw:
            raise typer.BadParameter("Features lack OHLC and --raw not provided")
        ohlc = _read_parquet_glob(raw, cols=["ts", "symbol", "open", "high", "low", "close"])
        if ohlc.empty:
            raise typer.BadParameter("Raw parquet empty or OHLC not found")
        df = df.merge(ohlc, on=["ts", "symbol"], how="inner")
    missing = [c for c in cols_req if c not in df.columns]
    if missing:
        raise typer.BadParameter(f"Missing required columns after join: {missing}")
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    out_root = out.rstrip("/")
    Path(out_root).mkdir(parents=True, exist_ok=True)
    params = TBParams(k=k, H=H, atr_window=atr_window)

    parts: List[pd.DataFrame] = []
    for sym, grp in df.groupby("symbol", sort=False):
        labeled = _label_symbol(grp, params)
        if not labeled.empty:
            parts.append(labeled)
    labels_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if labels_df.empty:
        typer.echo("No labels produced; aborting.")
        raise typer.Exit(code=1)

    # Ensure unique key and no NaNs
    ensure_unique_key(labels_df, ["symbol", "ts"])
    if labels_df[["label", "tp_px", "sl_px", "timeout_ts"]].isna().any().any():
        typer.echo("NaN found in output columns; failing.")
        raise typer.Exit(code=1)

    # Write partitioned by day
    write_parquet_daily_files(labels_df, out_root, str(labels_df["symbol"].iloc[0]))
    typer.echo(f"Wrote labels to {out_root}")


@app.command("validate")
def validate(
    labels: str = typer.Option(..., "--labels"),
    features: str = typer.Option(..., "--features"),
    report: str = typer.Option("reports/p3_qa_5m_80d.json", "--report"),
) -> None:
    lab = _read_parquet_glob(labels)
    if lab.empty:
        typer.echo("No labels found; failing QA.")
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        with open(report, "w") as f:
            json.dump({"error": "no_labels"}, f, indent=2)
        raise typer.Exit(code=1)
    ensure_unique_key(lab, ["symbol", "ts"])
    # Join with features by (symbol, ts)
    con = duckdb.connect()
    try:
        # DuckDB doesn't allow parameters in DDL; embed paths directly (trusted local paths)
        con.execute(f"CREATE OR REPLACE VIEW _lab AS SELECT * FROM read_parquet('{labels}')")
        con.execute(f"CREATE OR REPLACE VIEW _feat AS SELECT * FROM read_parquet('{features}')")
        lab_cnt = con.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM _lab").fetchone()
        join_cnt = con.execute("SELECT COUNT(*) FROM _lab l INNER JOIN _feat f USING(symbol, ts)").fetchone()[0]
        hist = dict(con.execute("SELECT label, COUNT(*) FROM _lab GROUP BY 1 ORDER BY 1").fetchall())
    finally:
        con.close()

    total = int(lab_cnt[0])
    wait_ratio = float(hist.get("WAIT", 0) / total) if total else 0.0
    report_obj = {
        "labels": {"count": total, "min_ts": str(lab_cnt[1]), "max_ts": str(lab_cnt[2])},
        "join_inner_count": int(join_cnt),
        "class_hist": hist,
        "wait_ratio": wait_ratio,
        "warn_wait_ratio_gt_0_8": wait_ratio > 0.8,
    }
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w") as f:
        json.dump(report_obj, f, indent=2)
    if wait_ratio > 0.8:
        typer.echo("WARNING: WAIT ratio exceeds 80% of labels.")
    typer.echo("Labels validated.")


@app.command("sample")
def sample(
    labels: str = typer.Option(..., "--labels"),
    n: int = typer.Option(10, "--n"),
) -> None:
    lab = _read_parquet_glob(labels)
    if lab.empty:
        typer.echo("No labels to sample.")
        return
    print(lab.sort_values("ts").head(n).to_string(index=False))


if __name__ == "__main__":
    app()
