from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd
import typer

from utils_cg import write_parquet_daily_files


app = typer.Typer(help="P7 – EV gate policy with ATR% barriers and trade discipline")


def _read_parquet(glob: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT * FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _dynamic_k(vol_pct: float, k_min: float, k_max: float) -> float:
    vol_pct = float(np.clip(vol_pct, 0.0, 1.0))
    return float(k_min + (k_max - k_min) * vol_pct)


@dataclass
class CostModel:
    bps: float = 5.0

    def cost_frac(self) -> float:
        return self.bps / 1e4


def _compute_vol_pctile(atr_df: pd.DataFrame) -> pd.DataFrame:
    if "vol_pctile" in atr_df.columns:
        return atr_df
    # Global percentile rank of atr_pct per symbol
    out = []
    for sym, g in atr_df.groupby(atr_df.get("symbol", pd.Series(["BTCUSDT"] * len(atr_df)))):
        r = g["atr_pct"].rank(method="average", pct=True)
        gg = g.copy()
        gg["vol_pctile"] = r.astype(float).to_numpy()
        out.append(gg)
    return pd.concat(out, ignore_index=True)


def _ev_decision_row(p_long: float, p_short: float, close: float, atr_pct: float, k_min: float, k_max: float, vol_pct: float, cost: CostModel) -> Tuple[str, float, float, float, float, str]:
    # Determine k by vol percentile
    k = _dynamic_k(vol_pct, k_min, k_max)
    tp = k * atr_pct
    sl = k * atr_pct
    ev_long = p_long * tp - (1.0 - p_long) * sl - cost.cost_frac()
    ev_short = p_short * tp - (1.0 - p_short) * sl - cost.cost_frac()
    if max(ev_long, ev_short) <= 0:
        return "WAIT", 0.0, np.nan, np.nan, 0.0, "ev<=0"
    side = "LONG" if ev_long >= ev_short else "SHORT"
    max_prob = max(p_long, p_short)
    size = float(np.clip(max_prob, 0.2, 1.0))
    if side == "LONG":
        tp_px = close * (1.0 + tp)
        sl_px = close * (1.0 - sl)
        ev = ev_long
    else:
        tp_px = close * (1.0 - tp)
        sl_px = close * (1.0 + sl)
        ev = ev_short
    return side, size, float(tp_px), float(sl_px), float(ev), "enter"


@app.command("decide")
def decide(
    probs: str = typer.Option(..., "--probs", help="Calibrated probabilities parquet with columns ts,p_0,p_1,p_2[,symbol] (WAIT,LONG,SHORT)"),
    atr: str = typer.Option(..., "--atr", help="ATR parquet with columns ts,close,atr_pct[,vol_pctile][,symbol]"),
    k_range_min: float = typer.Option(1.0, "--k-range-min"),
    k_range_max: float = typer.Option(1.5, "--k-range-max"),
    H: int = typer.Option(36, "--H", help="H bars (no re-entry cooldown)"),
    out: str = typer.Option("decisions", "--out"),
) -> None:
    dfp = _read_parquet(probs)
    dfa = _read_parquet(atr)
    dfp = dfp.rename(columns={"p_WAIT": "p_0", "p_LONG": "p_1", "p_SHORT": "p_2"})
    if not {"p_0", "p_1", "p_2"}.issubset(dfp.columns):
        raise typer.BadParameter("probs file must include p_0,p_1,p_2 (WAIT,LONG,SHORT)")
    # Join on ts (+symbol if present)
    keys = ["ts"] + (["symbol"] if "symbol" in dfp.columns and "symbol" in dfa.columns else [])
    dfa = _compute_vol_pctile(dfa)
    df = pd.merge(dfp, dfa, on=keys, how="inner")
    df = df.sort_values(keys).reset_index(drop=True)

    dt = pd.Timedelta(minutes=5)
    cooldown = H * dt
    last_entry_ts: pd.Timestamp | None = None
    entry_side: str | None = None
    entry_close: float = np.nan
    entry_tp_px: float = np.nan
    max_close_since_entry: float = -np.inf
    min_close_since_entry: float = np.inf
    early_exit_marked: bool = False
    follow_frac = 0.10  # require at least 10% of TP distance within 3 bars
    rows = []
    cm = CostModel(bps=5.0)
    for _, r in df.iterrows():
        ts = r["ts"]
        close = float(r.get("close", np.nan))
        if not np.isfinite(close):
            rows.append({"ts": ts, "side": "WAIT", "size": 0.0, "TP_px": np.nan, "SL_px": np.nan, "EV": 0.0, "reason": "no_close"})
            continue
        # Cooldown (no re-entry within H); track early-exit at 3 bars if no follow-through
        if last_entry_ts is not None and (ts - last_entry_ts) < cooldown:
            # update follow-through stats for first 3 bars
            bars_elapsed = int(((ts - last_entry_ts) / dt) + 1e-9)
            reason = "cooldown"
            if 1 <= bars_elapsed <= 3 and np.isfinite(close):
                max_close_since_entry = max(max_close_since_entry, close)
                min_close_since_entry = min(min_close_since_entry, close)
            if bars_elapsed == 3 and not early_exit_marked and entry_side is not None:
                if entry_side == "LONG":
                    dist = entry_tp_px - entry_close
                    moved = max(0.0, max_close_since_entry - entry_close)
                else:
                    dist = entry_close - entry_tp_px
                    moved = max(0.0, entry_close - min_close_since_entry)
                if dist > 0 and moved < follow_frac * dist:
                    reason = "early_exit"
                early_exit_marked = True
            rows.append({"ts": ts, "side": "WAIT", "size": 0.0, "TP_px": np.nan, "SL_px": np.nan, "EV": 0.0, "reason": reason})
            continue
        side, size, tp_px, sl_px, ev, reason = _ev_decision_row(
            float(r["p_1"]), float(r["p_2"]), close, float(r["atr_pct"]), k_range_min, k_range_max, float(r.get("vol_pctile", 0.5)), cm
        )
        rows.append({"ts": ts, "side": side, "size": size, "TP_px": tp_px, "SL_px": sl_px, "EV": ev, "reason": reason})
        if side != "WAIT":
            last_entry_ts = ts
            entry_side = side
            entry_close = close
            entry_tp_px = tp_px
            max_close_since_entry = close
            min_close_since_entry = close
            early_exit_marked = False
    dec = pd.DataFrame(rows)
    # Persist partitioned by day
    dec["symbol"] = df.get("symbol", pd.Series(["BTCUSDT"] * len(dec)))
    write_parquet_daily_files(dec, out, str(dec["symbol"].iloc[0]))
    typer.echo(f"Decisions written under {out}")


if __name__ == "__main__":
    app()
