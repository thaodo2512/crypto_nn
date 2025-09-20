from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path
import yaml

import numpy as np
import pandas as pd

from .state import PolicyState
from .io import write_decisions


@dataclass
class BarrierCfg:
    horizon_bars: int = 36
    k_atr_min: float = 1.0
    k_atr_max: float = 1.5


def _dynamic_k(vol_pctile: float, cfg: BarrierCfg) -> float:
    v = float(np.clip(vol_pctile, 0.0, 1.0))
    return float(cfg.k_atr_min + (cfg.k_atr_max - cfg.k_atr_min) * v)


def load_barrier_cfg(path: str = "conf/barrier.yaml") -> BarrierCfg:
    p = Path(path)
    if not p.exists():
        return BarrierCfg()
    try:
        data = yaml.safe_load(p.read_text()) or {}
        return BarrierCfg(
            horizon_bars=int(data.get("horizon_bars", 36)),
            k_atr_min=float(data.get("k_atr_min", 1.0)),
            k_atr_max=float(data.get("k_atr_max", 1.5)),
        )
    except Exception:
        return BarrierCfg()


def _impact_bps(liq60_pctile: float) -> float:
    return float(0.3 + 0.4 * float(np.clip(liq60_pctile, 0.0, 1.0)))


def decide(row: pd.Series, probs: Dict[str, float], state: PolicyState, cfg: Dict) -> Dict:
    """Single-row EV-gated decision.

    Inputs
    - row: must include ts, symbol, open, high, low, close, atr_pct, vol_pctile, liq60_z_pctile
    - probs: calibrated probabilities with keys p_long, p_short, p_wait
    - state: PolicyState (mutated on entry/exit)
    - cfg: dict with keys tau_long, tau_short, barrier (BarrierCfg) and half_spread_bps
    """
    ts = pd.to_datetime(row["ts"], utc=True)
    symbol = str(row.get("symbol", "BTCUSDT"))
    close = float(row["close"])
    high = float(row.get("high", close))
    low = float(row.get("low", close))
    atr_pct = float(row["atr_pct"])  # fraction (e.g., 0.004)
    vol_pct = float(row.get("vol_pctile", 0.5))
    liq_pct = float(row.get("liq60_z_pctile", 0.0))

    tau_long = float(cfg.get("tau_long", 0.55))
    tau_short = float(cfg.get("tau_short", 0.55))
    barrier: BarrierCfg = cfg.get("barrier", BarrierCfg())
    half_spread_bps = float(cfg.get("half_spread_bps", 0.5))
    H = int(barrier.horizon_bars)

    # Exit handling if position is open (TP/SL or early-exit progression)
    if state.has_position:
        state.bars_since_entry += 1
        # Check barrier hits on current bar
        if state.side == "LONG":
            if low <= state.sl_px:
                # Exit at SL
                state.reset_position()
                return {
                    "ts": ts,
                    "symbol": symbol,
                    "side": "WAIT",
                    "size": 0.0,
                    "TP_px": 0.0,
                    "SL_px": 0.0,
                    "EV_bps": 0.0,
                    "k_atr": _dynamic_k(vol_pct, barrier),
                    "vol_pctile": vol_pct,
                    "liq60_pctile": liq_pct,
                    "reason": "EXIT_SL",
                    "tau_long": tau_long,
                    "tau_short": tau_short,
                }
            if high >= state.tp_px:
                state.reset_position()
                return {
                    "ts": ts,
                    "symbol": symbol,
                    "side": "WAIT",
                    "size": 0.0,
                    "TP_px": 0.0,
                    "SL_px": 0.0,
                    "EV_bps": 0.0,
                    "k_atr": _dynamic_k(vol_pct, barrier),
                    "vol_pctile": vol_pct,
                    "liq60_pctile": liq_pct,
                    "reason": "EXIT_TP",
                    "tau_long": tau_long,
                    "tau_short": tau_short,
                }
        else:  # SHORT
            if high >= state.sl_px:
                state.reset_position()
                return {
                    "ts": ts,
                    "symbol": symbol,
                    "side": "WAIT",
                    "size": 0.0,
                    "TP_px": 0.0,
                    "SL_px": 0.0,
                    "EV_bps": 0.0,
                    "k_atr": _dynamic_k(vol_pct, barrier),
                    "vol_pctile": vol_pct,
                    "liq60_pctile": liq_pct,
                    "reason": "EXIT_SL",
                    "tau_long": tau_long,
                    "tau_short": tau_short,
                }
            if low <= state.tp_px:
                state.reset_position()
                return {
                    "ts": ts,
                    "symbol": symbol,
                    "side": "WAIT",
                    "size": 0.0,
                    "TP_px": 0.0,
                    "SL_px": 0.0,
                    "EV_bps": 0.0,
                    "k_atr": _dynamic_k(vol_pct, barrier),
                    "vol_pctile": vol_pct,
                    "liq60_pctile": liq_pct,
                    "reason": "EXIT_TP",
                    "tau_long": tau_long,
                    "tau_short": tau_short,
                }

        # Early exit at 3 bars without favorable move
        if state.bars_since_entry >= 3:
            if state.side == "LONG":
                moved = max(0.0, close - state.entry_close)
                dist = max(1e-12, state.tp_px - state.entry_close)
            else:
                moved = max(0.0, state.entry_close - close)
                dist = max(1e-12, state.entry_close - state.tp_px)
            if moved < 0.0 + 0.0 * dist:  # "≥ 0 ticks" condition → moved must be > 0 to keep; else exit
                state.reset_position()
                return {
                    "ts": ts,
                    "symbol": symbol,
                    "side": "WAIT",
                    "size": 0.0,
                    "TP_px": 0.0,
                    "SL_px": 0.0,
                    "EV_bps": 0.0,
                    "k_atr": _dynamic_k(vol_pct, barrier),
                    "vol_pctile": vol_pct,
                    "liq60_pctile": liq_pct,
                    "reason": "EARLY_EXIT_NO_FT",
                    "tau_long": tau_long,
                    "tau_short": tau_short,
                }

        # Overlap prevention while position is open
        return {
            "ts": ts,
            "symbol": symbol,
            "side": "WAIT",
            "size": 0.0,
            "TP_px": 0.0,
            "SL_px": 0.0,
            "EV_bps": 0.0,
            "k_atr": _dynamic_k(vol_pct, barrier),
            "vol_pctile": vol_pct,
            "liq60_pctile": liq_pct,
            "reason": "OVERLAP",
            "tau_long": tau_long,
            "tau_short": tau_short,
        }

    # Cooldown gate
    if state.last_entry_ts is not None:
        dt = pd.Timedelta(minutes=5)
        if (ts - state.last_entry_ts) < barrier.horizon_bars * dt:
            return {
                "ts": ts,
                "symbol": symbol,
                "side": "WAIT",
                "size": 0.0,
                "TP_px": np.nan,
                "SL_px": np.nan,
                "EV_bps": 0.0,
                "k_atr": _dynamic_k(vol_pct, barrier),
                "vol_pctile": vol_pct,
                "liq60_pctile": liq_pct,
                "reason": "COOLDOWN",
                "tau_long": tau_long,
                "tau_short": tau_short,
            }

    # Threshold pre-filter based on side
    p_long = float(probs.get("p_long", np.nan))
    p_short = float(probs.get("p_short", np.nan))
    side_pref = "LONG" if p_long >= p_short else "SHORT"
    tau_side = tau_long if side_pref == "LONG" else tau_short
    max_prob = max(p_long, p_short)
    if max_prob < tau_side:
        return {
            "ts": ts,
            "symbol": symbol,
            "side": "WAIT",
            "size": 0.0,
            "TP_px": 0.0,
            "SL_px": 0.0,
            "EV_bps": 0.0,
            "k_atr": _dynamic_k(vol_pct, barrier),
            "vol_pctile": vol_pct,
            "liq60_pctile": liq_pct,
            "reason": "TAU",
            "tau_long": tau_long,
            "tau_short": tau_short,
        }

    # EV gating with costs
    k = _dynamic_k(vol_pct, barrier)
    tp_pct = max(0.0, k * atr_pct)
    sl_pct = tp_pct
    cost_bps = half_spread_bps + _impact_bps(liq_pct)
    cost_frac = cost_bps / 1e4
    ev_long = p_long * tp_pct - (1.0 - p_long) * sl_pct - cost_frac
    ev_short = p_short * tp_pct - (1.0 - p_short) * sl_pct - cost_frac
    ev_frac = max(ev_long, ev_short)
    if ev_frac <= 0.0:
        return {
            "ts": ts,
            "symbol": symbol,
            "side": "WAIT",
            "size": 0.0,
            "TP_px": 0.0,
            "SL_px": 0.0,
            "EV_bps": ev_frac * 1e4,
            "k_atr": k,
            "vol_pctile": vol_pct,
            "liq60_pctile": liq_pct,
            "reason": "EV≤0",
            "tau_long": tau_long,
            "tau_short": tau_short,
        }

    side = "LONG" if ev_long >= ev_short else "SHORT"
    size = float(np.clip(max_prob, 0.2, 1.0))
    if side == "LONG":
        tp_px = close * (1.0 + tp_pct)
        sl_px = close * (1.0 - sl_pct)
    else:
        tp_px = close * (1.0 - tp_pct)
        sl_px = close * (1.0 + sl_pct)

    # Mutate state on entry
    state.has_position = True
    state.side = side
    state.entry_close = close
    state.tp_px = float(tp_px)
    state.sl_px = float(sl_px)
    state.bars_since_entry = 0
    state.last_entry_ts = ts  # cooldown starts at entry

    return {
        "ts": ts,
        "symbol": symbol,
        "side": side,
        "size": size,
        "TP_px": float(tp_px),
        "SL_px": float(sl_px),
        "EV_bps": ev_frac * 1e4,
        "k_atr": k,
        "vol_pctile": vol_pct,
        "liq60_pctile": liq_pct,
        "reason": "ENTER",
        "tau_long": tau_long,
        "tau_short": tau_short,
    }


def step(batch_df: pd.DataFrame, probs_df: pd.DataFrame, state: PolicyState, cfg: Dict, out_path: Optional[str] = None) -> pd.DataFrame:
    """Run policy over a batch (time-ordered) and optionally persist.

    - batch_df must contain columns: ts,symbol,open,high,low,close,atr_pct,vol_pctile,liq60_z_pctile
    - probs_df must contain columns: ts,symbol,p_long,p_short,p_wait
    - cfg contains tau_long,tau_short,barrier,half_spread_bps
    """
    df = pd.merge(
        batch_df.copy(),
        probs_df[["ts", "symbol", "p_long", "p_short", "p_wait"]].copy(),
        on=["ts", "symbol"],
        how="inner",
    ).sort_values(["ts", "symbol"]).reset_index(drop=True)

    decisions = []
    for _, r in df.iterrows():
        d = decide(r, {"p_long": r["p_long"], "p_short": r["p_short"], "p_wait": r["p_wait"]}, state, cfg)
        decisions.append(d)

    out = pd.DataFrame(decisions)
    # Ensure schema and no nulls in required fields
    out["symbol"] = out["symbol"].astype(str)
    required = [
        "ts",
        "symbol",
        "side",
        "size",
        "TP_px",
        "SL_px",
        "EV_bps",
        "k_atr",
        "vol_pctile",
        "liq60_pctile",
        "reason",
        "tau_long",
        "tau_short",
    ]
    for c in required:
        if c not in out.columns:
            out[c] = np.nan

    if out_path:
        write_decisions(out, out_path)

    return out
