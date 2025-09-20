import pandas as pd
import numpy as np

from app.policy.ev_policy import decide, step, BarrierCfg
from app.policy.state import PolicyState


def _row(ts, close=100.0, atr_pct=0.01, vol_pctile=0.5, liq_pctile=0.0):
    return pd.Series(
        {
            "ts": pd.Timestamp(ts, tz="UTC"),
            "symbol": "BTCUSDT",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "atr_pct": atr_pct,
            "vol_pctile": vol_pctile,
            "liq60_z_pctile": liq_pctile,
        }
    )


def test_tau_filter():
    state = PolicyState()
    cfg = {"tau_long": 0.7, "tau_short": 0.7, "barrier": BarrierCfg()}
    row = _row("2025-09-20 00:00:00")
    d = decide(row, {"p_long": 0.68, "p_short": 0.10, "p_wait": 0.22}, state, cfg)
    assert d["side"] == "WAIT" and d["reason"] == "TAU"


def test_ev_gate_positive_long():
    state = PolicyState()
    cfg = {"tau_long": 0.55, "tau_short": 0.55, "barrier": BarrierCfg()}
    row = _row("2025-09-20 00:05:00", close=100.0, atr_pct=0.02, vol_pctile=0.2, liq_pctile=0.0)
    d = decide(row, {"p_long": 0.75, "p_short": 0.10, "p_wait": 0.15}, state, cfg)
    assert d["side"] == "LONG"
    assert d["TP_px"] > d["SL_px"] > 0


def test_costs_applied_flip_to_wait():
    state = PolicyState()
    cfg = {"tau_long": 0.55, "tau_short": 0.55, "barrier": BarrierCfg(), "half_spread_bps": 5.0}
    # With very high costs via liquidity percentile, EV should be <= 0
    row = _row("2025-09-20 00:10:00", close=100.0, atr_pct=0.002, vol_pctile=0.2, liq_pctile=1.0)
    d = decide(row, {"p_long": 0.65, "p_short": 0.10, "p_wait": 0.25}, state, cfg)
    assert d["side"] == "WAIT" and d["reason"].startswith("EV")


def test_no_overlap_then_cooldown():
    # Enter, then next bar should be OVERLAP; after early exit at 3 bars, COOLDOWN applies
    state = PolicyState()
    cfg = {"tau_long": 0.55, "tau_short": 0.55, "barrier": BarrierCfg(horizon_bars=8)}
    probs = {"p_long": 0.8, "p_short": 0.05, "p_wait": 0.15}

    # t0: enter
    d0 = decide(_row("2025-09-20 00:00:00"), probs, state, cfg)
    assert d0["side"] in ("LONG", "SHORT")
    # t1: still overlapping position
    d1 = decide(_row("2025-09-20 00:05:00"), probs, state, cfg)
    assert d1["side"] == "WAIT" and d1["reason"] == "OVERLAP"
    # t2,t3: progress bars to trigger early-exit (no favorable move and no barrier)
    decide(_row("2025-09-20 00:10:00"), probs, state, cfg)
    d3 = decide(_row("2025-09-20 00:15:00"), probs, state, cfg)
    assert d3["side"] == "WAIT" and d3["reason"] == "EARLY_EXIT_NO_FT"
    # Next bar: cooldown enforced
    d4 = decide(_row("2025-09-20 00:20:00"), probs, state, cfg)
    assert d4["side"] == "WAIT" and d4["reason"] == "COOLDOWN"


def test_outputs_schema_with_step():
    state = PolicyState()
    cfg = {"tau_long": 0.55, "tau_short": 0.55, "barrier": BarrierCfg()}
    ts = pd.date_range("2025-09-20 01:00:00", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "ts": ts,
            "symbol": ["BTCUSDT"] * len(ts),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "atr_pct": 0.01,
            "vol_pctile": 0.5,
            "liq60_z_pctile": 0.2,
        }
    )
    probs = pd.DataFrame(
        {
            "ts": ts,
            "symbol": ["BTCUSDT"] * len(ts),
            "p_long": [0.8, 0.6, 0.6, 0.6, 0.6],
            "p_short": [0.1] * 5,
            "p_wait": [0.1] * 5,
        }
    )
    out = step(df, probs, state, cfg)
    required_cols = {
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
    }
    assert required_cols.issubset(out.columns)
    assert out.shape[0] == len(ts)
    # No NaN in required columns
    assert not out[list(required_cols)].isna().any().any()

