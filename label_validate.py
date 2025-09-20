from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import typer

from transforms import winsorize_causal


app = typer.Typer(help="P3 Validator – Validate Triple-Barrier labels for 5m BTCUSDT (last 80 days)")


ALLOWED_LABELS = {"LONG", "SHORT", "WAIT"}
EXPECTED_80D_BARS = 80 * 24 * 12  # 23040
USABLE_50D_BARS = 50 * 24 * 12  # 14400


def _read_parquet(glob: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
    sel = "*" if not cols else ",".join(cols)
    con = duckdb.connect()
    try:
        df = con.execute(f"SELECT {sel} FROM read_parquet('{glob}')").df()
    finally:
        con.close()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _ensure_ohlc(features: pd.DataFrame, raw_glob: Optional[str]) -> pd.DataFrame:
    need = any(c not in features.columns for c in ["open", "high", "low", "close"])
    if not need:
        return features
    if not raw_glob:
        raise typer.BadParameter("Features are missing OHLC and --raw is not provided")
    raw = _read_parquet(raw_glob, ["ts", "symbol", "open", "high", "low", "close"])
    if raw.empty:
        raise typer.BadParameter("Raw parquet is empty or lacks OHLC columns")
    merged = features.merge(raw, on=["ts", "symbol"], how="inner")
    missing_after = [c for c in ["open", "high", "low", "close"] if c not in merged.columns]
    if missing_after:
        raise typer.BadParameter(f"Failed to obtain OHLC columns: {missing_after}")
    return merged


def _grid_5m(min_ts: pd.Timestamp, days: int) -> pd.DatetimeIndex:
    start = pd.to_datetime(min_ts, utc=True).floor("5min")
    return pd.date_range(start=start, periods=days * 24 * 12, freq="5min")


def _compute_atr_pct(g: pd.DataFrame, atr_window: int) -> pd.Series:
    # Past-only TR and ATR
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
    atrpct = (atr / prev_close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    atrpct = winsorize_causal(atrpct, window=atr_window, low=0.01, high=0.99)
    return atrpct.astype(float)


def _sample_rulecheck(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    H: int,
    k: float,
    atr_window: int,
    sample_size: int = 500,
    seed: int = 42,
) -> float:
    """Return mismatch ratio between labels and recomputed barrier outcomes for a random sample."""
    rng = random.Random(seed)
    # Organize features per symbol
    feat_by_sym: Dict[str, pd.DataFrame] = {}
    for sym, grp in features.groupby("symbol", sort=False):
        g = grp.sort_values("ts").reset_index(drop=True)
        g["atrpct"] = _compute_atr_pct(g, atr_window)
        feat_by_sym[str(sym)] = g

    # Candidate indices where there are at least H future bars
    candidates: List[Tuple[str, pd.Timestamp]] = []
    for sym, g in feat_by_sym.items():
        # We will only check timestamps that exist in both labels and features
        label_ts_sym = labels.loc[labels["symbol"] == sym, "ts"]
        if label_ts_sym.empty:
            continue
        # Use set for speed
        ts_set = set(pd.to_datetime(label_ts_sym, utc=True).tolist())
        # Ensure future availability
        for i in range(0, len(g) - H - 1):
            t = g.loc[i, "ts"]
            if t in ts_set:
                candidates.append((sym, t))

    if not candidates:
        return 0.0

    picks = rng.sample(candidates, k=min(sample_size, len(candidates)))
    mismatches = 0
    for sym, t in picks:
        g = feat_by_sym[sym]
        i = int(g.index[g["ts"] == t][0])  # position
        close_t = float(g.loc[i, "close"]) if "close" in g.columns else np.nan
        atrpct_t = float(g.loc[i, "atrpct"]) if "atrpct" in g.columns else 0.0
        up = close_t * (1.0 + k * atrpct_t)
        dn = close_t * (1.0 - k * atrpct_t)
        exp_label = "WAIT"
        # scan future
        for j in range(1, H + 1):
            oj = float(g.loc[i + j, "open"]) if "open" in g.columns else np.nan
            hj = float(g.loc[i + j, "high"]) if "high" in g.columns else np.nan
            lj = float(g.loc[i + j, "low"]) if "low" in g.columns else np.nan
            up_cross = (oj >= up) or (hj >= up)
            dn_cross = (oj <= dn) or (lj <= dn)
            if up_cross or dn_cross:
                if up_cross and dn_cross:
                    exp_label = "WAIT"
                elif up_cross:
                    exp_label = "LONG"
                else:
                    exp_label = "SHORT"
                break
        got_label = str(labels.set_index(["symbol", "ts"]).loc[(sym, t), "label"])  # type: ignore
        if got_label != exp_label:
            mismatches += 1

    return mismatches / len(picks)


@app.command("run")
def run(
    labels: str = typer.Option(..., "--labels", help="Labels parquet glob"),
    features: str = typer.Option(..., "--features", help="Features parquet glob"),
    k: float = typer.Option(1.2, "--k"),
    H: int = typer.Option(36, "--H"),
    atr_window: int = typer.Option(14, "--atr-window"),
    days: int = typer.Option(80, "--days"),
    tz: str = typer.Option("UTC", "--tz"),
    out_json: str = typer.Option("reports/p3_validate_5m_80d.json", "--out-json"),
    strict_wait: bool = typer.Option(False, "--strict-wait", help="Fail if WAIT share > 0.8"),
    sample_size: int = typer.Option(500, "--sample"),
    raw: Optional[str] = typer.Option(None, "--raw", help="Optional P1 raw lake glob to supply OHLC if features lack it"),
) -> None:
    # Load data
    lab = _read_parquet(labels)
    feat = _read_parquet(features)
    # Early diagnostics always write a report for operator visibility
    if lab.empty:
        Path(Path(out_json).parent).mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump({
                "error": "no_labels",
                "labels_glob": labels,
                "features_glob": features,
                "pass": False,
                "violations": ["no_labels_found"],
            }, f, indent=2)
        typer.echo("FAIL: no label rows found (wrote report)")
        raise typer.Exit(code=1)
    if feat.empty:
        Path(Path(out_json).parent).mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump({
                "error": "no_features",
                "labels_glob": labels,
                "features_glob": features,
                "pass": False,
                "violations": ["no_features_found"],
            }, f, indent=2)
        typer.echo("FAIL: no features rows found (wrote report)")
        raise typer.Exit(code=1)
    typer.echo(f"Rows: labels={len(lab):,}, features={len(feat):,}")
    # Ensure OHLC present for ATR/barrier checks
    feat = _ensure_ohlc(feat, raw)

    # 1) Schema & keys
    req_cols = {"ts", "symbol", "label", "tp_px", "sl_px", "timeout_ts"}
    missing = sorted(list(req_cols - set(lab.columns)))
    violations: List[str] = []
    if missing:
        violations.append(f"missing_columns:{','.join(missing)}")
    # dtypes/NaN
    if lab["ts"].isna().any():
        violations.append("nan_ts")
    if lab["label"].isna().any():
        violations.append("nan_label")
    if lab["tp_px"].isna().any() or lab["sl_px"].isna().any() or lab["timeout_ts"].isna().any():
        violations.append("nan_tp_sl_timeout")
    invalid_labels = set(map(str, lab["label"].unique())) - ALLOWED_LABELS
    if invalid_labels:
        violations.append(f"invalid_labels:{','.join(sorted(invalid_labels))}")
    dup_key_count = int(lab.duplicated(subset=["symbol", "ts"]).sum())
    if dup_key_count > 0:
        violations.append(f"dup_key_count:{dup_key_count}")

    # 2) Coverage & gaps on usable 50d window
    feat_min = pd.to_datetime(feat["ts"].min(), utc=True)
    full_grid = _grid_5m(feat_min, days)
    start_usable = feat_min + pd.Timedelta(days=max(0, days - 50))  # last 50d of the 80d window
    usable_grid = full_grid[full_grid >= start_usable]
    # Restrict labels to usable window
    lab_u = lab[(lab["ts"] >= usable_grid.min()) & (lab["ts"] <= usable_grid.max())]
    present = int(lab_u["ts"].nunique())
    expected = USABLE_50D_BARS if days == 80 else len(usable_grid)
    gap_ratio = float(1 - (present / expected if expected else 1.0))
    if gap_ratio > 0.005:
        violations.append(f"gap_ratio_after_warmup:{gap_ratio:.6f}>")

    # 3) Join integrity
    con = duckdb.connect()
    try:
        con.execute(f"CREATE OR REPLACE VIEW _lab AS SELECT * FROM read_parquet('{labels}')")
        con.execute(f"CREATE OR REPLACE VIEW _feat AS SELECT * FROM read_parquet('{features}')")
        join_cnt = con.execute("SELECT COUNT(*) FROM _lab l INNER JOIN _feat f USING(symbol, ts)").fetchone()[0]
    finally:
        con.close()
    if int(join_cnt) != int(len(lab)):
        violations.append(f"join_count_mismatch:{join_cnt}!={len(lab)}")

    # 4) Class histogram sanity
    hist = lab_u["label"].value_counts().to_dict()
    wait_share = float(hist.get("WAIT", 0) / max(1, len(lab_u)))
    wait_warn = wait_share > 0.80
    if strict_wait and wait_warn:
        violations.append(f"wait_share>{wait_share:.3f}")

    # 5) Sample-based rule checks
    try:
        mismatch_ratio = _sample_rulecheck(lab_u[["symbol", "ts", "label"]], feat, H=H, k=k, atr_window=atr_window, sample_size=sample_size)
    except Exception:
        mismatch_ratio = 1.0
    if mismatch_ratio > 0.02:
        violations.append(f"rulecheck_mismatch>{mismatch_ratio:.3f}")

    # 6) Unit tests hook for P3
    try:
        import pytest  # type: ignore

        rc = pytest.main(["-q", "tests/test_label_p3.py"])  # run only P3 tests
        if rc != 0:
            violations.append("unit_tests_failed")
    except Exception:
        # If pytest not available in runtime image, mark as violation instead of crashing
        violations.append("pytest_unavailable")

    passed = len(violations) == 0

    # Report JSON
    report = {
        "expected_usable_bars_50d": USABLE_50D_BARS,
        "present_labels": int(len(lab_u)),
        "gap_ratio_after_warmup": float(gap_ratio),
        "dup_key_count": dup_key_count,
        "class_histogram": {str(k): int(v) for k, v in hist.items()},
        "sample_rulecheck_mismatch_ratio": float(mismatch_ratio),
        "join_count": int(join_cnt),
        "wait_share": float(wait_share),
        "wait_warn": bool(wait_warn),
        "pass": bool(passed),
        "violations": violations,
    }
    Path(Path(out_json).parent).mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    if passed:
        typer.echo("PASS")
        raise typer.Exit(code=0)
    else:
        typer.echo("FAIL: " + "; ".join(violations))
        raise typer.Exit(code=1)


@app.callback()
def _main_callback(
    ctx: typer.Context,
    labels: Optional[str] = typer.Option(None, "--labels"),
    features: Optional[str] = typer.Option(None, "--features"),
) -> None:
    # Allow calling without subcommand: python label_validate.py --labels ... --features ...
    if ctx.invoked_subcommand is None:
        if not labels or not features:
            typer.echo("Missing required options. Use --help for usage.")
            raise typer.Exit(code=2)
