from __future__ import annotations

from typing import List, Dict, Optional
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _select_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"ts", "symbol", "label"}
    cols = [c for c in df.columns if c not in exclude and df[c].dtype.kind in {"f", "i"}]
    return cols


def _robust_scale_past(X: pd.DataFrame) -> pd.DataFrame:
    med = X.median()
    mad = (X - med).abs().median()
    mad = mad.replace(0, 1.0)
    Z = (X - med) / mad
    return Z.fillna(0.0)


def fit_if_rolling(
    df_features: pd.DataFrame,
    q: float = 0.995,
    rolling_days: int = 30,
    seed: int = 42,
    batch: int = 512,
) -> pd.DataFrame:
    """Compute IsolationForest keep-mask per timestamp using only past data.

    - Trains IF on a rolling past window ending at t-1 (length=rolling_days in time).
    - Scores x_t and keeps it if score ≤ q-quantile of past scores within the window.
    Returns a DataFrame [ts, symbol, score, keep].
    """
    df = df_features.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)
    feats = _select_feature_cols(df)
    if not feats:
        raise ValueError("No numeric feature columns found for IsolationForest gate")
    out_rows = []
    win = pd.Timedelta(days=rolling_days)
    logger = logging.getLogger("p4")
    for sym, grp in df.groupby("symbol", sort=False):
        g = grp.copy()
        ts = g["ts"]
        scores = np.full(len(g), np.nan, dtype=float)
        logger.info(f"[IF] symbol={sym} rows={len(g)} window_days={rolling_days}")
        step = max(1, len(g) // 20)
        for i in range(len(g)):
            t = ts.iat[i]
            mask = (ts < t) & (ts >= t - win)
            if mask.sum() < 32:  # need minimal window to fit IF stably
                continue
            past = _robust_scale_past(g.loc[mask, feats])
            model = IsolationForest(random_state=seed, n_estimators=100, contamination="auto")
            model.fit(past.values)
            x_t = _robust_scale_past(g.loc[[i], feats])
            s = -model.score_samples(x_t.values)[0]  # higher s => more anomalous
            # Compare to past scores distribution
            past_scores = -model.score_samples(past.values)
            thr = np.quantile(past_scores, q)
            keep = 1 if s <= thr else 0
            out_rows.append({"ts": t, "symbol": sym, "score": float(s), "keep": int(keep)})
            if (i % step) == 0:
                pct = (i + 1) / len(g)
                logger.info(f"[IF] {sym} progress: {i+1}/{len(g)} ({pct:.0%})")
    return pd.DataFrame(out_rows)


def export_mask_per_fold(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    folds: List[Dict],
    q: float,
    rolling_days: int,
    seed: int,
) -> pd.DataFrame:
    """Run IF gate and merge with folds to form a mask per (ts, symbol, fold_id)."""
    df = features.merge(labels[["ts", "symbol", "label"]], on=["ts", "symbol"], how="inner")
    logger = logging.getLogger("p4")
    logger.info("[IF] Fitting rolling isolation forest and building mask per fold...")
    gate = fit_if_rolling(df, q=q, rolling_days=rolling_days, seed=seed)
    # Assign fold_id by ts position
    all_ts = df.sort_values(["symbol", "ts"]).reset_index(drop=True)["ts"]
    masks: List[pd.DataFrame] = []
    for f in folds:
        idx = f["train_idx"]
        ts_train = all_ts.iloc[idx]
        m = gate.merge(ts_train.to_frame("ts").assign(_in_train=1), on="ts", how="inner")
        m = m.assign(fold_id=f["fold_id"]).loc[:, ["ts", "symbol", "keep", "fold_id"]]
        masks.append(m)
    mask_df = pd.concat(masks, ignore_index=True)
    logger.info(f"[IF] Mask rows total={len(mask_df):,}")
    return mask_df
