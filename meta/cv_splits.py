from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

BARS_PER_DAY = 288  # 5m bars per day


@dataclass
class Fold:
    fold_id: int
    train: List[List[str]]
    val: List[List[str]]
    oos: List[List[str]]
    counts: dict


def _eligible_idx(n: int, W: int, H: int) -> tuple[int, int]:
    """Return inclusive [i0, i1] index bounds eligible for decision timestamps.

    i0 ensures we have W-1 bars of lookback; i1 ensures we have H bars of lookahead.
    """
    start = max(W - 1, 0)
    end = max(n - H - 1, start)
    return start, end


def build_walkforward(
    ts: pd.Series,
    n_folds: int = 5,
    W: int = 144,
    H: int = 36,
    embargo: int = BARS_PER_DAY,
    min_oos: int = BARS_PER_DAY,
    val_bars: int = 144,
) -> List[Fold]:
    """Construct purged walk-forward folds with embargo and non-empty OOS per fold.

    - ts must be a sorted Datetime Series of decision timestamps (right-closed labels).
    - Returns contiguous spans for train/val/oos with counts.
    """
    ts = pd.to_datetime(ts, utc=True).sort_values().reset_index(drop=True)
    n = len(ts)
    i0, i1 = _eligible_idx(n, W, H)
    # Sanity: ensure we can allocate n_folds OOS slices separated by nothing (contiguous ok) + embargo & val lengths
    min_required = n_folds * min_oos + embargo + val_bars
    if (i1 - i0 + 1) <= min_required:
        raise ValueError("Dataset too short for folds with embargo/min_oos")

    # Evenly place OOS windows within eligible band
    oos_len = int(min_oos)
    oos_starts = np.linspace(i0 + val_bars + embargo, i1 - oos_len + 1, n_folds).astype(int)
    folds: List[Fold] = []
    for k, s in enumerate(oos_starts):
        oos_start = int(s)
        oos_end = int(s + oos_len - 1)
        # Train ends embargo bars before OOS
        train_end = max(oos_start - embargo - 1, i0)
        # Val is the tail of train of fixed length
        val_end = train_end
        val_start = max(val_end - val_bars + 1, i0)
        # Build spans as ISO strings
        train_span = [[ts[i0].isoformat(), ts[train_end].isoformat()]] if train_end >= i0 else []
        val_span = [[ts[val_start].isoformat(), ts[val_end].isoformat()]] if val_end >= val_start else []
        oos_span = [[ts[oos_start].isoformat(), ts[oos_end].isoformat()]] if oos_end >= oos_start else []
        counts = {
            "train": max(train_end - i0 + 1, 0),
            "val": max(val_end - val_start + 1, 0),
            "oos": max(oos_end - oos_start + 1, 0),
        }
        if counts["oos"] <= 0:
            raise AssertionError(f"empty OOS in fold{k}")
        if counts["train"] <= 0 or counts["val"] <= 0:
            raise AssertionError(f"empty TRAIN/VAL in fold{k}")
        folds.append(Fold(k, train_span, val_span, oos_span, counts))
    return folds


def emit_folds_json(
    ts: pd.Series,
    out_path: str | Path,
    *,
    tf: str = "5m",
    symbol: str = "BTCUSDT",
    window: int = 144,
    horizon: int = 36,
    embargo_bars: int = BARS_PER_DAY,
    n_folds: int = 5,
    min_oos: int = BARS_PER_DAY,
    val_bars: int = 144,
) -> Path:
    """Emit folds.json with meta and folds, returns the path written."""
    folds = build_walkforward(ts, n_folds=n_folds, W=window, H=horizon, embargo=embargo_bars, min_oos=min_oos, val_bars=val_bars)
    payload = {
        "meta": {
            "tf": tf,
            "symbol": symbol,
            "window": window,
            "horizon": horizon,
            "embargo_bars": embargo_bars,
            "n_folds": n_folds,
        },
        "folds": [asdict(f) for f in folds],
    }
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(payload, f, indent=2)
    return outp

