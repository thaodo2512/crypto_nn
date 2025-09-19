from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    oos_idx: np.ndarray


def make_purged_folds(ts_index: pd.DatetimeIndex, n_folds: int = 5, embargo: str = "1D") -> List[Dict]:
    """Create time-ordered purged walk-forward folds with an embargo gap.

    - Splits the time range into n_folds contiguous validation blocks.
    - Training uses all data strictly before the validation block, excluding an
      embargo window immediately before and after the validation block.
    - OOS uses data strictly after the validation block, excluding the embargo after.
    Returns a list of dicts with numpy index arrays.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(ts_index, utc=True)).sort_values()
    n = len(ts)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    # Determine fold boundaries by equally splitting unique timestamps
    fold_edges = np.linspace(0, n, n_folds + 1, dtype=int)
    embargo_td = pd.to_timedelta(embargo)
    folds: List[Dict] = []
    for k in range(n_folds):
        val_start = ts[fold_edges[k]]
        val_end = ts[fold_edges[k + 1] - 1]
        # Embargo windows
        emb_before_start = val_start - embargo_td
        emb_after_end = val_end + embargo_td
        is_val = (ts >= val_start) & (ts <= val_end)
        is_train = ts < emb_before_start
        is_oos = ts > emb_after_end
        folds.append(
            {
                "fold_id": k,
                "train_idx": np.where(is_train)[0],
                "val_idx": np.where(is_val)[0],
                "oos_idx": np.where(is_oos)[0],
            }
        )
    return folds

