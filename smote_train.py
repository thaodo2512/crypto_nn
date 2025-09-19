from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def smote_long_short_only(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE on LONG/SHORT only; WAIT is excluded from resampling and appended back.
    Returns augmented (X_smote∪WAIT, y_smote∪WAIT).
    """
    mask_ls = (y == "LONG") | (y == "SHORT")
    X_ls, y_ls = X[mask_ls], y[mask_ls]
    X_wait, y_wait = X[~mask_ls], y[~mask_ls]
    if len(np.unique(y_ls)) < 2 or len(y_ls) < 2:
        # Not enough for SMOTE; return original
        return X, y
    sm = SMOTE(random_state=seed)
    X_res, y_res = sm.fit_resample(X_ls, y_ls)
    X_out = np.vstack([X_res, X_wait])
    y_out = np.concatenate([y_res, y_wait])
    return X_out, y_out


def apply_per_fold(
    X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, folds: List[Dict], out_root: str, seed: int = 42
) -> Dict[int, Dict[str, int]]:
    """Apply SMOTE per fold on TRAIN only. Persist to data/aug/train_smote/<fold>/train.parquet.

    Returns class counts per fold before/after.
    """
    out_counts: Dict[int, Dict[str, int]] = {}
    Path(out_root).mkdir(parents=True, exist_ok=True)
    # Build ts index alignment
    order = meta.sort_values("ts").reset_index(drop=True)
    ts_all = order["ts"].values
    for f in folds:
        fid = f["fold_id"]
        # Train meta rows by intersecting indices
        train_rows = order.iloc[f["train_idx"]]
        idx_mask = order.index.isin(train_rows.index)
        X_tr, y_tr = X[idx_mask], y[idx_mask]
        # Pre counts
        pre_counts = {c: int((y_tr == c).sum()) for c in ["LONG", "SHORT", "WAIT"]}
        # SMOTE only on train
        X_aug, y_aug = smote_long_short_only(X_tr, y_tr, seed=seed)
        post_counts = {c: int((y_aug == c).sum()) for c in ["LONG", "SHORT", "WAIT"]}
        out_counts[fid] = {f"pre_{k}": v for k, v in pre_counts.items()} | {f"post_{k}": v for k, v in post_counts.items()}
        # Persist per fold
        fold_dir = Path(out_root) / str(fid)
        fold_dir.mkdir(parents=True, exist_ok=True)
        # Save as Parquet with meta
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = pd.DataFrame({f"x{i}": X_aug[:, i] for i in range(X_aug.shape[1])})
        df["y"] = y_aug
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), fold_dir / "train.parquet")
    return out_counts

