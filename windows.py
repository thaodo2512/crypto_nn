from __future__ import annotations

from typing import Tuple, List

import numpy as np
import pandas as pd


def build_sequence_windows(
    df_features: pd.DataFrame,
    df_labels: pd.DataFrame,
    W: int = 144,
    feature_cols: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build sliding windows of length W ending at label time t.

    Returns (X, y, meta) where:
      - X: shape [N, W*F] flattened
      - y: array of labels (strings)
      - meta: DataFrame with ['ts', 'symbol'] for each window
    """
    df = df_features.sort_values(["symbol", "ts"]).reset_index(drop=True)
    labels = df_labels.sort_values(["symbol", "ts"]).reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    labels["ts"] = pd.to_datetime(labels["ts"], utc=True)
    use_cols = feature_cols or [
        c
        for c in df.columns
        if c not in {"ts", "symbol"} and df[c].dtype.kind in {"f", "i"}
    ]
    # Build per-symbol windows
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    meta_rows: List[dict] = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.set_index("ts").sort_index()
        lab_sym = labels[labels["symbol"] == sym]
        for _, row in lab_sym.iterrows():
            t = row["ts"]
            win = g.loc[(g.index > t - pd.Timedelta(minutes=5 * W)) & (g.index <= t), use_cols]
            if len(win) != W:
                continue
            X_list.append(win.values.reshape(1, -1))
            y_list.append(str(row["label"]))
            meta_rows.append({"ts": t, "symbol": sym})
    if not X_list:
        return np.empty((0, 0)), np.array([]), pd.DataFrame(columns=["ts", "symbol"])  # type: ignore
    X = np.vstack(X_list)
    y = np.array(y_list)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta

