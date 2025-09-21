from __future__ import annotations

from typing import List, Dict

import numpy as np
from sklearn.ensemble import IsolationForest


def shap_like_for_if(model: IsolationForest, x: np.ndarray, background: np.ndarray, topk: int = 10, feature_names: List[str] | None = None) -> List[Dict[str, object]]:
    """A lightweight SHAP-like attribution for IsolationForest on -score_samples.

    - model: fitted IsolationForest
    - x: single row [F]
    - background: background sample [N,F] used to re-fit a tiny surrogate if needed
    - Returns top-k features with largest |Δ score| when perturbing each feature toward background mean.
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D [F]")
    F = x.shape[0]
    # Baseline = mean of background
    base = background.mean(axis=0) if background.size else np.zeros_like(x)
    base = base.astype(np.float32)
    # Reference score
    s0 = -float(model.score_samples(x.reshape(1, -1))[0])
    shap_vals = np.zeros(F, dtype=np.float32)
    for j in range(F):
        xj = x.copy()
        xj[j] = base[j]
        sj = -float(model.score_samples(xj.reshape(1, -1))[0])
        shap_vals[j] = s0 - sj  # contribution of feature j
    idx = np.argsort(np.abs(shap_vals))[::-1][: max(0, int(topk))]
    out: List[Dict[str, object]] = []
    for j in idx:
        name = feature_names[j] if feature_names and j < len(feature_names) else f"f{j}"
        out.append({"f": name, "shap": float(shap_vals[j])})
    return out

