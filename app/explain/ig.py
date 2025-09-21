from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn


BaselineType = Literal["zeros", "feature_means"]


@dataclass
class IGConfig:
    steps: int = 32
    baseline: BaselineType = "zeros"
    target: int = 1  # 0=WAIT,1=LONG,2=SHORT
    topk: int = 10


def _make_baseline(x: np.ndarray, baseline: BaselineType, feature_means: np.ndarray | None) -> np.ndarray:
    if baseline == "feature_means" and feature_means is not None:
        if feature_means.shape[0] != x.shape[1]:
            raise ValueError("feature_means length must equal F")
        return np.tile(feature_means[None, :], (x.shape[0], 1)).astype(np.float32)
    return np.zeros_like(x, dtype=np.float32)


def integrated_gradients(model: nn.Module, window: np.ndarray, cfg: IGConfig, feature_means: np.ndarray | None = None) -> np.ndarray:
    """Compute IG attributions for a single window [144,F] → [144,F]."""
    if window.ndim != 2:
        raise ValueError("window must be 2D [W,F]")
    W, F = window.shape
    if W != 144:
        raise ValueError("W must be 144 for Phase 10")
    x_np = window.astype(np.float32)
    base_np = _make_baseline(x_np, cfg.baseline, feature_means)
    model.eval()
    x = torch.tensor(x_np[None, :, :], dtype=torch.float32, requires_grad=True)
    baseline = torch.tensor(base_np[None, :, :], dtype=torch.float32)
    total_grad = torch.zeros_like(x)
    alphas = np.linspace(0.0, 1.0, max(1, int(cfg.steps)), dtype=np.float32)
    for a in alphas:
        xi = baseline + float(a) * (x - baseline)
        xi.requires_grad_(True)
        logits = model(xi)
        loss = logits[0, int(cfg.target)]
        model.zero_grad(set_to_none=True)
        if xi.grad is not None:
            xi.grad.zero_()
        loss.backward(retain_graph=True)
        grad = xi.grad.detach()
        total_grad += grad
    avg_grad = total_grad / len(alphas)
    attr = (x - baseline) * avg_grad
    return attr.detach().cpu().numpy()[0]


def topk_sparsify(attr: np.ndarray, k: int, feature_names: List[str] | None = None) -> List[Dict[str, object]]:
    W, F = attr.shape
    flat = attr.reshape(-1)
    idx = np.argsort(np.abs(flat))[::-1][: max(0, int(k))]
    items: List[Dict[str, object]] = []
    for i in idx:
        t = int(i // F)
        f_idx = int(i % F)
        f_name = feature_names[f_idx] if feature_names and f_idx < len(feature_names) else f"f{f_idx}"
        items.append({"t": t, "f": f_name, "attr": float(attr[t, f_idx])})
    return items


def summarize(attr: np.ndarray) -> Dict[str, float]:
    return {
        "sum": float(attr.sum()),
        "l1": float(np.abs(attr).sum()),
        "l2": float(np.sqrt((attr ** 2).sum())),
        "max_abs": float(np.abs(attr).max() if attr.size else 0.0),
    }

