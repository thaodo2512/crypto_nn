from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
import torch

from cli_p5 import GRUClassifier
from .onnx_export import _infer_dims_from_state


def sha256_path(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ParityReport:
    mse_probs: float
    n_samples: int
    onnx_sha256: str
    preproc_sha256: str
    ensemble_sha256: str
    opset: int
    window: int
    dynamic_axes: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "mse_probs": self.mse_probs,
                "n_samples": self.n_samples,
                "sha256": {
                    "onnx": self.onnx_sha256,
                    "preproc": self.preproc_sha256,
                    "ensemble": self.ensemble_sha256,
                },
                "opset": self.opset,
                "window": self.window,
                "dynamic_axes": list(self.dynamic_axes),
            },
            indent=2,
        )


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def parity_check(
    ckpt_path: str,
    onnx_path: str,
    temperature: float,
    samples: int,
    window: int,
    seed: int = 42,
) -> float:
    """Compute MSE between PyTorch calibrated probs and ONNX probs on synthetic deterministic windows."""
    # Load model dims
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("state_dict", sd)
    input_dim, hidden = _infer_dims_from_state(state)
    # PyTorch
    torch.manual_seed(seed)
    x = torch.randn(samples, window, input_dim, dtype=torch.float32)
    net = GRUClassifier(input_dim=input_dim, hidden=hidden)
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        logits = net(x) / float(temperature)
        p_t = torch.softmax(logits, dim=1).cpu().numpy()
    # ONNX
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_rt = x.numpy().astype(np.float16)
    p_o = sess.run(["probs"], {"inputs": x_rt})[0].astype(np.float32)
    # MSE
    return float(np.mean((p_t - p_o) ** 2))

