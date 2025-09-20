from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import onnxruntime as ort


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class ModelSession:
    onnx_path: Optional[str]
    window: int
    input_dim: int
    apply_temp: bool = False
    temperature: float = 1.0

    def __post_init__(self) -> None:
        self.dummy = self.onnx_path is None
        self.input_name = "inputs"
        self.output_name = "probs"
        self.sess = None
        if not self.dummy and self.onnx_path:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                self.sess = ort.InferenceSession(self.onnx_path, providers=providers)
            except Exception:
                self.sess = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])

    def predict_proba(self, win: np.ndarray) -> np.ndarray:
        # win: [W,F]
        x = win[np.newaxis, :, :].astype(np.float32)
        if self.dummy or self.sess is None:
            logits = x.mean(axis=(1, 2), keepdims=True) * np.array([[[-0.1, 0.2, -0.1]]])
            p = _softmax(logits / max(self.temperature, 1e-6)).squeeze(0)
            return p.astype(np.float32)
        # Try fetch probs; fall back to logits
        names = [o.name for o in self.sess.get_outputs()]
        if "probs" in names and not self.apply_temp:
            probs = self.sess.run(["probs"], {self.input_name: x.astype(np.float16)})[0]
            return probs.squeeze(0).astype(np.float32)
        # Otherwise, expect logits and apply temperature once
        logits = self.sess.run(["logits"], {self.input_name: x.astype(np.float16)})[0]
        p = _softmax(logits / max(self.temperature, 1e-6)).squeeze(0)
        return p.astype(np.float32)

