from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

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
    provider: str = "CPU"
    is_calibrated: bool = False
    _lgbm: Optional[object] = None

    def __post_init__(self) -> None:
        self.dummy = self.onnx_path is None
        self.input_name = "inputs"
        self.output_name = "probs"
        self.sess = None
        self._has_probs_output = False
        if not self.dummy and self.onnx_path:
            # Prefer TensorRT → CUDA → CPU
            providers: List[str] = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            try:
                self.sess = ort.InferenceSession(self.onnx_path, providers=providers)
                active = self.sess.get_providers()[0] if self.sess.get_providers() else "CPUExecutionProvider"
                self.provider = (
                    "TensorRT" if "Tensorrt" in active else ("CUDA" if "CUDA" in active else "CPU")
                )
            except Exception:
                self.sess = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
                self.provider = "CPU"
            # Detect if ONNX already outputs probabilities
            try:
                names = [o.name for o in self.sess.get_outputs()]
                self._has_probs_output = ("probs" in names)
                if self._has_probs_output:
                    self.is_calibrated = True
            except Exception:
                pass
        # Optional LightGBM fallback
        try:
            lgbm_path = Path("export/model_5m_lgbm.txt")
            if lgbm_path.exists():
                import lightgbm as lgb  # type: ignore

                self._lgbm = lgb.Booster(model_file=str(lgbm_path))
        except Exception:
            self._lgbm = None

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

    def file_checksum(self) -> Optional[str]:
        try:
            if not self.onnx_path:
                return None
            import hashlib

            h = hashlib.sha256()
            with open(self.onnx_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    def has_lgbm(self) -> bool:
        return self._lgbm is not None

    def predict_proba_lgbm(self, win: np.ndarray) -> np.ndarray:
        """Predict probabilities with LightGBM fallback.
        Flattens [W,F] → [W*F] and maps a 2-class output into (WAIT,LONG,SHORT) by putting leftover mass into WAIT.
        """
        if self._lgbm is None:
            raise RuntimeError("LGBM model not loaded")
        x = win.astype(np.float32).reshape(1, -1)
        try:
            # Expect predict_proba-like with shape [1,2] for long/short; map to 3 classes.
            probs = self._lgbm.predict(x)
            if isinstance(probs, list):
                probs = np.array(probs)
            probs = np.asarray(probs)
            if probs.ndim == 1:
                # If a single score, use symmetric mapping
                p_long = float(1 / (1 + np.exp(-probs[0])))
                p_short = 1.0 - p_long
            else:
                p_long = float(probs[0, 0])
                p_short = float(probs[0, 1]) if probs.shape[1] > 1 else 1.0 - p_long
            p_wait = max(0.0, 1.0 - p_long - p_short)
            out = np.array([p_wait, p_long, p_short], dtype=np.float32)
            out = out / out.sum()
            return out
        except Exception:
            # Fallback to ONNX path if anything goes wrong
            return self.predict_proba(win)
