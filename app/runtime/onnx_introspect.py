from __future__ import annotations

from pathlib import Path
from typing import Optional

import onnx


def infer_io(onnx_path: str, expect_window: int = 144) -> int:
    """Infer input feature dimension from ONNX graph and enforce (batch, 144, F) input.

    Returns F. Raises RuntimeError on mismatch or unsupported shapes.
    """
    p = Path(onnx_path)
    if not p.exists():
        raise RuntimeError(f"ONNX not found: {onnx_path}")
    m = onnx.load(str(p))
    if not m.graph.input:
        raise RuntimeError("ONNX model has no inputs")
    inp = m.graph.input[0]
    shape = inp.type.tensor_type.shape.dim
    if len(shape) != 3:
        raise RuntimeError(f"Expected 3D input (batch, window, features), got {len(shape)} dims")
    # Extract dims
    b = shape[0]
    w = shape[1]
    f = shape[2]
    # Batch may be dynamic
    if w.dim_value == 0 and w.dim_param:
        # Window cannot be dynamic per spec
        raise RuntimeError("Window dimension must be static 144; got dynamic")
    win = w.dim_value or None
    if win != expect_window:
        raise RuntimeError(f"Window mismatch: expected {expect_window}, got {win}")
    feat = f.dim_value or None
    if feat is None or feat <= 0:
        raise RuntimeError("Feature dimension must be static and > 0")
    return int(feat)

