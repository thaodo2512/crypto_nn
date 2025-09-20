import numpy as np
from pathlib import Path

from app.export.onnx_export import export_fp16_probs
from app.runtime.model_session import ModelSession
from app.runtime.onnx_introspect import infer_io


def test_real_onnx_latency(tmp_path):
    # Export tiny GRU ONNX FP16
    import torch
    from cli_p5 import GRUClassifier
    input_dim, hidden, W = 6, 8, 144
    net = GRUClassifier(input_dim=input_dim, hidden=hidden)
    ckpt = tmp_path / "best.pt"
    torch.save({"state_dict": net.state_dict()}, ckpt)
    out = tmp_path / "m.onnx"
    export_fp16_probs(str(ckpt), str(out), temperature=1.0)
    F = infer_io(str(out), expect_window=W)
    assert F == input_dim
    ms = ModelSession(str(out), window=W, input_dim=F, apply_temp=False)
    win = np.random.randn(W, F).astype(np.float32)
    # Measure
    import time
    t0 = time.perf_counter(); _ = ms.predict_proba(win); dt = (time.perf_counter()-t0)*1000
    assert dt < 2000.0

