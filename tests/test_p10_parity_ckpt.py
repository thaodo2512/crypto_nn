import numpy as np
import torch
from typer.testing import CliRunner

from cli_p5 import GRUClassifier
from app.export.onnx_export import export_fp16_probs
from app.explain.cli import app as explain_app


def test_parity_guard_before_ig(tmp_path):
    # Build tiny GRU ckpt and export ONNX
    W, F = 144, 4
    net = GRUClassifier(input_dim=F, hidden=8)
    ckpt = tmp_path / "b.pt"
    torch.save({"state_dict": net.state_dict()}, ckpt)
    onnx = tmp_path / "m.onnx"
    export_fp16_probs(str(ckpt), str(onnx), temperature=1.0)
    # Window
    win = np.random.randn(W, F).astype(np.float32)
    np.save(tmp_path/"w.npy", win)
    # Run CLI; parity happens inside and should pass
    out_dir = tmp_path / "exp"
    r = CliRunner().invoke(explain_app, [
        "run", "--decision-id", "parity1",
        "--window-npy", str(tmp_path/"w.npy"),
        "--ckpt", str(ckpt), "--onnx", str(onnx),
        "--out-dir", str(out_dir)
    ])
    assert r.exit_code == 0, r.output

