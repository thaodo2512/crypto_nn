import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from typer.testing import CliRunner

from app.export.cli_p8_export import app as cli_app
from app.export.onnx_export import export_fp16_probs, ExportSpec
from cli_p5 import GRUClassifier


def _make_ckpt(tmp: Path, input_dim=8, hidden=16):
    net = GRUClassifier(input_dim=input_dim, hidden=hidden)
    ckpt = tmp / "best.pt"
    torch.save({"state_dict": net.state_dict()}, ckpt)
    return ckpt


def _write_ensemble(tmp: Path, temperature=1.0, weights=None):
    if weights is None:
        weights = {"gru": 1.0}
    path = tmp / "ensemble.json"
    path.write_text(json.dumps({"temperature": float(temperature), "weights": weights}))
    return path


def test_shapes_and_dynamic_axes(tmp_path):
    ckpt = _make_ckpt(tmp_path, input_dim=6, hidden=8)
    out = tmp_path / "m.onnx"
    export_fp16_probs(str(ckpt), str(out), temperature=1.0, spec=ExportSpec(window=144))
    m = onnx.load(str(out))
    # opset
    assert m.opset_import[0].version >= 17
    # input shape: (batch, 144, F)
    vi = m.graph.input[0]
    dims = [d.dim_param or (d.dim_value if d.dim_value != 0 else None) for d in vi.type.tensor_type.shape.dim]
    assert dims[0] is not None or dims[0] == "batch"
    assert dims[1] == 144
    # output name
    assert m.graph.output[0].name == "probs"


def test_fp16_graph_and_session(tmp_path):
    ckpt = _make_ckpt(tmp_path, input_dim=5, hidden=7)
    out = tmp_path / "m.onnx"
    export_fp16_probs(str(ckpt), str(out), temperature=1.0)
    m = onnx.load(str(out))
    # Check that there are FP16 initializers
    assert any(t.data_type == onnx.TensorProto.FLOAT16 for t in m.graph.initializer)
    # Session run
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.zeros((4, 144, 5), dtype=np.float16)
    y = sess.run(["probs"], {"inputs": x})[0]
    assert y.shape == (4, 3)
    assert np.allclose(y.sum(axis=1), 1.0, atol=1e-5)


def test_parity_pass_cli(tmp_path, monkeypatch):
    # Prepare ckpt and ensemble
    ckpt = _make_ckpt(tmp_path, input_dim=4, hidden=8)
    pre = tmp_path / "pre.yaml"; pre.write_text("window: 144\n")
    ens = _write_ensemble(tmp_path, temperature=1.0)
    out = tmp_path / "model.onnx"
    # Run CLI (export + parity)
    res = CliRunner().invoke(cli_app, [
        "onnx",
        "--ckpt", str(ckpt),
        "--out", str(out),
        "--window", "144",
        "--fp16",
        "--preproc", str(pre),
        "--ensemble", str(ens),
        "--sample", "256",
    ])
    assert res.exit_code == 0, res.output
    # Check report
    rpt = json.loads(Path("reports/p8_parity.json").read_text())
    assert rpt["mse_probs"] < 1e-3
    assert rpt["window"] == 144
    assert rpt["opset"] >= 17
    assert set(rpt["sha256"].keys()) == {"onnx", "preproc", "ensemble"}


def test_parity_fail_exitcode(tmp_path):
    # Export, then corrupt ONNX and run parity-only
    ckpt = _make_ckpt(tmp_path, input_dim=4, hidden=8)
    out = tmp_path / "model.onnx"
    pre = tmp_path / "pre.yaml"; pre.write_text("window: 144\n")
    ens = _write_ensemble(tmp_path, temperature=1.0)
    export_fp16_probs(str(ckpt), str(out), temperature=1.0)
    # Corrupt ONNX by flipping a byte
    data = bytearray(out.read_bytes())
    data[-10] = (data[-10] + 1) % 256
    out.write_bytes(bytes(data))
    res = CliRunner().invoke(cli_app, [
        "onnx",
        "--ckpt", str(ckpt),
        "--out", str(out),
        "--window", "144",
        "--fp16",
        "--preproc", str(pre),
        "--ensemble", str(ens),
        "--sample", "64",
        "--skip-export",
    ])
    assert res.exit_code != 0

