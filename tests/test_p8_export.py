import json
import numpy as np
import torch

from cli_p5 import GRUClassifier
from export_p8 import export_onnx
from typer.testing import CliRunner


def test_parity_mse_lt_1e3(tmp_path):
    # Create a tiny GRU and save checkpoint
    input_dim, hidden, W = 8, 16, 12
    net = GRUClassifier(input_dim=input_dim, hidden=hidden)
    ckpt = tmp_path / "best.pt"
    torch.save({"state_dict": net.state_dict()}, ckpt)
    # Dummy config files
    pre = tmp_path / "pre.yaml"; pre.write_text("{}");
    calib = tmp_path / "calib.json"; calib.write_text("{}")
    out = tmp_path / "model.onnx"
    r = CliRunner().invoke(export_onnx, [
        "--ckpt", str(ckpt), "--fp16", "--out", str(out),
        "--preproc", str(pre), "--calib", str(calib), "--window", "12", "--sample", "8",
    ])
    assert r.exit_code == 0, r.output
    # Parity JSON
    d = json.loads((tmp_path.parent / "reports" / "p8_parity.json").read_text())
    assert d["mse"] < 1e-3


def test_checksum_logged(tmp_path):
    # Create model and export
    input_dim, hidden, W = 4, 8, 6
    net = GRUClassifier(input_dim=input_dim, hidden=hidden)
    ckpt = tmp_path / "best.pt"
    torch.save({"state_dict": net.state_dict()}, ckpt)
    pre = tmp_path / "pre.yaml"; pre.write_text("a: 1")
    calib = tmp_path / "calib.json"; calib.write_text("{}")
    out = tmp_path / "model.onnx"
    CliRunner().invoke(export_onnx, ["--ckpt", str(ckpt), "--out", str(out), "--preproc", str(pre), "--calib", str(calib), "--window", "6", "--sample", "4"])  # fp32
    d = json.loads((tmp_path.parent / "reports" / "p8_parity.json").read_text())
    assert d.get("onnx_sha256") and isinstance(d["onnx_sha256"], str)

