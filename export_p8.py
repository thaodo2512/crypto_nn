from __future__ import annotations

import hashlib
import json
import logging
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import typer

from cli_p5 import GRUClassifier


app = typer.Typer(help="P8 – Export trained model to ONNX FP16 and validate parity")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_dims_from_state(sd: dict) -> Tuple[int, int]:
    w_ih = sd.get("gru.weight_ih_l0")
    w_hh = sd.get("gru.weight_hh_l0")
    if w_ih is None or w_hh is None:
        raise ValueError("State dict missing GRU weights to infer dimensions")
    hidden = w_hh.shape[1]
    input_dim = w_ih.shape[1]
    return input_dim, hidden


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@app.command("onnx")
def export_onnx(
    ckpt: str = typer.Option(..., "--ckpt", help="Checkpoint path or glob (first will be used)"),
    fp16: bool = typer.Option(True, "--fp16/--fp32", help="Export with FP16 weights"),
    out: str = typer.Option("export/model_5m_fp16.onnx", "--out"),
    preproc: str = typer.Option(..., "--preproc", help="Preprocessing YAML path"),
    calib: str = typer.Option(..., "--calib", help="Calibration JSON path (ensemble/temps)"),
    window: int = typer.Option(144, "--window", help="Window length W for dummy input"),
    sample: int = typer.Option(16, "--sample", help="Samples for parity check"),
) -> None:
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("p8")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/p8_export.log")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Resolve ckpt
    paths = sorted(glob(ckpt))
    if not paths:
        raise typer.BadParameter(f"No checkpoint files found for pattern: {ckpt}")
    ckpt_path = Path(paths[0])
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("state_dict", sd)
    input_dim, hidden = _infer_dims_from_state(state)
    net = GRUClassifier(input_dim=input_dim, hidden=hidden)
    net.load_state_dict(state)
    net.eval()
    if fp16:
        net = net.half()

    # Dummy input [B, W, F]
    dummy = torch.randn(1, window, input_dim, dtype=torch.float16 if fp16 else torch.float32)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export ONNX
    torch.onnx.export(
        net,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch", 1: "window"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # Parity check on random sample
    B = sample
    x = torch.randn(B, window, input_dim)
    net_float = GRUClassifier(input_dim=input_dim, hidden=hidden)
    net_float.load_state_dict(state)
    net_float.eval()
    with torch.no_grad():
        logits_t = net_float(x).cpu().numpy()
    # ONNX inference (match dtype)
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x_rt = x.numpy().astype(np.float16 if fp16 else np.float32)
    logits_o = sess.run(["logits"], {"input": x_rt})[0]
    p_t = _softmax_np(logits_t)
    p_o = _softmax_np(logits_o.astype(np.float32))
    mse = float(np.mean((p_t - p_o) ** 2))

    # Checksums
    onnx_sha = _sha256(out_path)
    calib_sha = _sha256(Path(calib)) if Path(calib).exists() else ""
    pre_sha = _sha256(Path(preproc)) if Path(preproc).exists() else ""

    # Report
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/p8_parity.json", "w") as f:
        json.dump({"mse": mse, "samples": B, "onnx_sha256": onnx_sha, "calib_sha256": calib_sha, "preproc_sha256": pre_sha}, f, indent=2)
    logger.info(f"Exported {out_path} sha256={onnx_sha} calib_sha256={calib_sha} preproc_sha256={pre_sha} mse={mse:.6g}")
    if mse >= 1e-3:
        raise SystemExit(1)
    typer.echo(f"ONNX exported. Parity MSE={mse:.6g}")


if __name__ == "__main__":
    app()

