from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from cli_p5 import GRUClassifier


@dataclass
class ExportSpec:
    window: int = 144
    opset: int = 17
    input_name: str = "inputs"
    output_name: str = "probs"


def _infer_dims_from_state(sd: dict) -> Tuple[int, int]:
    w_ih = sd.get("gru.weight_ih_l0")
    w_hh = sd.get("gru.weight_hh_l0")
    if w_ih is None or w_hh is None:
        raise ValueError("State dict missing GRU weights to infer dimensions")
    hidden = w_hh.shape[1]
    input_dim = w_ih.shape[1]
    return input_dim, hidden


class _ExportWrapper(nn.Module):
    """Wrap classifier and output calibrated probabilities (softmax(logits/T))."""

    def __init__(self, net: nn.Module, temperature: float = 1.0) -> None:
        super().__init__()
        self.net = net
        self.T = torch.tensor(float(temperature))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, W, F] -> [B, 3]
        logits = self.net(x)
        z = logits / self.T
        return torch.softmax(z, dim=1)


def export_fp16_probs(ckpt_path: str, out_path: str, temperature: float, spec: ExportSpec = ExportSpec()) -> Tuple[int, int]:
    """Export GRUClassifier checkpoint to ONNX FP16 graph that outputs probabilities.

    Returns: (input_dim, hidden)
    """
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    sd = torch.load(p, map_location="cpu")
    state = sd.get("state_dict", sd)
    input_dim, hidden = _infer_dims_from_state(state)

    net = GRUClassifier(input_dim=input_dim, hidden=hidden)
    net.load_state_dict(state)
    net.eval().half()
    wrapper = _ExportWrapper(net, temperature=temperature).half()

    dummy = torch.zeros(1, spec.window, input_dim, dtype=torch.float16)
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(outp),
        input_names=[spec.input_name],
        output_names=[spec.output_name],
        dynamic_axes={spec.input_name: {0: "batch"}, spec.output_name: {0: "batch"}},
        opset_version=spec.opset,
    )
    return input_dim, hidden

