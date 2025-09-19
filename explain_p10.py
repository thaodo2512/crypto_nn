from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import typer

from cli_p5 import GRUClassifier


app = typer.Typer(help="P10 – Integrated Gradients explain + retrieval API")


@dataclass
class IGConfig:
    steps: int = 32
    topk: int = 10
    target: int = 1  # class index: 1=LONG by default


def integrated_gradients(model: nn.Module, window: np.ndarray, target: int = 1, steps: int = 32) -> np.ndarray:
    """Integrated Gradients for sequence input [W,F] → attributions [W,F]."""
    model.eval()
    x = torch.tensor(window[None, :, :], dtype=torch.float32, requires_grad=True)
    baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)
    for a in np.linspace(0.0, 1.0, steps, dtype=np.float32):
        xi = baseline + a * (x - baseline)
        xi.requires_grad_(True)
        logits = model(xi)
        loss = logits[0, target]
        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward(retain_graph=True)
        grad = xi.grad.detach()
        total_grad += grad
    avg_grad = total_grad / steps
    attr = (x - baseline) * avg_grad
    return attr.detach().cpu().numpy()[0]


def topk_sparsify(attr: np.ndarray, k: int) -> List[Dict[str, Any]]:
    flat = attr.reshape(-1)
    idx = np.argsort(np.abs(flat))[::-1][:k]
    W, F = attr.shape
    items: List[Dict[str, Any]] = []
    for i in idx:
        t = int(i // F)
        f = int(i % F)
        items.append({"t": t, "f": f, "attr": float(attr[t, f])})
    return items


def build_summary(attr: np.ndarray) -> Dict[str, float]:
    return {
        "sum": float(attr.sum()),
        "l1": float(np.abs(attr).sum()),
        "l2": float(np.sqrt((attr ** 2).sum())),
        "max_abs": float(np.abs(attr).max()),
    }


@app.command("run")
def run(
    decision_id: str = typer.Option(..., "--decision-id"),
    out: str = typer.Option(..., "--out"),
    window_npy: Optional[str] = typer.Option(None, "--window-npy", help="Path to numpy .npy window [W,F]"),
    ckpt: Optional[str] = typer.Option(None, "--ckpt", help="PyTorch checkpoint to load GRU for IG"),
    steps: int = typer.Option(32, "--steps"),
    topk: int = typer.Option(10, "--topk"),
    target: int = typer.Option(1, "--target", help="Class index to attribute (1=LONG,2=SHORT)"),
) -> None:
    if window_npy is None:
        raise typer.BadParameter("--window-npy is required to compute IG")
    win = np.load(window_npy)
    if win.ndim != 2:
        raise typer.BadParameter("window must be [W,F]")
    if ckpt is None:
        # use a small GRU with input_dim inferred
        model = GRUClassifier(input_dim=win.shape[1], hidden=32)
    else:
        sd = torch.load(ckpt, map_location="cpu")
        state = sd.get("state_dict", sd)
        # Infer input_dim from state if possible
        w_ih = state.get("gru.weight_ih_l0")
        hidden = w_ih.shape[0] // 3 if w_ih is not None else 64
        in_dim = w_ih.shape[1] if w_ih is not None else win.shape[1]
        model = GRUClassifier(input_dim=in_dim, hidden=hidden)
        model.load_state_dict(state, strict=False)

    attr = integrated_gradients(model, win, target=target, steps=steps)
    W, F = attr.shape
    items = topk_sparsify(attr, k=topk)
    features = [f"f{i}" for i in range(F)]
    doc = {
        "id": decision_id,
        "ts_unix": int(time.time()),
        "window": win.tolist(),
        "features": features,
        "topk": items,
        "summary": build_summary(attr),
    }
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(doc, indent=2))
    # Also write a sample
    Path("reports").mkdir(parents=True, exist_ok=True)
    (Path("reports") / "p10_sample.json").write_text(json.dumps(doc, indent=2))
    typer.echo(f"Explain saved to {out}")


def make_api(explain_dir: Path, ttl_days: int = 30) -> FastAPI:
    api = FastAPI()
    ttl_s = ttl_days * 24 * 3600

    @api.get("/explain")
    def get_explain(id: str) -> JSONResponse:
        path = explain_dir / f"{id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="not found")
        st = path.stat()
        if (time.time() - st.st_mtime) > ttl_s:
            raise HTTPException(status_code=404, detail="expired")
        return JSONResponse(json.loads(path.read_text()))

    return api


@app.command("api")
def api(port: int = typer.Option(8081, "--port"), explain_dir: str = typer.Option("explain", "--dir")) -> None:
    app = make_api(Path(explain_dir))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    app()

