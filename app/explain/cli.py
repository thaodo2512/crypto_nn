from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import onnxruntime as ort
import torch
import typer
import uvicorn

from cli_p5 import GRUClassifier
from app.runtime.onnx_introspect import infer_io
from .ig import IGConfig, integrated_gradients, topk_sparsify, summarize
from .store import ExplainStore
from .api import make_router


app = typer.Typer(help="P10 – Explainability CLI (IG + optional IF-SHAP), API, and GC")


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def parity_guard(ckpt: str, onnx_path: str, window: int = 144, samples: int = 256, seed: int = 42) -> float:
    sd = torch.load(ckpt, map_location="cpu")
    state = sd.get("state_dict", sd)
    input_dim = state.get("gru.weight_ih_l0").shape[1]
    F = infer_io(onnx_path, expect_window=window)
    if F != input_dim:
        raise typer.BadParameter(f"ONNX feature dim {F} != ckpt input_dim {input_dim}")
    net = GRUClassifier(input_dim=input_dim)
    net.load_state_dict(state)
    net.eval()
    rng = np.random.RandomState(seed)
    X = rng.randn(samples, window, input_dim).astype(np.float32)
    with torch.no_grad():
        logits = net(torch.tensor(X)).cpu().numpy()
    p_t = _softmax_np(logits)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    p_o = sess.run(["probs"], {"inputs": X.astype(np.float16)})[0].astype(np.float32)
    mse = float(np.mean((p_t - p_o) ** 2))
    return mse


@app.command("run")
def run(
    decision_id: str = typer.Option(..., "--decision-id"),
    window_npy: str = typer.Option(..., "--window-npy"),
    ckpt: str = typer.Option(..., "--ckpt"),
    onnx: str = typer.Option("export/model_5m_fp16.onnx", "--onnx"),
    steps: int = typer.Option(32, "--steps"),
    topk: int = typer.Option(10, "--topk"),
    target: int = typer.Option(1, "--target"),
    baseline: str = typer.Option("zeros", "--baseline"),
    feature_means_npy: Optional[str] = typer.Option(None, "--feature-means-npy"),
    out_dir: str = typer.Option("explain", "--out-dir"),
    if_csv: Optional[str] = typer.Option(None, "--if-csv", help="CSV with alerts (id,alert=1) for IF-SHAP"),
) -> None:
    win = np.load(window_npy)
    if win.ndim != 2 or win.shape[0] != 144:
        raise typer.BadParameter("window must be [144,F]")
    # Parity guard
    mse = parity_guard(ckpt, onnx_path=onnx, window=144, samples=256)
    if mse >= 1e-3:
        raise SystemExit(f"Parity MSE too high: {mse}")
    # Load model for IG
    sd = torch.load(ckpt, map_location="cpu")
    state = sd.get("state_dict", sd)
    in_dim = state.get("gru.weight_ih_l0").shape[1]
    if win.shape[1] != in_dim:
        raise typer.BadParameter(f"window F={win.shape[1]} != model input_dim={in_dim}")
    model = GRUClassifier(input_dim=in_dim)
    model.load_state_dict(state)
    model.eval()
    # Feature means baseline (optional)
    fmeans = np.load(feature_means_npy) if (baseline == "feature_means" and feature_means_npy) else None
    cfg = IGConfig(steps=steps, baseline=baseline, target=target, topk=topk)
    attr = integrated_gradients(model, win, cfg, feature_means=fmeans)
    features = [f"f{i}" for i in range(win.shape[1])]
    top = topk_sparsify(attr, k=topk, feature_names=features)
    doc = {
        "id": decision_id,
        "ts_unix": int(time.time()),
        "window_shape": [144, int(win.shape[1])],
        "features": features,
        "topk": top,
        "summary": summarize(attr),
    }
    # Optional IF-SHAP (only when alerts.csv marks alert=1 for this id)
    if if_csv:
        try:
            import pandas as pd
            from sklearn.ensemble import IsolationForest
            from .shap_if import shap_like_for_if

            alerts = pd.read_csv(if_csv)
            row = alerts.loc[alerts["id"] == decision_id]
            if not row.empty and int(row.iloc[0].get("alert", 0)) == 1:
                # Fit a tiny IF on the last 100 rows from the same CSV if present, else on random noise
                bg = alerts.filter(like="f").to_numpy(dtype=np.float32, copy=True)
                if bg.size == 0:
                    bg = np.random.randn(256, win.shape[1]).astype(np.float32)
                if bg.shape[1] != win.shape[1]:
                    bg = np.random.randn(256, win.shape[1]).astype(np.float32)
                if_model = IsolationForest(random_state=42, n_estimators=100, contamination="auto")
                if_model.fit(bg)
                shap_top = shap_like_for_if(if_model, x=win[-1], background=bg, topk=topk, feature_names=features)
                doc["if_shap"] = shap_top
        except Exception:
            pass
    # Store
    store = ExplainStore(out_dir, ttl_days=30)
    store.write_atomic(decision_id, doc)
    typer.echo(f"Explain saved to {store._path(decision_id)} (MSE={mse:.2e})")


@app.command("api")
def api(port: int = typer.Option(8081, "--port"), dir: str = typer.Option("explain", "--dir")) -> None:
    store = ExplainStore(dir, ttl_days=30)
    router = make_router(store)
    from fastapi import FastAPI

    srv = FastAPI()
    srv.include_router(router)
    uvicorn.run(srv, host="0.0.0.0", port=port)


@app.command("gc")
def gc(dir: str = typer.Option("explain", "--dir"), ttl_days: int = typer.Option(30, "--ttl-days")) -> None:
    store = ExplainStore(dir, ttl_days=ttl_days)
    n = store.gc()
    typer.echo(json.dumps({"removed": n}))


if __name__ == "__main__":
    app()

