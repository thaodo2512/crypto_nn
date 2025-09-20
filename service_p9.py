from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import typer

from policy_p7 import _ev_decision_row, _dynamic_k


app_cli = typer.Typer(help="P9 – Real-time scoring & decision service (Jetson)")


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class HealthStatus:
    temp_c: float
    gpu_util: float


class HealthMonitor:
    def __init__(self) -> None:
        self.tegrastats = shutil.which("tegrastats") if (shutil := __import__("shutil")) else None

    def read(self) -> HealthStatus:
        # Try parse tegrastats once; fall back to nominal values in dev
        try:
            if self.tegrastats:
                out = subprocess.check_output([self.tegrastats, "--interval", "1000", "--count", "1"], timeout=2).decode()
                # crude parse: look for GPU and temp
                # Example contains: "GPU@xx% ... temp CPU@yyC GPU@zzC"
                gpu_util = 0.0
                temp = 50.0
                for tok in out.replace("%", "").replace("@", " ").split():
                    if tok.endswith("C") and tok[:-1].isdigit():
                        temp = float(tok[:-1])
                    if tok.isdigit():
                        v = float(tok)
                        if 0 <= v <= 100:
                            gpu_util = v
                            break
                return HealthStatus(temp_c=temp, gpu_util=gpu_util)
        except Exception:
            pass
        return HealthStatus(temp_c=50.0, gpu_util=30.0)


class ModelSession:
    def __init__(self, onnx_path: Optional[str], window: int, input_dim: int, temperature: float = 1.0) -> None:
        self.window = window
        self.input_dim = input_dim
        self.temperature = temperature
        self.dummy = onnx_path is None or os.getenv("P9_DUMMY", "0") == "1"
        if not self.dummy:
            providers = [
                ("CUDAExecutionProvider", {}),
                ("CPUExecutionProvider", {}),
            ]
            try:
                self.sess = ort.InferenceSession(onnx_path, providers=[p for p, _ in providers])
            except Exception:
                self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        else:
            self.sess = None
        # ONNX IO names (Phase 8 spec): inputs -> probs
        self.input_name = "inputs"
        self.output_name = "probs"

    def predict_proba(self, win: np.ndarray) -> np.ndarray:
        # win: [W,F]
        x = win[np.newaxis, :, :].astype(np.float32)
        if self.dummy:
            # simple linear logits for speed
            logits = x.mean(axis=(1, 2), keepdims=True) * np.array([[[-0.1, 0.2, -0.1]]])
            p = softmax_np(logits / max(self.temperature, 1e-6)).squeeze(0)
            return p
        else:
            # P8 ONNX already outputs probabilities
            probs = self.sess.run([self.output_name], {self.input_name: x.astype(np.float16)})[0]
            return probs.squeeze(0).astype(np.float32)


def create_app(model: ModelSession, k_min: float, k_max: float, H: int, monitor: Optional[HealthMonitor] = None) -> FastAPI:
    api = FastAPI()

    @api.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    @api.post("/score")
    def score(payload: Dict[str, Any]) -> JSONResponse:
        # Expect payload: {"window": [[...W x F...]], "meta": {...}}
        win = np.array(payload.get("window", []), dtype=np.float32)
        if win.ndim != 2:
            return JSONResponse({"error": "window must be 2D [W,F]"}, status_code=400)
        t0 = time.perf_counter()
        p = model.predict_proba(win)
        dt = (time.perf_counter() - t0) * 1000
        return JSONResponse({"p_wait": float(p[0]), "p_long": float(p[1]), "p_short": float(p[2]), "latency_ms": dt})

    @api.post("/decide")
    def decide(payload: Dict[str, Any]) -> JSONResponse:
        win = np.array(payload.get("window", []), dtype=np.float32)
        if win.ndim != 2 or win.shape[0] != model.window:
            return JSONResponse({"error": f"window must be [W,F] with W={model.window}"}, status_code=400)
        close = float(payload.get("close", 0.0))
        atr_pct = float(payload.get("atr_pct", 0.0))
        vol_pct = float(payload.get("vol_pctile", 0.5))
        t0 = time.perf_counter()
        # Monitor throttle
        throttled = False
        if monitor:
            hs = monitor.read()
            throttled = (hs.temp_c >= 70.0) or (hs.gpu_util >= 80.0)
        p = model.predict_proba(win)
        side, size, tp_px, sl_px, ev, reason = _ev_decision_row(
            float(p[1]), float(p[2]), close, atr_pct, k_min, k_max, vol_pct, type("C", (), {"cost_frac": lambda self: 0.0005})()
        )
        # Apply throttle: downgrade weak alerts
        if throttled and (max(p[1], p[2]) < 0.9):
            side, size, reason = "WAIT", 0.0, "throttle"
        dt = (time.perf_counter() - t0) * 1000
        return JSONResponse({
            "side": side,
            "size": size,
            "TP_px": tp_px,
            "SL_px": sl_px,
            "EV": ev,
            "reason": reason,
            "latency_ms": dt,
        })

    return api


@app_cli.command("api")
def api(
    onnx_path: Optional[str] = typer.Option(None, "--onnx", help="ONNX model path (optional; dummy if missing)"),
    window: int = typer.Option(144, "--window"),
    input_dim: int = typer.Option(16, "--input-dim"),
    temperature: float = typer.Option(1.0, "--temperature"),
    k_min: float = typer.Option(1.0, "--k-min"),
    k_max: float = typer.Option(1.5, "--k-max"),
    H: int = typer.Option(36, "--H"),
    port: int = typer.Option(8080, "--port"),
) -> None:
    monitor = HealthMonitor()
    model = ModelSession(onnx_path, window=window, input_dim=input_dim, temperature=temperature)
    app = create_app(model, k_min=k_min, k_max=k_max, H=H, monitor=monitor)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


@app_cli.command("loadtest")
def loadtest(
    rate: float = typer.Option(12.0, "--rate", help="alerts per hour"),
    duration: str = typer.Option("10m", "--duration"),
    out: str = typer.Option("reports/p9_latency.json", "--out"),
    window: int = typer.Option(144, "--window"),
    input_dim: int = typer.Option(16, "--input-dim"),
) -> None:
    # In-process Poisson generator and scorer (dummy model)
    total_s = 600 if duration.endswith("m") else 60
    if duration.endswith("m"):
        total_s = int(float(duration[:-1]) * 60)
    elif duration.endswith("s"):
        total_s = int(float(duration[:-1]))
    lam = rate / 3600.0  # per second
    inter_arrival = lambda: np.random.exponential(1 / lam)
    model = ModelSession(onnx_path=None, window=window, input_dim=input_dim, temperature=1.0)
    latencies = []
    t = 0.0
    while t < total_s:
        # generate request
        win = np.random.randn(window, input_dim).astype(np.float32)
        t0 = time.perf_counter()
        _ = model.predict_proba(win)
        latencies.append((time.perf_counter() - t0) * 1000)
        t += inter_arrival()
    lat = np.array(latencies)
    p50 = float(np.percentile(lat, 50)) if len(lat) else 0.0
    p99 = float(np.percentile(lat, 99)) if len(lat) else 0.0
    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"p50_ms": p50, "p99_ms": p99, "n": len(lat)}, f, indent=2)
    print(json.dumps({"p50_ms": p50, "p99_ms": p99, "n": len(lat)}))


@app_cli.command("monitor")
def monitor(tegrastats_log: str = typer.Option("logs/tegrastats.log", "--tegrastats-log")) -> None:
    Path(Path(tegrastats_log).parent).mkdir(parents=True, exist_ok=True)
    with open(tegrastats_log, "w") as f:
        if (shutil := __import__("shutil")) and shutil.which("tegrastats"):
            proc = subprocess.Popen(["tegrastats"], stdout=f)
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
        else:
            # Fallback: write synthetic metrics
            for _ in range(60):
                f.write("temp=55C gpu=35%\n"); f.flush(); time.sleep(1)


if __name__ == "__main__":
    app_cli()
