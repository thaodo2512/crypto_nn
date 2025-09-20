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
from app.runtime.model_session import ModelSession
from app.runtime.onnx_introspect import infer_io
from app.runtime.throttle import Throttler, load_tau
from app.runtime.metrics import MetricsSink, aggregate_metrics


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


metrics_path = Path("logs/edge_metrics.jsonl")


def create_app(model: ModelSession, k_min: float, k_max: float, H: int, monitor: Optional[Throttler] = None, taus: Optional[dict] = None) -> FastAPI:
    api = FastAPI()
    sink = MetricsSink(metrics_path)

    tau_long = float(taus.get("tau_long", 0.55)) if taus else 0.55
    tau_short = float(taus.get("tau_short", 0.55)) if taus else 0.55

    @api.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    @api.post("/score")
    def score(payload: Dict[str, Any]) -> JSONResponse:
        # Expect payload: {"window": [[...W x F...]], "meta": {...}}
        win = np.array(payload.get("window", []), dtype=np.float32)
        if win.ndim != 2:
            return JSONResponse({"error": "window must be 2D [W,F]"}, status_code=400)
        t_total0 = time.perf_counter()
        t_feat_ms = 0.0
        t_nn0 = time.perf_counter()
        p = model.predict_proba(win)
        t_nn_ms = (time.perf_counter() - t_nn0) * 1000
        t_policy_ms = 0.0
        t_total_ms = (time.perf_counter() - t_total0) * 1000
        rec = {"t_feat_ms": t_feat_ms, "t_nn_ms": t_nn_ms, "t_policy_ms": t_policy_ms, "t_total_ms": t_total_ms}
        sink.append({"type": "score", **rec})
        return JSONResponse({"p_wait": float(p[0]), "p_long": float(p[1]), "p_short": float(p[2]), **rec})

    @api.post("/decide")
    def decide(payload: Dict[str, Any]) -> JSONResponse:
        win = np.array(payload.get("window", []), dtype=np.float32)
        if win.ndim != 2 or win.shape[0] != model.window:
            return JSONResponse({"error": f"window must be [W,F] with W={model.window}"}, status_code=400)
        close = float(payload.get("close", 0.0))
        atr_pct = float(payload.get("atr_pct", 0.0))
        vol_pct = float(payload.get("vol_pctile", 0.5))
        candle_close_ts = payload.get("candle_close_ts")
        t_total0 = time.perf_counter()
        t_feat_ms = 0.0
        # Monitor readings
        hs = monitor.read() if monitor else None
        t_nn0 = time.perf_counter()
        p = model.predict_proba(win)
        t_nn_ms = (time.perf_counter() - t_nn0) * 1000
        side, size, tp_px, sl_px, ev, reason = _ev_decision_row(
            float(p[1]), float(p[2]), close, atr_pct, k_min, k_max, vol_pct, type("C", (), {"cost_frac": lambda self: 0.0005})()
        )
        t_policy_ms = (time.perf_counter() - t_nn0) * 1000 - t_nn_ms
        # Throttle based on taus with 2% cushion
        maxp = float(max(p[1], p[2]))
        side_pref = "LONG" if p[1] >= p[2] else "SHORT"
        tau_side = tau_long if side_pref == "LONG" else tau_short
        weak = maxp < (tau_side + 0.02)
        throttled = monitor.should_throttle(hs.temp_c, hs.gpu_util) if monitor and hs else False
        if throttled and weak:
            side, size, reason = "WAIT", 0.0, "throttle"
        t_total_ms = (time.perf_counter() - t_total0) * 1000
        rec = {"t_feat_ms": t_feat_ms, "t_nn_ms": t_nn_ms, "t_policy_ms": t_policy_ms, "t_total_ms": t_total_ms}
        sink.append({"type": "decide", **rec})
        # SLA logging
        if candle_close_ts:
            try:
                cc = pd.to_datetime(candle_close_ts, utc=True)
                now = pd.Timestamp.utcnow().tz_localize("UTC")
                delta_min = float((now - cc).total_seconds() / 60.0)
                ok = bool(delta_min <= 10.0)
                sink.append_sla(now.isoformat(), str(cc), delta_min, ok)
            except Exception:
                pass
        return JSONResponse({
            "side": side,
            "size": size,
            "TP_px": tp_px,
            "SL_px": sl_px,
            "EV": ev,
            "reason": reason,
            **rec,
        })

    return api


@app_cli.command("api")
def api(
    onnx_path: Optional[str] = typer.Option(None, "--onnx", help="ONNX model path (optional; dummy if missing)"),
    window: int = typer.Option(144, "--window"),
    apply_temp: bool = typer.Option(False, "--apply-temp/--no-apply-temp"),
    k_min: float = typer.Option(1.0, "--k-min"),
    k_max: float = typer.Option(1.5, "--k-max"),
    H: int = typer.Option(36, "--H"),
    port: int = typer.Option(8080, "--port"),
) -> None:
    taus = load_tau("reports/calibration_metrics.json")
    throttler = Throttler(thresh_temp=70.0, thresh_gpu=80.0)
    if onnx_path:
        F = infer_io(onnx_path, expect_window=window)
        model = ModelSession(onnx_path, window=window, input_dim=F, apply_temp=apply_temp)
    else:
        model = ModelSession(None, window=window, input_dim=16, apply_temp=False)
    app = create_app(model, k_min=k_min, k_max=k_max, H=H, monitor=throttler, taus=taus)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


@app_cli.command("loadtest")
def loadtest(
    rate: float = typer.Option(12.0, "--rate", help="alerts per hour"),
    duration: str = typer.Option("10m", "--duration"),
    out: str = typer.Option("reports/p9_latency.json", "--out"),
    window: int = typer.Option(144, "--window"),
    onnx: Optional[str] = typer.Option(None, "--onnx"),
) -> None:
    # In-process Poisson generator and scorer with real ONNX if provided
    total_s = 600 if duration.endswith("m") else 60
    if duration.endswith("m"):
        total_s = int(float(duration[:-1]) * 60)
    elif duration.endswith("s"):
        total_s = int(float(duration[:-1]))
    lam = rate / 3600.0  # per second
    inter_arrival = lambda: np.random.exponential(1 / lam) if lam > 0 else 0
    if onnx:
        F = infer_io(onnx, expect_window=window)
        model = ModelSession(onnx, window=window, input_dim=F, apply_temp=False)
    else:
        model = ModelSession(None, window=window, input_dim=16, apply_temp=False)
    sink = MetricsSink(metrics_path)
    t = 0.0
    n = 0
    while t < total_s:
        win = np.random.randn(window, model.input_dim).astype(np.float32)
        t_total0 = time.perf_counter()
        t_feat_ms = 0.0
        t_nn0 = time.perf_counter()
        p = model.predict_proba(win)
        t_nn_ms = (time.perf_counter() - t_nn0) * 1000
        t_policy_ms = 0.0
        t_total_ms = (time.perf_counter() - t_total0) * 1000
        sink.append({"type": "loadtest", "t_feat_ms": t_feat_ms, "t_nn_ms": t_nn_ms, "t_policy_ms": t_policy_ms, "t_total_ms": t_total_ms})
        n += 1
        if lam <= 0:
            break
        t += inter_arrival()
    # Aggregate metrics
    report = aggregate_metrics(metrics_path, out)
    print(json.dumps(report))


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
