from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import typer

from policy_p7 import _ev_decision_row
from app.runtime.model_session import ModelSession
from app.runtime.onnx_introspect import infer_io
from app.runtime.throttle import Throttler, load_tau
from app.runtime.metrics import MetricsSink, aggregate_metrics, ScoreDecideSink


app_cli = typer.Typer(help="P9 – Real-time scoring & decision service (Jetson)")


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


metrics_path = Path("logs/edge_metrics.jsonl")


class LRUSet:
    def __init__(self, maxsize: int = 4096) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, None] = OrderedDict()

    def add(self, key: str) -> bool:
        """Add key; return True if newly added; False if it existed (duplicate)."""
        if key in self._data:
            self._data.move_to_end(key)
            return False
        self._data[key] = None
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)
        return True


def create_app(model: ModelSession, k_min: float, k_max: float, H: int, monitor: Optional[Throttler] = None, taus: Optional[dict] = None) -> FastAPI:
    api = FastAPI()
    sink = MetricsSink(metrics_path)
    sd_sink = ScoreDecideSink("logs/score_decide.jsonl")
    tau_long = float(taus.get("tau_long", 0.55)) if taus else 0.55
    tau_short = float(taus.get("tau_short", 0.55)) if taus else 0.55
    # Idempotency and ordering
    recent_ids = LRUSet(maxsize=8192)
    key_locks: Dict[Tuple[str, str], asyncio.Lock] = defaultdict(asyncio.Lock)
    # Explain queue
    explain_q: asyncio.Queue = asyncio.Queue(maxsize=2048)
    explain_state: Dict[str, str] = {}

    async def _explain_worker() -> None:
        while True:
            did, sym, ts, wh = await explain_q.get()
            try:
                # Placeholder: just log queue processing
                rec = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "decision_id": did,
                    "symbol": sym,
                    "ts_close": ts,
                    "window_hash": wh,
                    "state": "done",
                }
                Path("logs").mkdir(exist_ok=True)
                with open("logs/queue_explain.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
                explain_state[did] = "done"
            finally:
                explain_q.task_done()

    @api.on_event("startup")
    async def _startup() -> None:
        asyncio.create_task(_explain_worker())

    @api.get("/health")
    def health() -> JSONResponse:
        hs = monitor.read() if monitor else None
        return JSONResponse({
            "ok": True,
            "provider": model.provider,
            "model_checksum": model.file_checksum(),
            "is_calibrated": model.is_calibrated,
            "throttle": (hs.throttle if hs else False),
            "queue_depth": explain_q.qsize(),
        })

    @api.get("/explain/status")
    def explain_status(id: str) -> JSONResponse:
        state = explain_state.get(id, "not_found")
        return JSONResponse({"id": id, "state": state})

    @api.post("/score")
    async def score(payload: Dict[str, Any], request: Request) -> JSONResponse:
        # Expect payload: {"window": [[...W x F...]], "meta": {...}}
        win = np.array(payload.get("window", []), dtype=np.float32)
        if win.ndim != 2:
            return JSONResponse({"error": "window must be 2D [W,F]"}, status_code=400)
        t_total0 = time.perf_counter()
        t_feat_ms = 0.0
        # NN stage
        t_nn0 = time.perf_counter()
        hs = monitor.read() if monitor else None
        throttled = bool(hs.throttle) if hs else False
        # No fallback for /score; only annotate throttle
        p = model.predict_proba(win)
        t_nn_ms = (time.perf_counter() - t_nn0) * 1000
        t_policy_ms = 0.0
        t_total_ms = (time.perf_counter() - t_total0) * 1000
        # Budgets
        degraded = False
        exceed = None
        if t_feat_ms > 20: degraded, exceed = True, "feat"
        if t_nn_ms > 30: degraded, exceed = True, "nn"
        if t_policy_ms > 10: degraded, exceed = True, "policy"
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": "score",
            "throttled": throttled,
            "t_feat_ms": t_feat_ms,
            "t_nn_ms": t_nn_ms,
            "t_policy_ms": t_policy_ms,
            "t_total_ms": t_total_ms,
            "request_id": request.headers.get("X-Request-ID", ""),
            "degraded": degraded,
            "exceed_stage": exceed,
        }
        sd_sink.append({**rec, "p_long": float(p[1]), "p_short": float(p[2])})
        return JSONResponse({
            "p_wait": float(p[0]),
            "p_long": float(p[1]),
            "p_short": float(p[2]),
            "provider": model.provider,
            "throttled": throttled,
            **{k: rec[k] for k in ("t_feat_ms", "t_nn_ms", "t_policy_ms", "t_total_ms", "degraded", "exceed_stage")},
        })

    @api.post("/decide")
    async def decide(payload: Dict[str, Any], request: Request) -> JSONResponse:
        win = np.array(payload.get("window", []), dtype=np.float32)
        if win.ndim != 2 or win.shape[0] != model.window:
            return JSONResponse({"error": f"window must be [W,F] with W={model.window}"}, status_code=400)
        symbol = str(payload.get("symbol", "BTCUSDT"))
        ts_close = str(payload.get("ts", payload.get("candle_close_ts", "")))
        # decision_id for idempotency
        h = hashlib.sha256()
        h.update(symbol.encode())
        h.update(b"|")
        h.update(str(ts_close).encode())
        h.update(b"|")
        h.update(win.tobytes())
        decision_id = h.hexdigest()[:16]
        req_id = request.headers.get("X-Request-ID", "")
        key = (symbol, ts_close)
        lock = key_locks[key]
        async with lock:
            # Duplicate detection
            if not recent_ids.add(decision_id):
                return JSONResponse({"duplicate": True, "decision_id": decision_id}, status_code=200)

            close = float(payload.get("close", 0.0))
            atr_pct = float(payload.get("atr_pct", 0.0))
            vol_pct = float(payload.get("vol_pctile", 0.5))
            t_total0 = time.perf_counter()
            t_feat_ms = 0.0
            # Monitor
            hs = monitor.read() if monitor else None
            throttled = bool(hs.throttle) if hs else False
            # Inference
            t_nn0 = time.perf_counter()
            if throttled and model.has_lgbm():
                p = model.predict_proba_lgbm(win)
            else:
                p = model.predict_proba(win)
            t_nn_ms = (time.perf_counter() - t_nn0) * 1000
            # Policy
            side, size, tp_px, sl_px, ev, reason = _ev_decision_row(
                float(p[1]), float(p[2]), close, atr_pct, k_min, k_max, vol_pct, type("C", (), {"cost_frac": lambda self: 0.0005})()
            )
            t_policy_ms = (time.perf_counter() - t_nn0) * 1000 - t_nn_ms
            # Throttle τ+0.05 cushion
            maxp = float(max(p[1], p[2]))
            side_pref = "LONG" if p[1] >= p[2] else "SHORT"
            tau_side = tau_long if side_pref == "LONG" else tau_short
            weak = maxp < (tau_side + 0.05)
            if throttled and weak:
                side, size, reason = "WAIT", 0.0, "throttle"
            t_total_ms = (time.perf_counter() - t_total0) * 1000
            degraded = False
            exceed = None
            if t_feat_ms > 20: degraded, exceed = True, "feat"
            if t_nn_ms > 30: degraded, exceed = True, "nn"
            if t_policy_ms > 10: degraded, exceed = True, "policy"
            # Queue non-blocking explain
            win_hash = hashlib.sha256(win.tobytes()).hexdigest()[:16]
            explain_state[decision_id] = "queued"
            try:
                explain_q.put_nowait((decision_id, symbol, ts_close, win_hash))
            except asyncio.QueueFull:
                explain_state.pop(decision_id, None)
            # Log
            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "type": "decide",
                "decision_id": decision_id,
                "symbol": symbol,
                "ts_close": ts_close,
                "throttled": throttled,
                "p_long": float(p[1]),
                "p_short": float(p[2]),
                "side": side,
                "ev_bps": float(ev) * 1e4,
                "t_feat_ms": t_feat_ms,
                "t_nn_ms": t_nn_ms,
                "t_policy_ms": t_policy_ms,
                "t_total_ms": t_total_ms,
                "degraded": degraded,
                "exceed_stage": exceed,
                "request_id": req_id,
            }
            sd_sink.append(rec)
            return JSONResponse({
                "decision_id": decision_id,
                "side": side,
                "size": size,
                "TP_px": tp_px,
                "SL_px": sl_px,
                "EV": ev,
                "reason": reason,
                "provider": model.provider,
                "throttled": throttled,
                **{k: rec[k] for k in ("t_feat_ms", "t_nn_ms", "t_policy_ms", "t_total_ms", "degraded", "exceed_stage")},
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
    thresh_temp = float(os.getenv("THERM_TEMP", "70"))
    thresh_gpu = float(os.getenv("THERM_GPU", "80"))
    throttler = Throttler(thresh_temp=thresh_temp, thresh_gpu=thresh_gpu)
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
    # Poisson arrivals with mixed traffic (80% /score, 20% /decide)
    total_s = 600
    if duration.endswith("m"):
        total_s = int(float(duration[:-1]) * 60)
    elif duration.endswith("s"):
        total_s = int(float(duration[:-1]))
    lam = rate / 60.0  # per minute → per second below
    lam = lam / 60.0
    rng = np.random.default_rng(42)
    if onnx:
        F = infer_io(onnx, expect_window=window)
        model = ModelSession(onnx, window=window, input_dim=F, apply_temp=False)
    else:
        model = ModelSession(None, window=window, input_dim=16, apply_temp=False)
    sd_sink = ScoreDecideSink("logs/score_decide.jsonl")
    t = 0.0
    while t < total_s:
        win = rng.standard_normal((window, model.input_dim)).astype(np.float32)
        is_decide = (rng.random() < 0.2)
        t0 = time.perf_counter()
        t_nn0 = time.perf_counter()
        p = model.predict_proba(win)
        t_nn_ms = (time.perf_counter() - t_nn0) * 1000
        t_policy_ms = 0.0
        if is_decide:
            # Minimal policy compute
            side, size, tp_px, sl_px, ev, reason = _ev_decision_row(
                float(p[1]), float(p[2]), 100_000.0, 0.005, 1.0, 1.5, 0.5, type("C", (), {"cost_frac": lambda self: 0.0005})()
            )
        t_total_ms = (time.perf_counter() - t0) * 1000
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": ("decide" if is_decide else "score"),
            "t_feat_ms": 0.0,
            "t_nn_ms": t_nn_ms,
            "t_policy_ms": (t_policy_ms if is_decide else 0.0),
            "t_total_ms": t_total_ms,
        }
        sd_sink.append(rec)
        if lam <= 0:
            break
        t += float(rng.exponential(1.0 / max(lam, 1e-9)))
    report = aggregate_metrics("logs/score_decide.jsonl", out)
    overall_ok = (report.get("p50", {}).get("t_total_ms", 1e9) < 500.0) and (report.get("p99", {}).get("t_total_ms", 1e9) < 2000.0)
    print(json.dumps(report))
    raise SystemExit(0 if overall_ok else 1)


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
            for _ in range(60):
                f.write("temp=55C gpu=35%\n"); f.flush(); time.sleep(1)


if __name__ == "__main__":
    app_cli()
