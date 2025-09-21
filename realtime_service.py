from __future__ import annotations

"""
Phase 9 – Edge Realtime Service (Jetson‑friendly)

This service exposes request/response APIs to return probabilities (/score) or
decisions (/decide) using a small ONNX FP16 model. It does NOT poll exchanges.
It expects a prepared 144×F feature window (+ ATR%) from a local store or via
request body.

Key features:
- GPU provider selection (TensorRT → CUDA → CPU) with calibration guard
- Jetson thermal/GPU throttling (tegrastats); fallback to LightGBM if present
- Idempotent /decide with LRU de‑dup and per‑key ordering
- Stage latency budgets (feat/nn/policy) and structured JSONL logging
- Non‑blocking explain queue (stub) per decision_id
- Optional DuckDB window resolver to fetch last 144 features
- Poisson load test to verify SLA (p50<500ms, p99<2000ms)
- Terminal UI (rich) to view last candles & signals (optional)

Config (YAML):
  runtime:
    symbols: [BTCUSDT]
    window: 144
    model_path: export/model_5m_fp16.onnx
    features_glob: data/features/5m/{symbol}/y=*/m=*/d=*/part-*.parquet
    atr_parquet: data/atr_5m.parquet
    latency_budget_sec: 600
    throttle_gpu_util_pct: 80
    throttle_temp_c: 70
"""

import asyncio
import hashlib
import json
import os
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import typer

from app.runtime.model_session import ModelSession
from app.runtime.onnx_introspect import infer_io
from app.runtime.throttle import Throttler
from app.runtime.metrics import ScoreDecideSink, aggregate_metrics
from policy_p7 import _ev_decision_row


app_cli = typer.Typer(help="Phase 9 – Edge realtime service")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LRUSet:
    def __init__(self, maxsize: int = 8192) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, None] = OrderedDict()

    def add(self, key: str) -> bool:
        if key in self._data:
            self._data.move_to_end(key)
            return False
        self._data[key] = None
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)
        return True


@dataclass
class RuntimeConfig:
    symbols: List[str]
    window: int
    model_path: str
    features_glob: str
    atr_parquet: Optional[str] = None
    latency_budget_sec: int = 600
    throttle_gpu_util_pct: float = 80.0
    throttle_temp_c: float = 70.0


def load_runtime_config(path: str) -> RuntimeConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    rt = cfg.get("runtime", cfg)
    return RuntimeConfig(
        symbols=list(rt.get("symbols", ["BTCUSDT"])),
        window=int(rt.get("window", 144)),
        model_path=str(rt.get("model_path", "export/model_5m_fp16.onnx")),
        features_glob=str(rt.get("features_glob", "data/features/5m/{symbol}/y=*/m=*/d=*/part-*.parquet")),
        atr_parquet=rt.get("atr_parquet"),
        latency_budget_sec=int(rt.get("latency_budget_sec", 600)),
        throttle_gpu_util_pct=float(rt.get("throttle_gpu_util_pct", 80)),
        throttle_temp_c=float(rt.get("throttle_temp_c", 70)),
    )


class FeatureWindowResolver:
    """Resolve last 144×F feature window (and atr_pct, vol_pctile) from DuckDB/Parquet."""

    def __init__(self, features_glob_tpl: str, window: int = 144) -> None:
        self.features_glob_tpl = features_glob_tpl
        self.window = window

    def fetch(self, symbol: str, ts: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        glob = self.features_glob_tpl.format(symbol=symbol)
        con = duckdb.connect()
        try:
            df = con.execute(f"SELECT * FROM read_parquet('{glob}') WHERE symbol='{symbol}' ORDER BY ts").df()
        finally:
            con.close()
        if df.empty or len(df) < self.window:
            raise RuntimeError("Not enough feature rows to build window")
        # Select contiguous tail of window rows
        win_df = df.tail(self.window)
        cols = [c for c in win_df.columns if c not in ("ts", "symbol") and np.issubdtype(win_df[c].dtype, np.number)]
        X = win_df[cols].to_numpy(dtype=np.float32)
        # Derive aux (fall back to reasonable defaults)
        aux = {
            "close": float(win_df.get("close", win_df[cols[0]]).iloc[-1]) if "close" in win_df.columns else 0.0,
            "atr_pct": float(win_df.get("atr_pct", 0.005).iloc[-1]) if "atr_pct" in win_df.columns else 0.005,
            "vol_pctile": float(win_df.get("vol_pctile", 0.5).iloc[-1]) if "vol_pctile" in win_df.columns else 0.5,
        }
        return X, aux


def build_api(model: ModelSession, cfg: RuntimeConfig) -> FastAPI:
    api = FastAPI()
    throttler = Throttler(thresh_temp=cfg.throttle_temp_c, thresh_gpu=cfg.throttle_gpu_util_pct)
    resolver = FeatureWindowResolver(cfg.features_glob, window=cfg.window)
    sd_sink = ScoreDecideSink("logs/score_decide.jsonl")
    # Idempotency & order
    recent = LRUSet(8192)
    key_locks: Dict[Tuple[str, str], asyncio.Lock] = defaultdict(asyncio.Lock)
    # Explain queue
    explain_q: asyncio.Queue = asyncio.Queue(maxsize=2048)
    explain_state: Dict[str, str] = {}

    async def _worker() -> None:
        while True:
            did, sym, ts = await explain_q.get()
            try:
                with open("logs/queue_explain.jsonl", "a") as f:
                    f.write(json.dumps({"ts": _now_iso(), "decision_id": did, "symbol": sym, "ts_close": ts, "state": "done"}) + "\n")
                explain_state[did] = "done"
            finally:
                explain_q.task_done()

    @api.on_event("startup")
    async def _startup() -> None:
        asyncio.create_task(_worker())

    @api.get("/health")
    def health() -> JSONResponse:
        h = throttler.read()
        return JSONResponse({
            "ok": True,
            "provider": model.provider,
            "is_calibrated": model.is_calibrated,
            "checksum": model.file_checksum(),
            "throttle": h.throttle,
            "temp_c": h.temp_c,
            "gpu_util": h.gpu_util,
            "queue_depth": explain_q.qsize(),
        })

    @api.get("/explain/status")
    def explain_status(id: str) -> JSONResponse:
        return JSONResponse({"id": id, "state": explain_state.get(id, "not_found")})

    @api.post("/score")
    async def score(payload: Dict[str, Any], request: Request) -> JSONResponse:
        win = np.array(payload.get("window", []), dtype=np.float32)
        if win.ndim != 2 or win.shape[0] != cfg.window:
            return JSONResponse({"error": f"window must be [W,F] with W={cfg.window}"}, status_code=400)
        t0 = time.perf_counter(); t_feat_ms = 0.0
        h = throttler.read()
        # inference
        t_nn0 = time.perf_counter()
        p = model.predict_proba(win)
        t_nn_ms = (time.perf_counter() - t_nn0) * 1000
        t_policy_ms = 0.0
        t_total_ms = (time.perf_counter() - t0) * 1000
        degraded = (t_nn_ms > 30.0)
        sd_sink.append({
            "ts": _now_iso(), "type": "score", "t_feat_ms": t_feat_ms, "t_nn_ms": t_nn_ms, "t_policy_ms": t_policy_ms, "t_total_ms": t_total_ms,
            "throttled": h.throttle, "provider": model.provider, "request_id": request.headers.get("X-Request-ID", "")
        })
        return JSONResponse({
            "p_wait": float(p[0]), "p_long": float(p[1]), "p_short": float(p[2]),
            "provider": model.provider, "throttled": h.throttle,
            "t_feat_ms": t_feat_ms, "t_nn_ms": t_nn_ms, "t_policy_ms": t_policy_ms, "t_total_ms": t_total_ms,
            "degraded": degraded,
        })

    @api.post("/decide")
    async def decide(payload: Dict[str, Any], request: Request) -> JSONResponse:
        symbol = str(payload.get("symbol", cfg.symbols[0]))
        ts_close = str(payload.get("ts", payload.get("candle_close_ts", "")))
        # Resolve window
        if "window" in payload:
            win = np.array(payload.get("window"), dtype=np.float32)
            aux = {
                "close": float(payload.get("close", 0.0)),
                "atr_pct": float(payload.get("atr_pct", 0.0)),
                "vol_pctile": float(payload.get("vol_pctile", 0.5)),
            }
        else:
            win, aux = resolver.fetch(symbol, ts_close)
        if win.shape[0] != cfg.window:
            return JSONResponse({"error": f"window must be [W,F] with W={cfg.window}"}, status_code=400)
        # decision id
        hsh = hashlib.sha256(); hsh.update(symbol.encode()); hsh.update(b"|"); hsh.update(ts_close.encode()); hsh.update(b"|"); hsh.update(win.tobytes())
        decision_id = hsh.hexdigest()[:16]
        key = (symbol, ts_close)
        lock = key_locks[key]
        async with lock:
            if not recent.add(decision_id):
                return JSONResponse({"duplicate": True, "decision_id": decision_id})
            t0 = time.perf_counter(); t_feat_ms = 0.0
            h = throttler.read(); throttled = h.throttle
            # inference (fallback to LGBM under throttle if available)
            t_nn0 = time.perf_counter()
            if throttled and model.has_lgbm():
                p = model.predict_proba_lgbm(win)
            else:
                p = model.predict_proba(win)
            t_nn_ms = (time.perf_counter() - t_nn0) * 1000
            # policy
            side, size, tp_px, sl_px, ev, reason = _ev_decision_row(
                float(p[1]), float(p[2]), float(aux["close"]), float(aux["atr_pct"]), 1.0, 1.5, float(aux["vol_pctile"]), type("C", (), {"cost_frac": lambda self: 0.0005})()
            )
            # throttle cushion τ+0.05
            tau = 0.5
            weak = max(float(p[1]), float(p[2])) < (tau + 0.05)
            if throttled and weak:
                side, size, reason = "WAIT", 0.0, "throttle"
            t_policy_ms = (time.perf_counter() - t_nn0) * 1000 - t_nn_ms
            t_total_ms = (time.perf_counter() - t0) * 1000
            degraded = (t_nn_ms > 30.0) or (t_policy_ms > 10.0)
            # queue explain
            try:
                explain_state[decision_id] = "queued"
                explain_q.put_nowait((decision_id, symbol, ts_close))
            except asyncio.QueueFull:
                explain_state.pop(decision_id, None)
            # log
            sd_sink.append({
                "ts": _now_iso(), "type": "decide", "decision_id": decision_id, "symbol": symbol, "ts_close": ts_close,
                "p_long": float(p[1]), "p_short": float(p[2]), "side": side, "ev_bps": float(ev) * 1e4,
                "t_feat_ms": t_feat_ms, "t_nn_ms": t_nn_ms, "t_policy_ms": t_policy_ms, "t_total_ms": t_total_ms,
                "throttled": throttled, "provider": model.provider, "request_id": request.headers.get("X-Request-ID", ""),
                "degraded": degraded,
            })
            return JSONResponse({
                "decision_id": decision_id, "side": side, "size": size, "TP_px": tp_px, "SL_px": sl_px, "EV": ev, "reason": reason,
                "provider": model.provider, "throttled": throttled, "t_feat_ms": t_feat_ms, "t_nn_ms": t_nn_ms, "t_policy_ms": t_policy_ms, "t_total_ms": t_total_ms,
                "degraded": degraded,
            })

    return api


@app_cli.command("api")
def api(conf: str = typer.Option("conf/runtime.yaml", "--conf"), port: int = typer.Option(8080, "--port")) -> None:
    cfg = load_runtime_config(conf)
    F = infer_io(cfg.model_path, expect_window=cfg.window)
    model = ModelSession(cfg.model_path, window=cfg.window, input_dim=F, apply_temp=False)
    app = build_api(model, cfg)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


@app_cli.command("loadtest")
def loadtest(
    conf: str = typer.Option("conf/runtime.yaml", "--conf"),
    rate: float = typer.Option(12.0, "--rate", help="events per hour"),
    duration: str = typer.Option("10m", "--duration"),
    out: str = typer.Option("reports/p9_latency.json", "--out"),
) -> None:
    cfg = load_runtime_config(conf)
    F = infer_io(cfg.model_path, expect_window=cfg.window)
    model = ModelSession(cfg.model_path, window=cfg.window, input_dim=F, apply_temp=False)
    sd_sink = ScoreDecideSink("logs/score_decide.jsonl")
    # Poisson arrivals
    total_s = 600
    if duration.endswith("m"): total_s = int(float(duration[:-1]) * 60)
    elif duration.endswith("s"): total_s = int(float(duration[:-1]))
    lam = (rate / 60.0) / 60.0
    rng = np.random.default_rng(42)
    t = 0.0
    while t < total_s:
        win = rng.standard_normal((cfg.window, F)).astype(np.float32)
        is_decide = (rng.random() < 0.2)
        t0 = time.perf_counter(); t_nn0 = time.perf_counter()
        p = model.predict_proba(win)
        t_nn_ms = (time.perf_counter() - t_nn0) * 1000
        t_policy_ms = 0.0
        if is_decide:
            _ = _ev_decision_row(float(p[1]), float(p[2]), 100_000.0, 0.005, 1.0, 1.5, 0.5, type("C", (), {"cost_frac": lambda self: 0.0005})())
        t_total_ms = (time.perf_counter() - t0) * 1000
        sd_sink.append({"ts": _now_iso(), "type": ("decide" if is_decide else "score"), "t_feat_ms": 0.0, "t_nn_ms": t_nn_ms, "t_policy_ms": (t_policy_ms if is_decide else 0.0), "t_total_ms": t_total_ms})
        if lam <= 0: break
        t += float(rng.exponential(1.0 / max(lam, 1e-9)))
    report = aggregate_metrics("logs/score_decide.jsonl", out)
    ok = (report.get("p50",{}).get("t_total_ms",1e9) < 500.0) and (report.get("p99",{}).get("t_total_ms",1e9) < 2000.0)
    print(json.dumps(report))
    raise SystemExit(0 if ok else 1)


@app_cli.command("terminal")
def terminal(conf: str = typer.Option("conf/runtime.yaml", "--conf"), refresh: int = typer.Option(300, "--refresh")) -> None:
    """Simple terminal UI to display latest candles + decisions (reads features)."""
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        Console = None
    cfg = load_runtime_config(conf)
    resolver = FeatureWindowResolver(cfg.features_glob, window=cfg.window)
    while True:
        try:
            X, aux = resolver.fetch(cfg.symbols[0])
            close = aux.get("close", 0.0)
            # Dummy signal coloring based on mock probability
            rng = np.random.default_rng()
            p_long = float(rng.uniform()); p_short = float(1.0 - p_long); side = "LONG" if p_long >= p_short else "SHORT"
            if Console:
                c = Console(); tbl = Table(title=f"{cfg.symbols[0]} — last {cfg.window} bars (close={close:.2f})")
                tbl.add_column("Field"); tbl.add_column("Value")
                tbl.add_row("Side", f"[green]{side}[/]" if side=="LONG" else f"[red]{side}[/]")
                tbl.add_row("p_long", f"{p_long:.3f}"); tbl.add_row("p_short", f"{p_short:.3f}")
                c.clear(); c.print(tbl)
            else:
                print(f"side={side} p_long={p_long:.3f} p_short={p_short:.3f} close={close:.2f}")
        except Exception as e:
            print("terminal error:", e)
        time.sleep(max(1, int(refresh)))


if __name__ == "__main__":
    app_cli()

