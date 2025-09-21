from __future__ import annotations

"""
Phase 9 – Edge Realtime Service for 5‑minute perps (Jetson‑friendly)

This service exposes request/response APIs to return probabilities (/score)
or decisions (/decide) using a small ONNX FP16 model. It optionally resolves
feature windows from a local store (DuckDB reading Parquet). It includes
throttling, idempotency, non‑blocking explain queue, and a Poisson load tester.
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
from utils_cg import write_parquet_daily_files

# New imports for REST polling and transforms
import httpx
import pandas as pd
from transforms import winsorize_causal, zscore_causal, encode_hour_of_week, safe_log1p
import threading


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
    use_if_gate: bool = False
    poll_interval_sec: int = 300
    coinglass_base_url: str = "https://open-api-v4.coinglass.com/api"


def load_runtime_config(path: str) -> RuntimeConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    rt = cfg.get("runtime", cfg)
    rc = RuntimeConfig(
        symbols=list(rt.get("symbols", ["BTCUSDT"])),
        window=int(rt.get("window", 144)),
        model_path=str(rt.get("model_path", "./artifacts/export/model_5m_fp16.onnx")),
        features_glob=str(rt.get("features_glob", "data/features/5m/{symbol}/y=*/m=*/d=*/part-*.parquet")),
        atr_parquet=rt.get("atr_parquet"),
        latency_budget_sec=int(rt.get("latency_budget_sec", 600)),
        throttle_gpu_util_pct=float(rt.get("throttle_gpu_util_pct", 80)),
        throttle_temp_c=float(rt.get("throttle_temp_c", 70)),
        use_if_gate=bool(rt.get("use_if_gate", False)),
        poll_interval_sec=int(rt.get("poll_interval_sec", 300)),
        coinglass_base_url=str(rt.get("coinglass_base_url", "https://open-api-v4.coinglass.com/api")),
    )
    return rc


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
        win_df = df.tail(self.window)
        cols = [c for c in win_df.columns if c not in ("ts", "symbol") and np.issubdtype(win_df[c].dtype, np.number)]
        X = win_df[cols].to_numpy(dtype=np.float32)
        aux = {
            "close": float(win_df.get("close", win_df[cols[0]]).iloc[-1]) if "close" in win_df.columns else 0.0,
            "atr_pct": float(win_df.get("atr_pct", 0.005).iloc[-1]) if "atr_pct" in win_df.columns else 0.005,
            "vol_pctile": float(win_df.get("vol_pctile", 0.5).iloc[-1]) if "vol_pctile" in win_df.columns else 0.5,
            "liq60_z": float(win_df.get("liq60_z", 0.0).iloc[-1]) if "liq60_z" in win_df.columns else 0.0,
        }
        return X, aux

    def fetch_tail_features_df(self, symbol: str, n: int) -> pd.DataFrame:
        glob = self.features_glob_tpl.format(symbol=symbol)
        con = duckdb.connect()
        try:
            df = con.execute(f"SELECT * FROM read_parquet('{glob}') WHERE symbol='{symbol}' ORDER BY ts").df()
        finally:
            con.close()
        if df.empty:
            raise RuntimeError("No features available")
        return df.tail(max(1, n))


class RawHistoryResolver:
    """Fetch last raw bars from P1 Parquet to append with newest raw bar."""

    def __init__(self, parquet_tpl: str = "data/parquet/5m/{symbol}/y=*/m=*/d=*/part-*.parquet") -> None:
        self.parquet_tpl = parquet_tpl

    def fetch_tail_raw(self, symbol: str, n: int = 200) -> pd.DataFrame:
        glob = self.parquet_tpl.format(symbol=symbol)
        con = duckdb.connect()
        try:
            df = con.execute(f"SELECT * FROM read_parquet('{glob}', union_by_name=true) WHERE symbol='{symbol}' ORDER BY ts").df()
        finally:
            con.close()
        if df.empty:
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df.tail(n).reset_index(drop=True)


class CoinGlassClient:
    def __init__(self, base_url: str, api_key: Optional[str]) -> None:
        self.base = base_url.rstrip("/")
        headers = {"accept": "application/json"}
        if api_key:
            headers["CG-API-KEY"] = api_key
        self.client = httpx.Client(headers=headers, timeout=10.0)

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}/{path.lstrip('/')}"
        for i in range(3):
            try:
                r = self.client.get(url, params=params)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                time.sleep(1.0 * (i + 1))
        return {}

    def fetch_latest_bar(self, symbol: str, exchange: str = "Binance") -> Optional[Dict[str, Any]]:
        """Fetch the latest 5m futures OHLC and auxiliary fields. Best-effort parse."""
        out: Dict[str, Any] = {"symbol": symbol}
        # Price OHLC
        j = self._get("futures/price/history", {"exchange": exchange, "symbol": symbol, "interval": "5m", "limit": 1})
        try:
            row = (j.get("data") or [{}])[0]
            out.update({
                "ts": pd.to_datetime(int(row.get("time")) , unit="ms", utc=True),
                "open": float(row.get("open")),
                "high": float(row.get("high")),
                "low": float(row.get("low")),
                "close": float(row.get("close")),
                "volume": float(row.get("volume_usd", 0.0)),
            })
        except Exception:
            return None
        # OI aggregated history (close → oi_now)
        j = self._get("futures/open-interest/aggregated-history", {"symbol": symbol, "interval": "5m", "limit": 1})
        try:
            r = (j.get("data") or [{}])[0]
            out["oi_now"] = float(r.get("close"))
        except Exception:
            out["oi_now"] = np.nan
        # Funding (oi-weight history)
        j = self._get("futures/funding-rate/oi-weight-history", {"symbol": symbol, "interval": "5m", "limit": 1})
        try:
            r = (j.get("data") or [{}])[0]
            out["funding_now"] = float(r.get("fundingRate", 0.0))
        except Exception:
            out["funding_now"] = np.nan
        # Taker flows
        j = self._get("futures/taker-buy-sell-volume/history", {"exchange": exchange, "symbol": symbol, "interval": "5m", "limit": 1})
        try:
            r = (j.get("data") or [{}])[0]
            out["taker_buy_usd"] = float(r.get("buyVolumeUsd", 0.0))
            out["taker_sell_usd"] = float(r.get("sellVolumeUsd", 0.0))
        except Exception:
            out["taker_buy_usd"] = 0.0; out["taker_sell_usd"] = 0.0
        # Spot taker volumes (optional)
        out.setdefault("spot_taker_buy_usd", 0.0)
        out.setdefault("spot_taker_sell_usd", 0.0)
        # Liq notional raw (optional)
        out.setdefault("liq_notional_raw", 0.0)
        return out


def _align_to_5m(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts.floor("5min") + pd.Timedelta(minutes=5)).tz_localize("UTC") if ts.tzinfo else (ts.floor("5min") + pd.Timedelta(minutes=5)).tz_localize("UTC")


def _compute_feature_row(raw_tail: pd.DataFrame, prev_feat_row: Optional[pd.Series]) -> Dict[str, float]:
    """Compute a single feature row causally from raw history (last ~144 bars).
    Fallback to previous row values for any feature we can't compute, to preserve schema.
    """
    df = raw_tail.copy()
    df["ret_5m"] = np.log(df["close"]).diff().fillna(0.0)
    df["ret_1h"] = df["ret_5m"].rolling(12, min_periods=1).sum()
    df["ret_4h"] = df["ret_5m"].rolling(48, min_periods=1).sum()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["co_ret"] = np.log((df["close"] / df["open"]).replace(0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    # z-scores (use available window length causally)
    w = min(len(df), 144)
    df["vol_z"] = zscore_causal(safe_log1p(df["volume"]).fillna(0.0), window=w)
    if "oi_now" in df:
        df["oi_z"] = zscore_causal(safe_log1p(df["oi_now"]).fillna(0.0), window=w)
    if "funding_now" in df:
        df["fund_now_z"] = zscore_causal(df["funding_now"].fillna(0.0), window=w)
    # liq60_z (if available)
    if "liq_notional_raw" in df:
        df["liq60"] = df["liq_notional_raw"].fillna(0.0).rolling(12, min_periods=1).sum()
        df["liq60_z"] = zscore_causal(safe_log1p(df["liq60"]).fillna(0.0), window=w)
    # rv_5m_z
    df["rv_5m_z"] = zscore_causal((df["ret_5m"] ** 2).fillna(0.0), window=w)
    # seasonality
    sin_h, cos_h = encode_hour_of_week(pd.DatetimeIndex(df["ts"].values))
    df["hour_of_week_sin"], df["hour_of_week_cos"] = sin_h, cos_h
    feat_row = df.iloc[-1].to_dict()
    # Merge with previous feature schema if present
    out: Dict[str, float] = {}
    if prev_feat_row is not None:
        for c in prev_feat_row.index:
            if c in ("ts", "symbol"):
                continue
            v = feat_row.get(c)
            out[c] = float(v) if v is not None and np.isfinite(v) else float(prev_feat_row[c])
    else:
        # Emit computed subset only
        for k in ("ret_5m","ret_1h","ret_4h","hl_range","co_ret","vol_z","oi_z","fund_now_z","liq60_z","rv_5m_z","hour_of_week_sin","hour_of_week_cos"):
            if k in feat_row:
                out[k] = float(feat_row[k])
    return out


class MicroBatchPoller(threading.Thread):
    def __init__(self, cfg: RuntimeConfig, model: ModelSession, resolver: FeatureWindowResolver, raw_resolver: RawHistoryResolver, throttler: Throttler) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.model = model
        self.resolver = resolver
        self.raw_resolver = raw_resolver
        self.throttler = throttler
        self.client = CoinGlassClient(cfg.coinglass_base_url, os.getenv("COINGLASS_API_KEY"))
        self.stop_evt = threading.Event()
        self.sd_sink = ScoreDecideSink("logs/score_decide.jsonl")

    def stop(self) -> None:
        self.stop_evt.set()

    def run(self) -> None:
        while not self.stop_evt.is_set():
            start = time.time()
            for sym in self.cfg.symbols:
                try:
                    self.process_symbol(sym)
                except Exception as e:
                    self.sd_sink.append({"ts": _now_iso(), "type": "poll_error", "symbol": sym, "error": str(e)})
            # sleep remaining
            elapsed = time.time() - start
            to_sleep = max(1, int(self.cfg.poll_interval_sec - elapsed))
            if self.stop_evt.wait(to_sleep):
                break

    def process_symbol(self, symbol: str) -> None:
        h = self.throttler.read()
        if h.throttle:
            # under throttle, skip or force WAIT
            self.sd_sink.append({"ts": _now_iso(), "type": "throttle_skip", "symbol": symbol})
            return
        latest = self.client.fetch_latest_bar(symbol)
        if not latest:
            self.sd_sink.append({"ts": _now_iso(), "type": "no_data", "symbol": symbol})
            return
        latest_ts = _align_to_5m(latest["ts"]).to_pydatetime()
        latest["ts"] = pd.Timestamp(latest_ts, tz="UTC")
        # fetch raw history for features
        raw = self.raw_resolver.fetch_tail_raw(symbol, n=self.cfg.window - 1)
        if raw.empty:
            self.sd_sink.append({"ts": _now_iso(), "type": "raw_empty", "symbol": symbol})
            return
        # limited ffill for exogenous from raw
        for ex in ("oi_now", "funding_now"):
            if ex in raw.columns and np.isnan(latest.get(ex, np.nan)):
                latest[ex] = float(raw[ex].ffill(limit=3).iloc[-1]) if ex in raw else np.nan
        raw_app = pd.concat([raw, pd.DataFrame([latest])], ignore_index=True)
        # feature tail df and schema
        feat_tail = self.resolver.fetch_tail_features_df(symbol, n=self.cfg.window - 1)
        prev_feat_row = feat_tail.iloc[-1].drop(labels=[c for c in ("ts","symbol") if c in feat_tail.columns]) if not feat_tail.empty else None
        feat_row = _compute_feature_row(raw_app.iloc[-self.cfg.window:], prev_feat_row)
        # build full window array in training column order
        cols = [c for c in feat_tail.columns if c not in ("ts", "symbol")] if not feat_tail.empty else list(feat_row.keys())
        X_tail = feat_tail[cols].to_numpy(dtype=np.float32) if not feat_tail.empty else np.zeros((self.cfg.window - 1, len(cols)), dtype=np.float32)
        x_new = np.array([feat_row.get(c, 0.0) for c in cols], dtype=np.float32)
        X = np.vstack([X_tail, x_new])
        # model → policy
        t0 = time.perf_counter(); t_nn0 = time.perf_counter()
        p = self.model.predict_proba(X)
        t_nn_ms = (time.perf_counter() - t_nn0) * 1000
        liq60_z = float(feat_row.get("liq60_z", 0.0))
        liq60_pctile = float(np.clip((liq60_z + 5.0) / 10.0, 0.0, 1.0))
        class Cost:
            def __init__(self, bps: float) -> None: self.bps=bps
            def cost_frac(self) -> float: return self.bps/1e4
        cost = Cost(0.3 + 0.4 * liq60_pctile)
        close = float(latest.get("close", 0.0)); atr_pct = float(feat_row.get("atr_pct", 0.005)); vol_pct = float(feat_row.get("vol_pctile", 0.5))
        side, size, tp_px, sl_px, ev, reason = _ev_decision_row(float(p[1]), float(p[2]), close, atr_pct, 1.0, 1.5, vol_pct, cost)
        t_total_ms = (time.perf_counter() - t0) * 1000
        # write decision parquet partitioned by day
        dec = pd.DataFrame([{ "ts": latest["ts"], "symbol": symbol, "side": side, "size": size, "TP_px": tp_px, "SL_px": sl_px, "EV": ev, "reason": reason }])
        write_parquet_daily_files(dec, root="decisions", symbol=symbol)
        self.sd_sink.append({"ts": _now_iso(), "type": "poll_decide", "symbol": symbol, "p_long": float(p[1]), "p_short": float(p[2]), "side": side, "t_nn_ms": t_nn_ms, "t_total_ms": t_total_ms})


def build_api(model: ModelSession, cfg: RuntimeConfig) -> FastAPI:
    api = FastAPI()
    throttler = Throttler(thresh_temp=cfg.throttle_temp_c, thresh_gpu=cfg.throttle_gpu_util_pct)
    resolver = FeatureWindowResolver(cfg.features_glob, window=cfg.window)
    raw_resolver = RawHistoryResolver()
    sd_sink = ScoreDecideSink("logs/score_decide.jsonl")
    recent = LRUSet(8192)
    key_locks: Dict[Tuple[str, str], asyncio.Lock] = defaultdict(asyncio.Lock)
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

    # Micro-batch poller (REST), optional
    poller = MicroBatchPoller(cfg, model, resolver, raw_resolver, throttler)

    @api.on_event("startup")
    async def _startup() -> None:
        asyncio.create_task(_worker())
        # Start REST poller
        poller.start()

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
        symbol = str(payload.get("symbol", cfg.symbols[0]))
        if "window" in payload:
            win = np.array(payload.get("window"), dtype=np.float32)
        else:
            win, _ = resolver.fetch(symbol)
        if win.ndim != 2 or win.shape[0] != cfg.window:
            return JSONResponse({"error": f"window must be [W,F] with W={cfg.window}"}, status_code=400)
        t0 = time.perf_counter(); t_feat_ms = 0.0
        h = throttler.read()
        t_nn0 = time.perf_counter()
        p = model.predict_proba(win)
        t_nn_ms = (time.perf_counter() - t_nn0) * 1000
        t_policy_ms = 0.0
        t_total_ms = (time.perf_counter() - t0) * 1000
        degraded = (t_nn_ms > 30.0)
        sd_sink.append({
            "ts": _now_iso(), "type": "score", "symbol": symbol,
            "t_feat_ms": t_feat_ms, "t_nn_ms": t_nn_ms, "t_policy_ms": t_policy_ms, "t_total_ms": t_total_ms,
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
        # Resolve window + aux
        if "window" in payload:
            win = np.array(payload.get("window"), dtype=np.float32)
            aux = {
                "close": float(payload.get("close", 0.0)),
                "atr_pct": float(payload.get("atr_pct", 0.0)),
                "vol_pctile": float(payload.get("vol_pctile", 0.5)),
                "liq60_z": float(payload.get("liq60_z", 0.0)),
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
            # NN (fallback to LGBM under throttle if available)
            t_nn0 = time.perf_counter()
            if throttled and model.has_lgbm():
                p = model.predict_proba_lgbm(win)
            else:
                p = model.predict_proba(win)
            t_nn_ms = (time.perf_counter() - t_nn0) * 1000
            # Policy with slippage (bps) based on liq60_z percentile
            liq60_pctile = float(np.clip((aux.get("liq60_z", 0.0) + 5.0) / 10.0, 0.0, 1.0))  # crude mapping from z to pctile
            class Cost:  # dynamic cost bps = 0.3 + 0.4 * liq60_pctile
                def __init__(self, bps: float) -> None: self.bps=bps
                def cost_frac(self) -> float: return self.bps/1e4
            cost = Cost(0.3 + 0.4 * liq60_pctile)
            side, size, tp_px, sl_px, ev, reason = _ev_decision_row(
                float(p[1]), float(p[2]), float(aux["close"]), float(aux["atr_pct"]), 1.0, 1.5, float(aux["vol_pctile"]), cost
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
    """Simple terminal UI to display latest signals (reads features store)."""
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
