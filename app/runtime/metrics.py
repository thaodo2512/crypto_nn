from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class MetricsSink:
    """Lightweight JSONL sink for generic metrics and SLA records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sla_path = self.path.parent / "sla.jsonl"

    def append(self, rec: Dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def append_sla(self, ts: str, candle_close_ts: str, delta_min: float, ok: bool) -> None:
        with open(self.sla_path, "a") as f:
            f.write(json.dumps({
                "ts": ts,
                "candle_close_ts": candle_close_ts,
                "delta_min": delta_min,
                "ok": ok,
            }) + "\n")


class ScoreDecideSink:
    """Structured JSONL writer for /score and /decide events."""

    def __init__(self, path: Path | str = "logs/score_decide.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: Dict) -> None:
        if "ts" not in rec:
            rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def _window_filter(times: List[float], vals: List[float], horizon_s: float, now_s: float) -> List[float]:
    lo = now_s - horizon_s
    return [v for (t, v) in zip(times, vals) if t >= lo]


def aggregate_metrics(jsonl_path: Path | str, out_path: Path | str) -> Dict:
    """Aggregate p50/p99 overall and per 1m/5m/10m windows from a metrics JSONL.
    Expects records with keys t_* and an ISO ts if available; uses file read time otherwise.
    """
    p = Path(jsonl_path)
    if not p.exists():
        out = {"p50": {}, "p99": {}, "n": 0}
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
        return out
    now_s = time.time()
    series: Dict[str, Tuple[List[float], List[float]]] = {
        "t_feat_ms": ([], []),
        "t_nn_ms": ([], []),
        "t_policy_ms": ([], []),
        "t_total_ms": ([], []),
    }
    n = 0
    with open(p, "r") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts_s = now_s
            if "ts" in d:
                try:
                    # Accept ISO Z
                    from datetime import datetime

                    ts_s = datetime.fromisoformat(d["ts"].replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts_s = now_s
            for k in series:
                if k in d:
                    series[k][0].append(ts_s)
                    series[k][1].append(float(d[k]))
            n += 1
    def P(vals: List[float], q: float) -> float:
        return float(np.percentile(vals, q)) if vals else 0.0
    out = {"p50": {}, "p99": {}, "n": n, "windows": {}}
    # Overall
    for k, (_, v) in series.items():
        out["p50"][k] = P(v, 50)
        out["p99"][k] = P(v, 99)
    # Per window
    for name, horizon in {"1m": 60.0, "5m": 300.0, "10m": 600.0}.items():
        out["windows"][name] = {"p50": {}, "p99": {}}
        for k, (t_arr, v_arr) in series.items():
            v_win = _window_filter(t_arr, v_arr, horizon, now_s)
            out["windows"][name]["p50"][k] = P(v_win, 50)
            out["windows"][name]["p99"][k] = P(v_win, 99)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out
