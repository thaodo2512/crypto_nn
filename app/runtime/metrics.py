from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


class MetricsSink:
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


def aggregate_metrics(jsonl_path: Path | str, out_path: Path | str) -> Dict:
    p = Path(jsonl_path)
    if not p.exists():
        out = {"p50": {}, "p99": {}, "n": 0}
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
        return out
    vals = {"t_feat_ms": [], "t_nn_ms": [], "t_policy_ms": [], "t_total_ms": []}
    n = 0
    with open(p, "r") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            for k in vals:
                if k in d:
                    vals[k].append(float(d[k]))
            n += 1
    p50 = {k: (float(np.percentile(v, 50)) if v else 0.0) for k, v in vals.items()}
    p99 = {k: (float(np.percentile(v, 99)) if v else 0.0) for k, v in vals.items()}
    out = {"p50": p50, "p99": p99, "n": n}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out

