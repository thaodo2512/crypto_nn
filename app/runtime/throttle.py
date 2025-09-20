from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class Health:
    temp_c: float
    gpu_util: float


class Throttler:
    def __init__(self, thresh_temp: float = 70.0, thresh_gpu: float = 80.0) -> None:
        self.thresh_temp = thresh_temp
        self.thresh_gpu = thresh_gpu

    def read(self) -> Health:
        try:
            out = subprocess.check_output(["tegrastats", "--interval", "1000", "--count", "1"], timeout=2).decode()
            temp = 50.0
            gpu = 30.0
            # crude parse
            toks = out.replace("%", "").replace("@", " ").split()
            for i, tok in enumerate(toks):
                if tok.endswith("C") and tok[:-1].isdigit():
                    temp = float(tok[:-1])
                if tok.isdigit():
                    v = float(tok)
                    if 0 <= v <= 100:
                        gpu = v
                        break
            return Health(temp, gpu)
        except Exception:
            return Health(50.0, 30.0)

    def should_throttle(self, temp_c: float, gpu_util: float) -> bool:
        return (temp_c >= self.thresh_temp) or (gpu_util >= self.thresh_gpu)


def load_tau(path: str) -> Dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {"tau_long": 0.55, "tau_short": 0.55}
    try:
        d = json.loads(p.read_text())
        return {
            "tau_long": float(d.get("tau_long", 0.55)),
            "tau_short": float(d.get("tau_short", 0.55)),
        }
    except Exception:
        return {"tau_long": 0.55, "tau_short": 0.55}

