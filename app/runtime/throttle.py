from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Health:
    temp_c: float
    gpu_util: float
    throttle: bool


class TegrastatsWatcher:
    """Background watcher for Jetson tegrastats output.

    - Spawns `/usr/bin/tegrastats --interval 1000`
    - Parses GPU temperature (°C) and GPU utilization (%)
    - Writes transitions to ops/tegrastats.csv
    - Exposes latest atomic state
    """

    def __init__(self, thresh_temp: float = 70.0, thresh_gpu: float = 80.0) -> None:
        self.thresh_temp = float(thresh_temp)
        self.thresh_gpu = float(thresh_gpu)
        self.temp_c: float = 50.0
        self.gpu_util: float = 0.0
        self.throttle: bool = False
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._csv_path = Path("ops/tegrastats.csv")
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["ts", "temp", "gpu_util", "throttle"])

    def _parse_tegrastats_line(self, line: str) -> Optional[Health]:
        # Best-effort parser; examples vary by L4T version
        try:
            txt = line.strip().replace("%", "").replace("@", " ")
            toks = txt.split()
            temp = self.temp_c
            util = self.gpu_util
            for i, tok in enumerate(toks):
                if tok.endswith("C") and tok[:-1].isdigit():
                    temp = float(tok[:-1])
                elif tok.isdigit():
                    v = float(tok)
                    if 0 <= v <= 100:
                        util = v
                        # keep scanning to capture temp if appears later
            thr = (temp >= self.thresh_temp) or (util >= self.thresh_gpu)
            return Health(temp, util, thr)
        except Exception:
            return None

    def _run(self) -> None:
        # Prefer tegrastats; if unavailable, synthesize idle values
        cmd = None
        for cand in ("/usr/bin/tegrastats", "tegrastats"):
            if shutil := __import__("shutil"):
                if shutil.which(cand):
                    cmd = [cand, "--interval", "1000"]
                    break
        if cmd is None:
            # fallback synthetic loop
            while not self._stop.is_set():
                self._update_and_maybe_log(Health(self.temp_c, self.gpu_util, False))
                time.sleep(1.0)
            return
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            if self._stop.is_set():
                break
            h = self._parse_tegrastats_line(line)
            if h:
                self._update_and_maybe_log(h)
        try:
            proc.terminate()
        except Exception:
            pass

    def _update_and_maybe_log(self, h: Health) -> None:
        prev = self.throttle
        self.temp_c, self.gpu_util, self.throttle = h.temp_c, h.gpu_util, h.throttle
        if prev != self.throttle:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(self._csv_path, "a", newline="") as f:
                csv.writer(f).writerow([ts, f"{self.temp_c:.1f}", f"{self.gpu_util:.1f}", int(self.throttle)])

    def start(self) -> None:
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, name="tegrastats", daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=1.0)

    def read(self) -> Health:
        return Health(self.temp_c, self.gpu_util, self.throttle)


class Throttler:
    """Compatibility shim with read() and should_throttle()."""

    def __init__(self, thresh_temp: float = 70.0, thresh_gpu: float = 80.0) -> None:
        self.watcher = TegrastatsWatcher(thresh_temp=thresh_temp, thresh_gpu=thresh_gpu)
        self.watcher.start()

    def read(self) -> Health:
        return self.watcher.read()

    def should_throttle(self, temp_c: float, gpu_util: float) -> bool:
        return (temp_c >= self.watcher.thresh_temp) or (gpu_util >= self.watcher.thresh_gpu)


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
