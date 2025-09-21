from __future__ import annotations

import hashlib
import numpy as np

from app.runtime.throttle import TegrastatsWatcher


def test_throttle_parse_line_and_thresholds() -> None:
    w = TegrastatsWatcher(thresh_temp=70.0, thresh_gpu=80.0)
    # Simulate a tegrastats line (values vary by L4T; we accept any order)
    line = "RAM 2000/8000MB (lfb 512x4MB) SWAP 0/4096MB CPU@55C GPU@72C GPU@85%"
    h = w._parse_tegrastats_line(line)  # type: ignore[attr-defined]
    assert h is not None
    assert h.temp_c >= 70.0
    assert h.gpu_util >= 80.0
    assert h.throttle is True


def _decision_id(symbol: str, ts_close: str, win: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(symbol.encode()); h.update(b"|"); h.update(ts_close.encode()); h.update(b"|")
    h.update(win.tobytes())
    return h.hexdigest()[:16]


def test_decision_id_determinism() -> None:
    sym = "BTCUSDT"; ts = "2025-09-21T00:00:00Z"
    win = np.zeros((144, 8), dtype=np.float32)
    a = _decision_id(sym, ts, win)
    b = _decision_id(sym, ts, win.copy())
    assert a == b
