import numpy as np
from fastapi.testclient import TestClient

from service_p9 import create_app, ModelSession


def test_api_contract():
    model = ModelSession(onnx_path=None, window=12, input_dim=4, temperature=1.0)
    app = create_app(model, k_min=1.0, k_max=1.5, H=36, monitor=None)
    client = TestClient(app)
    win = np.random.randn(12, 4).tolist()
    r = client.post("/score", json={"window": win})
    assert r.status_code == 200
    data = r.json()
    assert set(["p_wait", "p_long", "p_short"]).issubset(data.keys())
    r2 = client.post("/decide", json={"window": win, "close": 100.0, "atr_pct": 0.01, "vol_pctile": 0.5})
    assert r2.status_code == 200
    data2 = r2.json()
    assert set(["side", "size", "TP_px", "SL_px", "EV", "reason"]).issubset(data2.keys())


def test_latency_targets():
    # Use dummy session to ensure fast inference
    model = ModelSession(onnx_path=None, window=12, input_dim=8, temperature=1.0)
    win = np.random.randn(12, 8).astype(np.float32)
    lat = []
    for _ in range(200):
        import time
        t0 = time.perf_counter()
        _ = model.predict_proba(win)
        lat.append((time.perf_counter() - t0) * 1000)
    import numpy as np
    p50 = np.percentile(lat, 50)
    p99 = np.percentile(lat, 99)
    assert p50 < 500.0 and p99 < 2000.0


def test_throttle_rules():
    class HotMonitor:
        def read(self):
            from types import SimpleNamespace
            return SimpleNamespace(temp_c=75.0, gpu_util=90.0)

    model = ModelSession(onnx_path=None, window=12, input_dim=4, temperature=1.0)
    app = create_app(model, k_min=1.0, k_max=1.5, H=36, monitor=HotMonitor())
    client = TestClient(app)
    win = np.random.randn(12, 4).tolist()
    r = client.post("/decide", json={"window": win, "close": 100.0, "atr_pct": 0.01, "vol_pctile": 0.5})
    assert r.status_code == 200
    assert r.json().get("reason") in ("throttle", "ev<=0")

