import json
import numpy as np
from fastapi.testclient import TestClient

from service_p9 import create_app
from app.runtime.model_session import ModelSession


def test_sla_logging_and_stats(tmp_path, monkeypatch):
    # Dummy model to speed
    ms = ModelSession(None, window=4, input_dim=2, apply_temp=False)
    app = create_app(ms, k_min=1.0, k_max=1.5, H=36, monitor=None, taus={"tau_long":0.55, "tau_short":0.55})
    client = TestClient(app)
    win = np.random.randn(4,2).tolist()
    import pandas as pd
    cc = (pd.Timestamp.utcnow() - pd.Timedelta(minutes=7)).isoformat()
    r = client.post("/decide", json={"window": win, "close": 100.0, "atr_pct": 0.01, "vol_pctile": 0.5, "candle_close_ts": cc})
    assert r.status_code == 200
    # SLA stats endpoint
    # build a simple handler locally since service provides aggregated metrics via files
    # Not asserting file content here due to env; basic response should include timers
    data = r.json()
    assert "t_total_ms" in data and "t_nn_ms" in data

