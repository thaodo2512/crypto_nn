import json
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from explain_p10 import integrated_gradients, run as explain_run, make_api
from cli_p5 import GRUClassifier
from fastapi.testclient import TestClient


def test_ig_shape():
    W, F = 12, 4
    model = GRUClassifier(input_dim=F, hidden=8)
    win = np.random.randn(W, F).astype(np.float32)
    attr = integrated_gradients(model, win, target=1, steps=8)
    assert attr.shape == (W, F)
    # sparsify top-3 manually
    k = 3
    idx = np.argsort(np.abs(attr).ravel())[::-1][:k]
    assert len(idx) == k


def test_retrieval_window(tmp_path):
    # Write explain JSON
    doc = {"id": "abc123", "ts_unix": 0, "window": [[0,1],[2,3]], "features": ["f0","f1"], "topk": [], "summary": {}}
    expdir = tmp_path / "explain"
    expdir.mkdir(parents=True, exist_ok=True)
    (expdir / "abc123.json").write_text(json.dumps(doc))
    # in-range retrieval
    app = make_api(expdir, ttl_days=30)
    client = TestClient(app)
    r = client.get("/explain", params={"id": "abc123"})
    assert r.status_code == 200

