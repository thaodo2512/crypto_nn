import json
import time
from fastapi.testclient import TestClient

from app.explain.store import ExplainStore
from app.explain.api import make_router
from fastapi import FastAPI


def test_ttl_and_gc(tmp_path):
    store = ExplainStore(str(tmp_path), ttl_days=0)  # immediate expiry
    doc = {"id": "x", "ts_unix": int(time.time()), "window_shape": [144, 3], "features": ["f0","f1","f2"], "topk": [], "summary": {"sum":0,"l1":0,"l2":0,"max_abs":0}}
    store.write_atomic("x", doc)
    # API should 404 due to TTL=0
    api = FastAPI(); api.include_router(make_router(store))
    c = TestClient(api)
    r = c.get("/explain", params={"id":"x"})
    assert r.status_code == 404
    # GC should remove the file
    removed = store.gc()
    assert removed >= 1

