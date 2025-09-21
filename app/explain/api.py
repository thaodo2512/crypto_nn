from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .store import ExplainStore


def make_router(store: ExplainStore) -> APIRouter:
    r = APIRouter()

    @r.get("/health")
    def health():
        return {"ok": True}

    @r.head("/explain")
    def head_explain(id: str):
        doc = store.read(id)
        if doc is None:
            raise HTTPException(status_code=404, detail="not found")
        return JSONResponse({}, status_code=200)

    @r.get("/explain")
    def get_explain(id: str):
        doc = store.read(id)
        if doc is None:
            raise HTTPException(status_code=404, detail="not found or expired")
        return JSONResponse(doc)

    return r

