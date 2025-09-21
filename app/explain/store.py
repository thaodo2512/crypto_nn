from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional


class ExplainStore:
    def __init__(self, root: str = "explain", ttl_days: int = 30) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_s = int(ttl_days * 24 * 3600)

    def _path(self, decision_id: str) -> Path:
        return self.root / f"{decision_id}.json"

    def write_atomic(self, decision_id: str, payload: Dict) -> Path:
        dst = self._path(decision_id)
        tmp = dst.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, dst)
        return dst

    def read(self, decision_id: str) -> Optional[Dict]:
        p = self._path(decision_id)
        if not p.exists():
            return None
        st = p.stat()
        if (time.time() - st.st_mtime) > self.ttl_s:
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def gc(self) -> int:
        """Remove expired files; return count removed."""
        removed = 0
        now = time.time()
        for p in self.root.glob("*.json"):
            try:
                if (now - p.stat().st_mtime) > self.ttl_s:
                    p.unlink(missing_ok=True)
                    removed += 1
            except FileNotFoundError:
                pass
        return removed

