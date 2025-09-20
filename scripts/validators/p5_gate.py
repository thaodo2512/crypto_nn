from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    window = int(os.getenv("WINDOW", "144"))
    metrics = Path("reports/p5_cv_metrics.json")
    if not metrics.exists():
        print(json.dumps({"pass": False, "error": "P5:metrics json missing"}))
        sys.exit(1)
    d = json.loads(metrics.read_text())
    # Basic checks
    cv = d.get("cv", "")
    emb = str(d.get("embargo", ""))
    win = int(d.get("window", window))
    stable = bool(d.get("stable", True))
    ok = (cv.lower().startswith("walkforward")) and (emb in ("1D", "1d", "86400s")) and (win == window) and stable
    print(json.dumps({"cv": cv, "embargo": emb, "window": win, "stable": stable, "pass": ok}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

