from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    summary = Path("reports/p6_oos_summary.json")
    calib = Path("models/calib.json")
    if not summary.exists() or not calib.exists():
        print(json.dumps({"pass": False, "error": "P6:missing summary or calib"}))
        sys.exit(1)
    s = json.loads(summary.read_text())
    ece = float(s.get("ece", 0.0)) if isinstance(s, dict) else 0.0
    ev = float(s.get("ev_per_trade", 0.0)) if isinstance(s, dict) else 0.0
    ok = (ece <= 0.10 + 1e-9) and (ev > 0.0)
    print(json.dumps({"ece": ece, "ev_per_trade": ev, "pass": ok}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

