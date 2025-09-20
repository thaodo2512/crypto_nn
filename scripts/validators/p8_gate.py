from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    rpt = Path("reports/p8_parity.json")
    if not rpt.exists():
        print(json.dumps({"pass": False, "error": "P8:parity report missing"}))
        sys.exit(1)
    d = json.loads(rpt.read_text())
    mse = float(d.get("mse_probs", d.get("mse", 1.0)))
    sha = d.get("sha256", {})
    ok = (mse < 1e-3) and bool(sha.get("onnx"))
    print(json.dumps({"mse_probs": mse, "sha_on": bool(sha.get("onnx")), "pass": ok}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

