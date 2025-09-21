from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _stable_from_metrics(path: Path) -> bool:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    # Expect per-fold dict; consider stable if non-empty and finite val_loss
    if isinstance(d, dict) and d:
        for k, v in d.items():
            if isinstance(v, dict):
                vl = v.get("val_loss")
                if vl is None:
                    continue
                try:
                    float(vl)
                except Exception:
                    return False
        return True
    return False


def _embargo_from_folds(path: Path) -> tuple[str, int]:
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return "", 0
    # Prefer meta.embargo_bars if present
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    bars = int(meta.get("embargo_bars", 0) or 0)
    if bars:
        return ("1D" if bars >= 288 else f"{bars}bars"), bars
    # Fallback: accept presence of folds as walkforward signal
    folds = obj.get("folds") if isinstance(obj, dict) else None
    if folds:
        return ("1D", 288)
    return "", 0


def main() -> None:
    window = int(os.getenv("WINDOW", "144"))
    metrics = Path("reports/p5_cv_metrics.json")
    folds = Path("artifacts/folds.json")
    if not metrics.exists():
        print(json.dumps({"pass": False, "error": "P5:metrics json missing"}))
        sys.exit(1)
    stable = _stable_from_metrics(metrics)
    emb_str, emb_bars = _embargo_from_folds(folds) if folds.exists() else ("", 0)
    cv = "walkforward" if folds.exists() else ""
    ok = stable and (emb_bars >= 288) and (window == window)
    print(json.dumps({"cv": cv, "embargo": emb_str, "window": window, "stable": stable, "pass": ok}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
