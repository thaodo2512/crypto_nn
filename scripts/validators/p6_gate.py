from __future__ import annotations

import json
import sys
from pathlib import Path
import os

import duckdb


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _count_oos_rows() -> int:
    try:
        con = duckdb.connect()
        n = con.execute("select count(*) from read_parquet('artifacts/p5_oos_probs/fold*.parquet')").fetchone()[0]
        con.close()
        return int(n)
    except Exception:
        return 0


def main() -> None:
    # Prefer symbol-suffixed artifacts if present; otherwise use generic
    sym = (os.environ.get("SYM") or (os.environ.get("SYMS") or "BTCUSDT").split(",")[0]).strip()
    summary_candidates = [
        Path(f"reports/p6_oos_summary_{sym}.json"),
        Path("reports/p6_oos_summary.json"),
    ]
    calib_candidates = [
        Path(f"models/calib_{sym}.json"),
        Path("models/calib.json"),
    ]
    ens_candidates = [
        Path(f"models/ensemble_5m_{sym}.json"),
        Path("models/ensemble_5m.json"),
    ]
    summary_p = next((p for p in summary_candidates if p.exists()), summary_candidates[-1])
    calib_p = next((p for p in calib_candidates if p.exists()), calib_candidates[-1])
    ens_p = next((p for p in ens_candidates if p.exists()), ens_candidates[-1])
    if not summary_p.exists() or not calib_p.exists():
        print(json.dumps({"pass": False, "error": "P6:missing summary or calib", "have_summary": summary_p.exists(), "have_calib": calib_p.exists()}))
        sys.exit(1)

    s = _load_json(summary_p)
    c = _load_json(calib_p)
    e = _load_json(ens_p) if ens_p.exists() else {}

    ece = float(s.get("ece", 0.0)) if isinstance(s, dict) else 0.0
    # Support either key name
    ev = float(s.get("ev_per_trade", s.get("ev_trade", 0.0))) if isinstance(s, dict) else 0.0
    tau = float(s.get("best_tau", s.get("tau", 0.0))) if isinstance(s, dict) else 0.0
    n_oos = _count_oos_rows()

    violations = []
    if ece > 0.10 + 1e-9:
        violations.append("ece>0.10")
    if ev <= 0.0:
        violations.append("ev_per_trade<=0")

    out = {
        "ece": ece,
        "ev_per_trade": ev,
        "best_tau": tau,
        "n_oos_rows": n_oos,
        "calibration_keys": list(c.keys()),
        "ensemble": {"weights": e.get("weights", {}), "temperature": e.get("temperature")},
        "pass": len(violations) == 0,
        "violations": violations,
    }
    print(json.dumps(out))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
