#!/usr/bin/env python3
"""
P5 calibration artifacts gate

Checks that Phase-5 exporter wrote the right artifacts for calibration:
- Per-fold parquet files exist under artifacts/p5_oos_probs/<sym>/fold*.parquet
- Columns include: ts, symbol, y, fold_id, split in {val,oos}
- Probabilities present (p_wait,p_long,p_short) and sum≈1.0
- Logits present (logits_0..2) with finite values (no NaN/inf)
- Both VAL and OOS splits have >0 rows

Writes: reports/p5_calib_artifacts.json and exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import json
import math
import os
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _detect_symbol() -> str:
    # Prefer explicit envs
    sym = os.environ.get("SYM")
    if sym:
        return sym
    syms = os.environ.get("SYMS")
    if syms:
        return syms.split(",")[0].strip()
    # Derive from directory names
    base = Path("artifacts/p5_oos_probs")
    if base.exists():
        subs = [p.name for p in base.iterdir() if p.is_dir()]
        if subs:
            return subs[0]
    # Fallback default
    return "BTCUSDT"


def main() -> None:
    sym = _detect_symbol()
    root_dir_sym = Path(f"artifacts/p5_oos_probs/{sym}")
    root_dir = root_dir_sym if root_dir_sym.exists() else Path("artifacts/p5_oos_probs")
    files = sorted(glob(str(root_dir / "fold*.parquet")))
    report: Dict[str, object] = {"symbol": sym, "root": str(root_dir), "files": files}
    violations: List[str] = []

    if not files:
        violations.append("no_oos_files_found")
        _finish(report, violations)
        return

    total_rows = 0
    val_rows = 0
    oos_rows = 0
    folds_seen = set()

    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            violations.append(f"read_error:{os.path.basename(f)}:{e}")
            continue
        total_rows += len(df)
        if "split" in df.columns:
            val_rows += int((df["split"] == "val").sum())
            oos_rows += int((df["split"] == "oos").sum())
        if "fold_id" in df.columns:
            for v in pd.unique(df["fold_id"]):
                try:
                    folds_seen.add(int(v))
                except Exception:
                    pass

        # Required columns
        req = {"ts", "symbol", "y", "fold_id", "split"}
        missing = req - set(df.columns)
        if missing:
            violations.append(f"missing_columns:{os.path.basename(f)}:{sorted(missing)}")

        # Probability columns
        has_named_p = {"p_wait", "p_long", "p_short"}.issubset(set(df.columns))
        has_generic_p = any(c.startswith("p_") for c in df.columns)
        if not (has_named_p or has_generic_p):
            violations.append(f"missing_probs:{os.path.basename(f)}")
        else:
            if has_named_p:
                P = df[["p_wait", "p_long", "p_short"]].to_numpy()
            else:
                pcols = [c for c in df.columns if c.startswith("p_")][:3]
                P = df[pcols].to_numpy()
            if np.isnan(P).any() or np.isinf(P).any():
                violations.append(f"nan_inf_probs:{os.path.basename(f)}")
            sums = P.sum(axis=1)
            off = np.abs(sums - 1.0) > 1e-3
            if off.any():
                violations.append(f"prob_sum_mismatch:{os.path.basename(f)}:{int(off.sum())}")

        # Logits
        logit_cols = [c for c in df.columns if c.startswith("logits_")]
        if len(logit_cols) < 3:
            violations.append(f"missing_logits:{os.path.basename(f)}")
        else:
            L = df[logit_cols[:3]].to_numpy()
            if np.isnan(L).any() or np.isinf(L).any():
                violations.append(f"nan_inf_logits:{os.path.basename(f)}")

    report.update({
        "folds": sorted(folds_seen),
        "total_rows": total_rows,
        "val_rows": val_rows,
        "oos_rows": oos_rows,
    })

    # Split coverage checks
    if val_rows <= 0:
        violations.append("no_val_rows_for_calibration")
    if oos_rows <= 0:
        violations.append("no_oos_rows_for_eval")

    _finish(report, violations)


def _finish(report: Dict[str, object], violations: List[str]) -> None:
    report["pass"] = len(violations) == 0
    report["violations"] = violations
    Path("reports").mkdir(parents=True, exist_ok=True)
    out = Path("reports/p5_calib_artifacts.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

