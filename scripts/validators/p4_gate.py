from __future__ import annotations

import json
import sys
from glob import glob
from pathlib import Path

import duckdb


def _from_counts_json() -> list[float]:
    counts_fp = Path("data/train/counts.json")
    if not counts_fp.exists():
        return []
    try:
        d = json.loads(counts_fp.read_text())
    except Exception:
        return []
    ratios: list[float] = []
    for _, c in d.items():
        pw = float(c.get("post_WAIT", 0))
        pl = float(c.get("post_LONG", 0))
        ps = float(c.get("post_SHORT", 0))
        tot = pw + pl + ps
        if tot > 0:
            ratios.append(pw / tot)
    return ratios


def _from_augmented_parquet() -> list[float]:
    con = duckdb.connect()
    try:
        train_paths = sorted(glob("data/aug/train_smote/**/train.parquet", recursive=True))
        if not train_paths:
            train_paths = sorted(glob("data/aug/train_smote/**/*.parquet", recursive=True))
        ratios = []
        for p in train_paths[:10]:
            # prefer column 'y', else 'label'
            cols = con.execute("select * from read_parquet(?) limit 0", [p]).columns
            col = "y" if "y" in cols else ("label" if "label" in cols else None)
            if not col:
                continue
            r = con.execute(f"select avg({col}='WAIT') from read_parquet(?)", [p]).fetchone()
            if r and r[0] is not None:
                ratios.append(float(r[0]))
        return ratios
    finally:
        con.close()


def main() -> None:
    # Prefer counts.json produced by SMOTE; fallback to scanning augmented parquet
    wait_ratios = _from_counts_json()
    if not wait_ratios:
        wait_ratios = _from_augmented_parquet()
    if not wait_ratios:
        print(json.dumps({"pass": False, "error": "P4:no augmented train data found"}))
        sys.exit(1)
    ok = all(x <= 0.60 + 1e-9 for x in wait_ratios)
    report = {"wait_ratios_max": max(wait_ratios), "count": len(wait_ratios), "pass": ok}
    print(json.dumps(report))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
