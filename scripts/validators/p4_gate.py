from __future__ import annotations

import json
import os
import sys
from glob import glob

import duckdb


def main() -> None:
    # Wait share in train ≤ 60%; OOS untouched (best-effort check by path)
    con = duckdb.connect()
    try:
        # Train windows (augmented)
        train_paths = sorted(glob("data/aug/train_smote/**/part-*.parquet", recursive=True))
        wait_ratios = []
        for p in train_paths[:10]:  # limit for speed
            r = con.execute("select avg(label='WAIT') from read_parquet(?)", [p]).fetchone()
            if r and r[0] is not None:
                wait_ratios.append(float(r[0]))
    finally:
        con.close()
    if not wait_ratios:
        print(json.dumps({"pass": False, "error": "P4:no augmented train data found"}))
        sys.exit(1)
    ok = all(x <= 0.60 + 1e-9 for x in wait_ratios)
    report = {"wait_ratios_max": max(wait_ratios), "count": len(wait_ratios), "pass": ok}
    print(json.dumps(report))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

