from __future__ import annotations

import json
import os
import sys
from glob import glob

import duckdb


def main() -> None:
    tf = os.getenv("TF", "5m")
    sym = os.getenv("SYMS", "BTCUSDT").split(",")[0]
    glob_path = f"data/features/{tf}/{sym}/y=*/m=*/d=*/part-*.parquet"
    con = duckdb.connect()
    try:
        df = con.execute(f"select * from read_parquet('{glob_path}')").df()
    finally:
        con.close()
    if df.empty:
        print(json.dumps({"pass": False, "error": "P2:no features parquet found"}))
        sys.exit(1)
    # Count feature columns (exclude keys, flags allowed)
    # Count only numeric feature columns, excluding keys and internal flags (_-prefixed)
    cols = [c for c in df.columns if c not in ("ts", "symbol") and not str(c).startswith("_")]
    n = len(cols)
    # NaN
    nan_any = df[cols].isna().any().any()
    ok = (10 <= n <= 20) and (not nan_any)
    report = {"n_features": n, "nan_any": bool(nan_any), "pass": ok}
    print(json.dumps(report))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
