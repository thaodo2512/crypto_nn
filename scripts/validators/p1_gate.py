from __future__ import annotations

import json
import os
import sys
from glob import glob

import duckdb
import pandas as pd


def main() -> None:
    tf = os.getenv("TF", "5m")
    sym = os.getenv("SYMS", "BTCUSDT").split(",")[0]
    days = int(os.getenv("DAYS", "90"))
    glob_path = f"data/parquet/{tf}/{sym}/y=*/m=*/d=*/part-*.parquet"
    con = duckdb.connect()
    try:
        df = con.execute(f"select ts, symbol, * except(ts, symbol) from read_parquet('{glob_path}') order by ts").df()
    finally:
        con.close()
    if df.empty:
        print(json.dumps({"pass": False, "error": "P1:no parquet found"}))
        sys.exit(1)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # Build exact grid for last N days
    end = df["ts"].max()
    start = end - pd.Timedelta(days=days) + pd.Timedelta(minutes=5)
    grid = pd.date_range(start.floor(tf), end.ceil(tf), freq="5min", tz="UTC", inclusive="right")
    present = df[(df["ts"] >= grid.min()) & (df["ts"] <= grid.max())]["ts"].unique()
    gap_ratio = 1 - (len(present) / max(1, len(grid)))
    # NaN ratio, impute flags if present
    nan_any = df.isna().any().any()
    # acceptance
    ok = (gap_ratio <= 0.005) and (not nan_any)
    report = {"gap_ratio": float(gap_ratio), "nan_any": bool(nan_any), "pass": ok}
    print(json.dumps(report))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

