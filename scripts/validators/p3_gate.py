from __future__ import annotations

import json
import os
import sys

import duckdb


def main() -> None:
    tf = os.getenv("TF", "5m")
    sym = os.getenv("SYMS", "BTCUSDT").split(",")[0]
    feat_glob = f"data/features/{tf}/{sym}/y=*/m=*/d=*/part-*.parquet"
    lab_glob = f"data/labels/{tf}/{sym}/y=*/m=*/d=*/part-*.parquet"
    con = duckdb.connect()
    try:
        lab_cnt = con.execute(f"select count(*), min(ts), max(ts) from read_parquet('{lab_glob}')").fetchone()
        join_cnt = con.execute(
            "select count(*) from read_parquet(?) l inner join read_parquet(?) f using(ts, symbol)",
            [lab_glob, feat_glob],
        ).fetchone()
    finally:
        con.close()
    ok = (lab_cnt[0] > 0) and (join_cnt[0] == lab_cnt[0])
    report = {"labels": int(lab_cnt[0]), "join_count": int(join_cnt[0]), "pass": ok}
    print(json.dumps(report))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

