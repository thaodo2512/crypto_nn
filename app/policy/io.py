from __future__ import annotations

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_COLS = [
    "ts",
    "symbol",
    "side",
    "size",
    "TP_px",
    "SL_px",
    "EV_bps",
    "k_atr",
    "vol_pctile",
    "liq60_pctile",
    "reason",
    "tau_long",
    "tau_short",
]


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required decision columns: {missing}")
    # Enforce dtypes lightly
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str)
    for c in ["size", "TP_px", "SL_px", "EV_bps", "k_atr", "vol_pctile", "liq60_pctile", "tau_long", "tau_short"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["reason"] = df["reason"].astype(str)
    return df


def write_decisions(df: pd.DataFrame, out_root: str = "artifacts/decisions") -> None:
    """Write decisions partitioned by date=YYYY-MM-DD under out_root.

    One file per date partition, with ZSTD compression.
    """
    df = _ensure_schema(df)
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    # Partition by date
    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    for date, g in df.groupby("date", sort=True):
        part_dir = root / f"date={date}"
        part_dir.mkdir(parents=True, exist_ok=True)
        # Create deterministic filename
        fname = f"part-{date.replace('-', '')}.parquet"
        table = pa.Table.from_pandas(g.drop(columns=["date"]).reset_index(drop=True), preserve_index=False)
        pq.write_table(
            table,
            part_dir / fname,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
        )

