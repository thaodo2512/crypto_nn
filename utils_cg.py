import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError


ISO8601 = "%Y-%m-%dT%H:%M:%S%z"


def to_utc_ts(x: Any) -> pd.Timestamp:
    """Convert epoch seconds/ms or ISO string to tz-aware UTC Timestamp."""
    if x is None:
        raise ValueError("timestamp value is None")
    if isinstance(x, (int, float)):
        # Heuristic: ms if >= 10^12
        if float(x) >= 1e12:
            ts = pd.to_datetime(int(x), unit="ms", utc=True)
        else:
            ts = pd.to_datetime(int(x), unit="s", utc=True)
        return ts
    if isinstance(x, str):
        try:
            return pd.to_datetime(x, utc=True)
        except Exception as e:
            raise ValueError(f"Unrecognized timestamp string: {x}") from e
    if isinstance(x, datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=timezone.utc)
        return x.astimezone(timezone.utc)
    raise TypeError(f"Unsupported timestamp type: {type(x)}")


class CGClient:
    """CoinGlass HTTP client with retry/backoff and auth header.

    Expects API key in env var COINGLASS_API_KEY.
    Base URL and timeouts can be customized.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_s
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        api_key = os.getenv("COINGLASS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing COINGLASS_API_KEY environment variable for authentication"
            )
        self._headers = {"CG-API-KEY": api_key}
        self._client = httpx.Client(timeout=self.timeout, headers=self._headers)

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET request with basic retry/backoff for 429/5xx."""
        url = self._url(path)
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.get(url, params=params or {})
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except json.JSONDecodeError as e:
                        raise RuntimeError(f"Invalid JSON from {url}") from e
                if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                    # backoff
                    delay = self.backoff_base * (2**attempt)
                    time.sleep(delay)
                    continue
                # Other error: raise now
                raise RuntimeError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
                delay = self.backoff_base * (2**attempt)
                time.sleep(delay)
                continue
        raise RuntimeError(f"Request failed after retries: {url}") from last_exc

    def get_paginated(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        page_param: str = "page",
        page_size_param: str = "page_size",
        page_size: int = 500,
        data_key_candidates: Tuple[str, ...] = ("data", "list", "rows", "result"),
        max_pages: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch all pages for endpoints that support pagination.

        Attempts to discover data array under common keys.
        """
        page = 1
        out: List[Dict[str, Any]] = []
        while page <= max_pages:
            q = dict(params or {})
            q[page_param] = page
            q[page_size_param] = page_size
            payload = self.get_json(path, q)
            data = None
            for k in data_key_candidates:
                if isinstance(payload, dict) and k in payload and isinstance(payload[k], list):
                    data = payload[k]
                    break
            if data is None:
                # Try nested data
                maybe = payload
                for k in ("data", "result"):
                    if isinstance(maybe, dict) and k in maybe:
                        maybe = maybe[k]
                if isinstance(maybe, list):
                    data = maybe
            if not data:
                break
            out.extend(data)
            if len(data) < page_size:
                break
            page += 1
        return out


# --- Pydantic models for common bar/event types ---


class PriceBar(BaseModel):
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume_usd: float = Field(default=0.0)

    @classmethod
    def from_raw(cls, row: Dict[str, Any]) -> "PriceBar":
        # Accept various common keys
        ts = row.get("t") or row.get("ts") or row.get("timestamp")
        o = row.get("o") or row.get("open")
        h = row.get("h") or row.get("high")
        l = row.get("l") or row.get("low")
        c = row.get("c") or row.get("close")
        v = row.get("v") or row.get("volume") or row.get("volume_usd") or 0.0
        return cls(ts=to_utc_ts(ts), open=float(o), high=float(h), low=float(l), close=float(c), volume_usd=float(v))


class OIBar(BaseModel):
    ts: pd.Timestamp
    oi_now: float

    @classmethod
    def from_raw(cls, row: Dict[str, Any]) -> "OIBar":
        ts = row.get("t") or row.get("ts") or row.get("timestamp")
        # Use close as per requirement
        val = row.get("c") or row.get("close") or row.get("oi")
        return cls(ts=to_utc_ts(ts), oi_now=float(val))


class FundingBar(BaseModel):
    ts: pd.Timestamp
    funding_now: float

    @classmethod
    def from_raw(cls, row: Dict[str, Any]) -> "FundingBar":
        ts = row.get("t") or row.get("ts") or row.get("timestamp")
        # Funding can be rate per interval
        val = row.get("f") or row.get("funding") or row.get("fundingRate") or row.get("value")
        return cls(ts=to_utc_ts(ts), funding_now=float(val))


class TakerVolumeBar(BaseModel):
    ts: pd.Timestamp
    taker_buy_usd: float = 0.0
    taker_sell_usd: float = 0.0

    @classmethod
    def from_raw(cls, row: Dict[str, Any]) -> "TakerVolumeBar":
        ts = row.get("t") or row.get("ts") or row.get("timestamp")
        buy = row.get("buy") or row.get("taker_buy_usd") or row.get("buyVolumeUsd") or 0.0
        sell = row.get("sell") or row.get("taker_sell_usd") or row.get("sellVolumeUsd") or 0.0
        return cls(ts=to_utc_ts(ts), taker_buy_usd=float(buy), taker_sell_usd=float(sell))


class LiquidationEvent(BaseModel):
    ts: pd.Timestamp
    notional_usd: float

    @classmethod
    def from_raw(cls, row: Dict[str, Any]) -> "LiquidationEvent":
        ts = row.get("t") or row.get("ts") or row.get("timestamp")
        notional = row.get("notional") or row.get("value") or row.get("amountUsd")
        return cls(ts=to_utc_ts(ts), notional_usd=float(notional))


# --- Transformation helpers ---


def resample_15m_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Resample OHLCV to 15m with right-closed, right label."""
    if df.empty:
        return df
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume_usd": "sum",
    }
    out = (
        df.resample("15min", closed="right", label="right")
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"], how="any")
    )
    out = out.reset_index()
    return out


def resample_15m_sum(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Resample selected columns by summation to 15m."""
    if df.empty:
        return pd.DataFrame(columns=["ts", *cols])
    tmp = df.copy()
    tmp["ts"] = pd.to_datetime(tmp["ts"], utc=True)
    tmp = tmp.set_index("ts").sort_index()
    out = tmp[cols].resample("15min", closed="right", label="right").sum()
    return out.reset_index()


def reindex_15m_ffill_limit(df: pd.DataFrame, col: str, limit: int = 3) -> pd.DataFrame:
    """Reindex to 15m and forward-fill with a max bar limit. Adds boolean `<col>_imputed`."""
    if df.empty:
        return pd.DataFrame(columns=["ts", col, f"{col}_imputed"])  # typed columns
    tmp = df.copy()
    tmp["ts"] = pd.to_datetime(tmp["ts"], utc=True)
    tmp = tmp.set_index("ts").sort_index()
    full = tmp[[col]].resample("15min", closed="right", label="right").asfreq()
    ffilled = full.ffill(limit=limit)
    imputed = ffilled[col].notna() & full[col].isna()
    out = ffilled.reset_index()
    out[f"{col}_imputed"] = imputed.reset_index(drop=True)
    return out


def rolling_percentile_30d(series: pd.Series) -> pd.Series:
    """Compute percentile of current value within a 30D rolling window.

    Returns value in [0,1]. Uses a simple rank-based estimator.
    """
    s = series.copy()
    s.index = pd.to_datetime(s.index, utc=True)
    return s.rolling("30D", min_periods=10).apply(lambda x: (x <= x.iloc[-1]).mean(), raw=False)


def ensure_unique_key(df: pd.DataFrame, key: List[str]) -> None:
    """Raise ValueError if duplicates in key."""
    dup = df.duplicated(subset=key)
    if dup.any():
        dups = df.loc[dup, key]
        raise ValueError(f"Duplicate keys detected for {len(dups)} rows, first rows: {dups.head(5).to_dict(orient='records')}")


def add_partitions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d["y"] = d["ts"].dt.year.astype("int16")
    d["m"] = d["ts"].dt.month.astype("int8")
    d["d"] = d["ts"].dt.day.astype("int8")
    return d


def write_parquet_partitioned(df: pd.DataFrame, out_dir: str) -> None:
    """Write Parquet partitioned by y/m/d using ZSTD level 3."""
    import pyarrow as pa
    import pyarrow.dataset as ds

    if df.empty:
        return
    table = pa.Table.from_pandas(df, preserve_index=False)
    fmt = ds.ParquetFileFormat()
    file_opts = fmt.make_write_options(compression="zstd", compression_level=3)
    ds.write_dataset(
        table,
        base_dir=out_dir,
        format=fmt,
        partitioning=["y", "m", "d"],
        file_options=file_opts,
        existing_data_behavior="overwrite_or_ignore",
    )

