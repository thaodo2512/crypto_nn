import json
import hashlib
from pathlib import Path
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from collections import deque
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


def to_utc_dt(x: Any) -> datetime:
    """Convert various time formats to tz-aware Python datetime (UTC)."""
    return to_utc_ts(x).to_pydatetime()


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
        rpm_limit: Optional[int] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_s
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        # Rate limiting (requests per 60s window). Default from env or param.
        env_limit = os.getenv("CG_RPM_LIMIT")
        self.rpm_limit: Optional[int] = rpm_limit if rpm_limit is not None else (int(env_limit) if env_limit else 300)
        self._req_times: deque[float] = deque()
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

    def _throttle(self) -> None:
        """Block to respect rpm_limit in a sliding 60s window."""
        if not self.rpm_limit:
            return
        now = time.monotonic()
        # Drop requests older than 60s
        while self._req_times and now - self._req_times[0] > 60.0:
            self._req_times.popleft()
        # If at capacity, sleep until we can proceed
        while len(self._req_times) >= self.rpm_limit:
            head = self._req_times[0]
            sleep_s = max(0.0, 60.0 - (now - head))
            if sleep_s > 0:
                time.sleep(min(sleep_s, 1.0))  # sleep in small chunks
            now = time.monotonic()
            while self._req_times and now - self._req_times[0] > 60.0:
                self._req_times.popleft()

    def _mark_request(self) -> None:
        if self.rpm_limit:
            self._req_times.append(time.monotonic())

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET request with basic retry/backoff for 429/5xx."""
        url = self._url(path)
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                self._throttle()
                resp = self._client.get(url, params=params or {})
                self._mark_request()
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except json.JSONDecodeError as e:
                        raise RuntimeError(f"Invalid JSON from {url}") from e
                if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                    # backoff
                    # Honor Retry-After if provided
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = self.backoff_base * (2**attempt)
                    else:
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

    @staticmethod
    def _extract_records(payload: Any) -> List[Dict[str, Any]]:
        """Heuristically extract a list of dict records from arbitrary JSON payloads.

        Prefers lists whose items contain typical time/ohlc keys.
        """
        def score_list(lst: Any) -> int:
            if not isinstance(lst, list) or not lst:
                return -1
            if not isinstance(lst[0], dict):
                return -1
            keys = set(lst[0].keys())
            hits = len(keys & {"t", "ts", "timestamp", "o", "open", "c", "close"})
            return hits

        best: Tuple[int, Optional[List[Dict[str, Any]]]] = (-1, None)

        def walk(obj: Any) -> None:
            nonlocal best
            if isinstance(obj, list):
                s = score_list(obj)
                if s > best[0]:
                    best = (s, obj)  # type: ignore
            elif isinstance(obj, dict):
                for v in obj.values():
                    walk(v)

        walk(payload)
        return best[1] or []

    def get_list(self, path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        payload = self.get_json(path, params)
        return self._extract_records(payload)

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
    ts: datetime
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
        return cls(ts=to_utc_dt(ts), open=float(o), high=float(h), low=float(l), close=float(c), volume_usd=float(v))


class OIBar(BaseModel):
    ts: datetime
    oi_now: float

    @classmethod
    def from_raw(cls, row: Dict[str, Any]) -> "OIBar":
        ts = row.get("t") or row.get("ts") or row.get("timestamp")
        # Use close as per requirement
        val = row.get("c") or row.get("close") or row.get("oi")
        return cls(ts=to_utc_dt(ts), oi_now=float(val))


class FundingBar(BaseModel):
    ts: datetime
    funding_now: float

    @classmethod
    def from_raw(cls, row: Dict[str, Any]) -> "FundingBar":
        ts = row.get("t") or row.get("ts") or row.get("timestamp")
        # Funding can be rate per interval
        val = row.get("f") or row.get("funding") or row.get("fundingRate") or row.get("value")
        return cls(ts=to_utc_dt(ts), funding_now=float(val))


class TakerVolumeBar(BaseModel):
    ts: datetime
    taker_buy_usd: float = 0.0
    taker_sell_usd: float = 0.0

    @classmethod
    def from_raw(cls, row: Dict[str, Any]) -> "TakerVolumeBar":
        ts = row.get("t") or row.get("ts") or row.get("timestamp")
        buy = row.get("buy") or row.get("taker_buy_usd") or row.get("buyVolumeUsd") or 0.0
        sell = row.get("sell") or row.get("taker_sell_usd") or row.get("sellVolumeUsd") or 0.0
        return cls(ts=to_utc_dt(ts), taker_buy_usd=float(buy), taker_sell_usd=float(sell))


class LiquidationEvent(BaseModel):
    ts: datetime
    notional_usd: float

    @classmethod
    def from_raw(cls, row: Dict[str, Any]) -> "LiquidationEvent":
        ts = row.get("t") or row.get("ts") or row.get("timestamp")
        notional = row.get("notional") or row.get("value") or row.get("amountUsd")
        return cls(ts=to_utc_dt(ts), notional_usd=float(notional))


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
    # Use min_periods=1 to avoid NaN at head; small windows approximate
    return s.rolling("30D", min_periods=1).apply(lambda x: (x <= x.iloc[-1]).mean(), raw=False)


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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest_success(day_dir: Path, parquet_file: Path, rows: int, cols: int) -> None:
    manifest = day_dir / "MANIFEST.tsv"
    digest = _sha256_file(parquet_file)
    with open(manifest, "w") as f:
        f.write(f"{parquet_file.name}\t{digest}\t{rows}\t{cols}\n")
    (day_dir / "_SUCCESS").touch()


def write_parquet_daily_files(df: pd.DataFrame, out_root: str, symbol: str) -> None:
    """Write exactly one Parquet file per UTC day.

    Layout: {out_root}/y=YYYY/m=MM/d=DD/part-YYYYMMDD.parquet
    Compression: ZSTD level 3, dictionary enabled.
    Rewrites whole day atomically (cleans existing files before writing).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if df.empty:
        return
    d = add_partitions(df)
    d = d.sort_values("ts")
    for (year, month, day), chunk in d.groupby(["y", "m", "d"], sort=True):
        date_str = f"{int(year):04d}{int(month):02d}{int(day):02d}"
        day_dir = Path(out_root) / f"y={int(year):04d}" / f"m={int(month):02d}" / f"d={int(day):02d}"
        day_dir.mkdir(parents=True, exist_ok=True)

        # Clean existing outputs to honor whole-day rewrite semantics
        for p in day_dir.glob("part-*.parquet"):
            p.unlink(missing_ok=True)  # type: ignore
        (day_dir / "MANIFEST.tsv").unlink(missing_ok=True)  # type: ignore
        (day_dir / "_SUCCESS").unlink(missing_ok=True)  # type: ignore

        out_path = day_dir / f"part-{date_str}.parquet"
        table = pa.Table.from_pandas(chunk.drop(columns=["y", "m", "d"]), preserve_index=False)
        pq.write_table(
            table,
            out_path,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
            # Daily file is small; row_group_size is less relevant. Keep a single row group.
            row_group_size=None,
        )
        _write_manifest_success(day_dir, out_path, rows=table.num_rows, cols=table.num_columns)
