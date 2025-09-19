import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import typer
import yaml

from utils_cg import (
    CGClient,
    FundingBar,
    LiquidationEvent,
    OIBar,
    PriceBar,
    TakerVolumeBar,
    add_partitions,
    ensure_unique_key,
    resample_15m_ohlcv,
    resample_15m_sum,
    reindex_15m_ffill_limit,
    rolling_percentile_30d,
    write_parquet_daily_files,
)


app = typer.Typer(help="Data ingestion pipeline for CoinGlass → 15m BTCUSDT Parquet")
ingest_app = typer.Typer(help="Ingestion subcommands")
app.add_typer(ingest_app, name="ingest")


DEFAULT_BASE_URL = "https://open-api-v4.coinglass.com/api"


@dataclass
class IngestConfig:
    raw: Dict[str, Any]

    @property
    def base_url(self) -> str:
        return self.raw.get("base_url", DEFAULT_BASE_URL)

    @property
    def symbol(self) -> str:
        return self.raw["symbol"]

    @property
    def timeframe(self) -> str:
        return str(self.raw.get("timeframe", "15m")).lower()

    @property
    def horizon_days(self) -> Optional[int]:
        return int(self.raw.get("horizon_days", 0)) or None

    @property
    def start(self) -> Optional[str]:
        return self.raw.get("start")

    @property
    def end(self) -> Optional[str]:
        return self.raw.get("end")

    @property
    def sinks(self) -> Dict[str, Any]:
        return self.raw.get("sinks", {})

    @property
    def out_dir(self) -> str:
        # Prefer sinks.parquet_root, else legacy out_dir
        sinks = self.sinks
        if "parquet_root" in sinks:
            return str(sinks["parquet_root"]).rstrip("/")
        return self.raw.get("out_dir", f"data/parquet/15m/{self.symbol}").rstrip("/")

    @property
    def scope(self) -> Dict[str, Any]:
        return self.raw.get("scope", {})

    @property
    def endpoints(self) -> Dict[str, str]:
        return self.raw.get("endpoints", {})


def _load_config(path: str) -> IngestConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return IngestConfig(raw=cfg)


def _ts_to_epoch_ms(ts: str) -> int:
    t = pd.to_datetime(ts, utc=True)
    return int(t.value // 1_000_000)


def _safe_parse(model, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    recs = []
    for r in rows:
        try:
            rec = model.from_raw(r)
            recs.append(rec.dict())
        except Exception:
            continue
    return pd.DataFrame.from_records(recs)


def _fetch_price_ohlc(client: CGClient, symbol: str, start_ms: int, end_ms: int, endpoint_path: str) -> pd.DataFrame:
    # Endpoint naming may differ; these params are typical
    rows = client.get_paginated(
        path=endpoint_path,
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms},
    )
    df = _safe_parse(PriceBar, rows)
    return resample_15m_ohlcv(df)


def _fetch_oi(client: CGClient, symbol: str, start_ms: int, end_ms: int, endpoint_path: str) -> pd.DataFrame:
    rows = client.get_paginated(
        path=endpoint_path,
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms},
    )
    df = _safe_parse(OIBar, rows)
    if df.empty:
        return pd.DataFrame(columns=["ts", "oi_now"])  # typed
    tmp = df.copy()
    tmp["ts"] = pd.to_datetime(tmp["ts"], utc=True)
    tmp = tmp.set_index("ts").sort_index()
    out = tmp[["oi_now"]].resample("15min", closed="right", label="right").last().reset_index()
    return out


def _fetch_funding(
    client: CGClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    endpoint_exchange_list: Optional[str],
    endpoint_oi_weight: Optional[str],
    endpoint_vol_weight: Optional[str],
    exchange_preference: Optional[str],
    fallback_agg: bool,
) -> pd.DataFrame:
    # Funding per exchange or weighted; reindex→15m ffill≤3
    params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms}
    # Try per-exchange first if preference provided
    if endpoint_exchange_list and exchange_preference:
        p = dict(params)
        # Best-effort exchange param naming
        p.update({"exchange": exchange_preference, "exchangeName": exchange_preference})
        try:
            rows = client.get_paginated(path=endpoint_exchange_list, params=p)
            df = _safe_parse(FundingBar, rows)
            if not df.empty:
                return reindex_15m_ffill_limit(df, "funding_now", limit=3)
        except Exception:
            # fall through to weighted if allowed
            pass
    if fallback_agg:
        if endpoint_oi_weight:
            rows = client.get_paginated(path=endpoint_oi_weight, params=params)
            df = _safe_parse(FundingBar, rows)
            if not df.empty:
                return reindex_15m_ffill_limit(df, "funding_now", limit=3)
        if endpoint_vol_weight:
            rows = client.get_paginated(path=endpoint_vol_weight, params=params)
            df = _safe_parse(FundingBar, rows)
            if not df.empty:
                return reindex_15m_ffill_limit(df, "funding_now", limit=3)
    # default empty
    return pd.DataFrame(columns=["ts", "funding_now", "funding_now_imputed"])  # typed


def _fetch_taker_perp(client: CGClient, symbol: str, start_ms: int, end_ms: int, endpoint_path: str) -> pd.DataFrame:
    rows = client.get_paginated(
        path=endpoint_path,
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms},
    )
    df = _safe_parse(TakerVolumeBar, rows)
    if df.empty:
        return pd.DataFrame(columns=["ts", "taker_buy_usd", "taker_sell_usd"])  # typed
    return resample_15m_sum(df, ["taker_buy_usd", "taker_sell_usd"])  # perp


def _fetch_taker_spot_agg(
    client: CGClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    endpoint_spot_agg: str,
    endpoint_fut_agg: str,
) -> pd.DataFrame:
    # Aggregate spot and perp taker volumes if endpoint available
    rows_spot = client.get_paginated(
        path=endpoint_spot_agg,
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms},
    )
    spot_df = _safe_parse(TakerVolumeBar, rows_spot)
    spot_15 = resample_15m_sum(spot_df, ["taker_buy_usd", "taker_sell_usd"]) if not spot_df.empty else pd.DataFrame(columns=["ts", "taker_buy_usd", "taker_sell_usd"])  # type: ignore
    if not spot_15.empty:
        spot_15 = spot_15.rename(columns={
            "taker_buy_usd": "spot_taker_buy_usd",
            "taker_sell_usd": "spot_taker_sell_usd",
        })
    rows_perp_agg = client.get_paginated(
        path=endpoint_fut_agg,
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms},
    )
    perp_df = _safe_parse(TakerVolumeBar, rows_perp_agg)
    perp_15 = resample_15m_sum(perp_df, ["taker_buy_usd", "taker_sell_usd"]) if not perp_df.empty else pd.DataFrame(columns=["ts", "taker_buy_usd", "taker_sell_usd"])  # type: ignore
    if not perp_15.empty:
        perp_15 = perp_15.rename(columns={
            "taker_buy_usd": "perp_taker_buy_usd",
            "taker_sell_usd": "perp_taker_sell_usd",
        })
    # Merge spot + perp agg
    out = pd.merge(spot_15, perp_15, on="ts", how="outer")
    return out


def _fetch_liq(
    client: CGClient,
    symbol: str,
    start_ms: int,
    end_ms: int,
    endpoint_path: str,
    exchange_preference: Optional[str] = None,
) -> pd.DataFrame:
    def _extract_rows(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for k in ("data", "list", "rows", "result"):
                v = payload.get(k)
                if isinstance(v, list):
                    return v
            d = payload.get("data")
            if isinstance(d, dict):
                for k in ("list", "rows", "result"):
                    v = d.get(k)
                    if isinstance(v, list):
                        return v
        return []

    params_base: Dict[str, Any] = {"symbol": symbol}
    if exchange_preference:
        params_base.update({"exchange": exchange_preference, "exchangeName": exchange_preference})

    # Try direct time-bounded query first
    try:
        payload = client.get_json(endpoint_path, {**params_base, "startTime": start_ms, "endTime": end_ms})
        rows = _extract_rows(payload)
    except Exception:
        rows = []

    # Fallback: range-based snapshots stepping backward
    if not rows:
        day_ms = 24 * 60 * 60 * 1000
        acc: List[Dict[str, Any]] = []
        for win in (30, 7, 3, 1):
            step_end = end_ms
            used_any = False
            while step_end > start_ms:
                p = {**params_base, "range": f"{win}d", "endTime": step_end}
                try:
                    payload = client.get_json(endpoint_path, p)
                    part = _extract_rows(payload)
                    if part:
                        acc.extend(part)
                        used_any = True
                    # Move window back by win days
                    step_end -= win * day_ms
                except Exception:
                    break
            if used_any:
                rows = acc
                break

    df = _safe_parse(LiquidationEvent, rows)
    if df.empty:
        return pd.DataFrame(columns=["ts", "liq_notional_usd_60m"])  # typed
    agg15 = resample_15m_sum(df, ["notional_usd"]).rename(columns={"notional_usd": "liq_notional_usd"})
    agg15["ts"] = pd.to_datetime(agg15["ts"], utc=True)
    agg15 = agg15.set_index("ts").sort_index()
    agg15["liq_notional_usd_60m"] = agg15["liq_notional_usd"].rolling("60min", min_periods=1).sum()
    return agg15.reset_index()[["ts", "liq_notional_usd_60m"]]


def _compose_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d = d.sort_values("ts")
    # rv_15m
    d["rv_15m"] = (
        pd.Series(d["close"]).div(pd.Series(d["close"]).shift(1)).apply(lambda x: math.log(x) if pd.notna(x) and x > 0 else 0.0) ** 2
    )
    # perp taker net and CVD on 15m
    net_perp = (d.get("taker_buy_usd", 0) - d.get("taker_sell_usd", 0)).fillna(0)
    d["cvd_perp_15m"] = net_perp.cumsum()
    # perp/spot rolling share (60m)
    perp_60 = pd.Series(d.get("perp_taker_buy_usd", 0)).fillna(0).add(pd.Series(d.get("perp_taker_sell_usd", 0)).fillna(0)).rolling("60min", min_periods=1).sum()
    spot_60 = pd.Series(d.get("spot_taker_buy_usd", 0)).fillna(0).add(pd.Series(d.get("spot_taker_sell_usd", 0)).fillna(0)).rolling("60min", min_periods=1).sum()
    denom = perp_60.add(spot_60)
    d["perp_share_60m"] = perp_60.divide(denom.replace(0, 1e-9))
    # Percentiles (30D) for oi and funding
    d = d.set_index("ts").sort_index()
    if "oi_now" in d:
        d["oi_pctile_30d"] = rolling_percentile_30d(d["oi_now"]).fillna(0.5).values
    if "funding_now" in d:
        d["funding_pctile_30d"] = rolling_percentile_30d(d["funding_now"]).fillna(0.5).values
    d = d.reset_index()
    return d


@ingest_app.command("coinglass")
def ingest_coinglass(
    conf: str = typer.Option(..., "--conf", help="YAML config path (p1_inputs_cg.yaml)"),
) -> None:
    """Ingest CoinGlass endpoints and build 15m Parquet with features."""
    cfg = _load_config(conf)
    # Time window: prefer explicit start/end; else horizon_days → [now - horizon_days, now]
    if cfg.start and cfg.end:
        start_ms = _ts_to_epoch_ms(str(cfg.start))
        end_ms = _ts_to_epoch_ms(str(cfg.end))
    else:
        now = pd.Timestamp.now(tz="UTC")
        hz = cfg.horizon_days or 180
        start_ms = int((now - pd.Timedelta(days=hz)).value // 1_000_000)
        end_ms = int(now.value // 1_000_000)

    if cfg.timeframe != "15m":
        raise typer.BadParameter(f"Unsupported timeframe {cfg.timeframe}; only 15m is implemented.")

    out_dir = cfg.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    client = CGClient(base_url=cfg.base_url)

    # Endpoint resolution
    logical_to_v4 = {
        "price-ohlc-history": "futures/price/history",
        "oi-ohlc-aggregated-history": "futures/open-interest/aggregated-history",
        "fr-exchange-list": "futures/funding-rate/exchange-list",
        "oi-weight-ohlc-history": "futures/funding-rate/oi-weight-history",
        "vol-weight-ohlc-history": "futures/funding-rate/vol-weight-history",
        "taker-buysell-volume": "futures/taker-buy-sell-volume/history",
        "aggregated-taker-buysell-volume-history": "futures/aggregated-taker-buy-sell-volume/history",
        "spot-aggregated-taker-buysell-history": "spot/aggregated-taker-buy-sell-volume/history",
        "liquidation-aggregate-heatmap": "futures/liquidation/heatmap/model1",
    }

    ep = cfg.endpoints
    ep_price = logical_to_v4.get(ep.get("futures_ohlcv", "price-ohlc-history"))
    ep_oi = logical_to_v4.get(ep.get("oi_ohlc_agg", "oi-ohlc-aggregated-history"))
    ep_funding_list = logical_to_v4.get(ep.get("funding_now_list", "fr-exchange-list"))
    ep_funding_oi = logical_to_v4.get(ep.get("funding_oi_weighted", "oi-weight-ohlc-history"))
    ep_funding_vol = logical_to_v4.get(ep.get("funding_vol_weighted", "vol-weight-ohlc-history"))
    ep_taker_perp = logical_to_v4.get(ep.get("fut_taker", "taker-buysell-volume"))
    ep_taker_fut_agg = logical_to_v4.get(ep.get("fut_taker_agg", "aggregated-taker-buysell-volume-history"))
    ep_taker_spot_agg = logical_to_v4.get(ep.get("spot_taker_agg", "spot-aggregated-taker-buysell-history"))
    ep_liq = logical_to_v4.get(ep.get("liq_heatmap_agg", "liquidation-aggregate-heatmap"))

    scope = cfg.scope
    exchange_pref = scope.get("exchange_preference")
    fallback_agg = bool(scope.get("fallback_agg", True))

    typer.echo("Fetching price OHLC...")
    price15 = _fetch_price_ohlc(client, cfg.symbol, start_ms, end_ms, ep_price)

    typer.echo("Fetching OI...")
    oi15 = _fetch_oi(client, cfg.symbol, start_ms, end_ms, ep_oi)

    typer.echo("Fetching funding...")
    funding15 = _fetch_funding(
        client,
        cfg.symbol,
        start_ms,
        end_ms,
        ep_funding_list,
        ep_funding_oi,
        ep_funding_vol,
        exchange_pref,
        fallback_agg,
    )

    typer.echo("Fetching perp taker volumes...")
    perp_taker15 = _fetch_taker_perp(client, cfg.symbol, start_ms, end_ms, ep_taker_perp)

    typer.echo("Fetching spot/perp aggregated taker volumes...")
    agg_taker15 = _fetch_taker_spot_agg(
        client, cfg.symbol, start_ms, end_ms, ep_taker_spot_agg, ep_taker_fut_agg
    )

    typer.echo("Fetching liquidation heatmap...")
    liq15 = _fetch_liq(client, cfg.symbol, start_ms, end_ms, ep_liq, exchange_pref)

    # Compose base join on ts
    df = price15
    for piece in (oi15, funding15, perp_taker15, agg_taker15, liq15):
        if not piece.empty:
            df = pd.merge(df, piece, on="ts", how="left")

    df["symbol"] = cfg.symbol
    df = _compose_features(df)

    # Ensure key uniqueness and 15m alignment
    ensure_unique_key(df, ["symbol", "ts"])

    # Partitioned write
    write_parquet_daily_files(df, out_dir, cfg.symbol)
    typer.echo(f"Wrote daily Parquet dataset under {out_dir}")


if __name__ == "__main__":
    app()
