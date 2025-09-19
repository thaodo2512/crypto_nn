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
    write_parquet_partitioned,
)


app = typer.Typer(help="Data ingestion pipeline for CoinGlass → 15m BTCUSDT Parquet")
ingest_app = typer.Typer(help="Ingestion subcommands")
app.add_typer(ingest_app, name="ingest")


DEFAULT_BASE_URL = "https://open-api-v4.coinglass.com/api"


@dataclass
class IngestConfig:
    base_url: str
    symbol: str
    start: str  # ISO8601 or epoch
    end: str
    out_dir: str


def _load_config(path: str) -> IngestConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return IngestConfig(
        base_url=cfg.get("base_url", DEFAULT_BASE_URL),
        symbol=cfg["symbol"],
        start=str(cfg["start"]),
        end=str(cfg["end"]),
        out_dir=cfg.get("out_dir", f"data/parquet/15m/{cfg['symbol']}")
    )


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


def _fetch_price_ohlc(client: CGClient, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    # Endpoint naming may differ; these params are typical
    rows = client.get_paginated(
        path="futures/price/history",
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms},
    )
    df = _safe_parse(PriceBar, rows)
    return resample_15m_ohlcv(df)


def _fetch_oi(client: CGClient, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = client.get_paginated(
        path="futures/open-interest/aggregated-history",
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


def _fetch_funding(client: CGClient, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    # Funding per exchange or weighted; reindex→15m ffill≤3
    params = {"symbol": symbol, "startTime": start_ms, "endTime": end_ms}
    rows = client.get_paginated(path="futures/funding-rate/oi-weight-history", params=params)
    df = _safe_parse(FundingBar, rows)
    if df.empty:
        rows = client.get_paginated(path="futures/funding-rate/vol-weight-history", params=params)
        df = _safe_parse(FundingBar, rows)
    return reindex_15m_ffill_limit(df, "funding_now", limit=3)


def _fetch_taker_perp(client: CGClient, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = client.get_paginated(
        path="futures/taker-buy-sell-volume/history",
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms},
    )
    df = _safe_parse(TakerVolumeBar, rows)
    if df.empty:
        return pd.DataFrame(columns=["ts", "taker_buy_usd", "taker_sell_usd"])  # typed
    return resample_15m_sum(df, ["taker_buy_usd", "taker_sell_usd"])  # perp


def _fetch_taker_spot_agg(client: CGClient, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    # Aggregate spot and perp taker volumes if endpoint available
    rows_spot = client.get_paginated(
        path="spot/aggregated-taker-buy-sell-volume/history",
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
        path="futures/aggregated-taker-buy-sell-volume/history",
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


def _fetch_liq(client: CGClient, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = client.get_paginated(
        path="futures/liquidation/heatmap/coin/model1",
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms},
    )
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
    d["rv_15m"] = (pd.Series(d["close"]).div(pd.Series(d["close"]).shift(1)).apply(lambda x: math.log(x) if pd.notna(x) and x > 0 else float("nan")) ** 2)
    # perp taker net and CVD on 15m
    net_perp = (d.get("taker_buy_usd", 0) - d.get("taker_sell_usd", 0)).fillna(0)
    d["cvd_perp_15m"] = net_perp.cumsum()
    # perp/spot rolling share (60m)
    perp_60 = pd.Series(d.get("perp_taker_buy_usd", 0)).add(pd.Series(d.get("perp_taker_sell_usd", 0))).rolling("60min", min_periods=1).sum()
    spot_60 = pd.Series(d.get("spot_taker_buy_usd", 0)).add(pd.Series(d.get("spot_taker_sell_usd", 0))).rolling("60min", min_periods=1).sum()
    denom = perp_60.add(spot_60)
    d["perp_share_60m"] = perp_60.divide(denom.where(denom != 0, pd.NA))
    # Percentiles (30D) for oi and funding
    d = d.set_index("ts").sort_index()
    if "oi_now" in d:
        d["oi_pctile_30d"] = rolling_percentile_30d(d["oi_now"]).values
    if "funding_now" in d:
        d["funding_pctile_30d"] = rolling_percentile_30d(d["funding_now"]).values
    d = d.reset_index()
    return d


@ingest_app.command("coinglass")
def ingest_coinglass(
    conf: str = typer.Option(..., "--conf", help="YAML config path (p1_inputs_cg.yaml)"),
) -> None:
    """Ingest CoinGlass endpoints and build 15m Parquet with features."""
    cfg = _load_config(conf)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    start_ms = _ts_to_epoch_ms(cfg.start)
    end_ms = _ts_to_epoch_ms(cfg.end)

    client = CGClient(base_url=cfg.base_url)

    typer.echo("Fetching price OHLC...")
    price15 = _fetch_price_ohlc(client, cfg.symbol, start_ms, end_ms)

    typer.echo("Fetching OI...")
    oi15 = _fetch_oi(client, cfg.symbol, start_ms, end_ms)

    typer.echo("Fetching funding...")
    funding15 = _fetch_funding(client, cfg.symbol, start_ms, end_ms)

    typer.echo("Fetching perp taker volumes...")
    perp_taker15 = _fetch_taker_perp(client, cfg.symbol, start_ms, end_ms)

    typer.echo("Fetching spot/perp aggregated taker volumes...")
    agg_taker15 = _fetch_taker_spot_agg(client, cfg.symbol, start_ms, end_ms)

    typer.echo("Fetching liquidation heatmap...")
    liq15 = _fetch_liq(client, cfg.symbol, start_ms, end_ms)

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
    df = add_partitions(df)
    write_parquet_partitioned(df, cfg.out_dir)
    typer.echo(f"Wrote Parquet dataset to {cfg.out_dir}")


if __name__ == "__main__":
    app()
