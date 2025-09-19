from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import typer

from utils_cg import (
    CGClient,
    FundingBar,
    LiquidationEvent,
    OIBar,
    PriceBar,
    TakerVolumeBar,
    ensure_unique_key,
    reindex_5m_ffill_limit,
    resample_5m_ohlcv,
    resample_5m_sum,
    write_parquet_daily_files,
)


app = typer.Typer(help="P1 5m – Ingest CoinGlass endpoints and build 5m Parquet lake")


DEFAULT_BASE_URL = "https://open-api-v4.coinglass.com/api"


@dataclass
class IngestArgs:
    symbol: str
    tf: str
    days: int
    out: str
    exchange: Optional[str]
    base_url: str


def _normalize_exchange(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    mapping = {
        "BINANCE": "Binance",
        "BYBIT": "Bybit",
        "BITGET": "Bitget",
        "BITFINEX": "Bitfinex",
        "HUOBI": "Huobi",
        "OKX": "OKX",
    }
    return mapping.get(str(name).upper(), name)


def _build_params(symbol: str, start_ms: int, end_ms: int, interval: str, exchange: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p: Dict[str, Any] = {
        "symbol": symbol,
        "start_time": start_ms,
        "end_time": end_ms,
        "interval": interval,
    }
    ex = _normalize_exchange(exchange)
    if ex:
        p["exchange"] = ex
    if extra:
        p.update(extra)
    return p


def _timeslice_fetch(client: CGClient, path: str, symbol: str, start_ms: int, end_ms: int, interval: str, exchange: Optional[str], window_days: int = 30, limit: int = 4500) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    step = pd.Timedelta(days=window_days)
    cur = pd.to_datetime(start_ms, unit="ms", utc=True)
    end = pd.to_datetime(end_ms, unit="ms", utc=True)
    while cur < end:
        nxt = min(cur + step, end)
        params = _build_params(symbol, int(cur.value // 1_000_000), int(nxt.value // 1_000_000), interval, exchange, {"limit": limit})
        payload = client.get_json(path, params)
        rows = client._extract_records(payload)
        if rows:
            results.extend(rows)
        cur = nxt
    return results


def _safe_parse(model, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    recs = []
    for r in rows:
        try:
            rec = model.from_raw(r)
            recs.append(rec.dict())
        except Exception:
            continue
    return pd.DataFrame.from_records(recs)


def _ts_to_epoch_ms(ts: str) -> int:
    return int(pd.to_datetime(ts, utc=True).value // 1_000_000)


@app.command("ingest")
def ingest_coinglass(
    source: str = typer.Argument("coinglass"),
    symbol: str = typer.Option("BTCUSDT", "--symbol"),
    tf: str = typer.Option("5m", "--tf"),
    days: int = typer.Option(80, "--days"),
    out: str = typer.Option("data/parquet/5m/BTCUSDT", "--out"),
    exchange: Optional[str] = typer.Option("BINANCE", "--exchange"),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--base-url"),
) -> None:
    if source.lower() != "coinglass":
        raise typer.BadParameter("Only 'coinglass' source is supported")
    if tf.lower() != "5m":
        raise typer.BadParameter("Only 5m timeframe is supported in P1-5m")

    args = IngestArgs(symbol=symbol, tf=tf.lower(), days=days, out=out.rstrip("/"), exchange=exchange, base_url=base_url.rstrip("/"))
    Path(args.out).mkdir(parents=True, exist_ok=True)

    now = pd.Timestamp.now(tz="UTC")
    start_ms = int((now - pd.Timedelta(days=args.days)).value // 1_000_000)
    end_ms = int(now.value // 1_000_000)

    client = CGClient(base_url=args.base_url)

    # Endpoint mapping
    ep_price = "futures/price/history"
    ep_oi_agg = "futures/open-interest/aggregated-history"
    ep_fund_list = "futures/funding-rate/exchange-list"
    ep_fund_oiw = "futures/funding-rate/oi-weight-history"
    ep_taker_fut = "futures/taker-buy-sell-volume/history"
    ep_taker_spot_agg = "spot/aggregated-taker-buy-sell-volume/history"
    ep_taker_fut_agg = "futures/aggregated-taker-buy-sell-volume/history"
    ep_liq = "futures/liquidation/heatmap/model1"

    # Fetch price (base grid)
    price_rows = _timeslice_fetch(client, ep_price, args.symbol, start_ms, end_ms, args.tf, args.exchange)
    price_df = _safe_parse(PriceBar, price_rows)
    price_5 = resample_5m_ohlcv(price_df)

    if price_5.empty:
        typer.echo("No OHLCV rows; aborting.")
        raise typer.Exit(code=1)

    # OI aggregated → 5m (last) then ffill≤3
    oi_rows = _timeslice_fetch(client, ep_oi_agg, args.symbol, start_ms, end_ms, args.tf, None)
    oi_df = _safe_parse(OIBar, oi_rows)
    oi_5 = pd.DataFrame(columns=["ts", "oi_now"]) if oi_df.empty else (
        oi_df.assign(ts=pd.to_datetime(oi_df["ts"], utc=True)).set_index("ts").sort_index()[["oi_now"]]
        .resample("5min", closed="right", label="right").last().reset_index()
    )
    oi_5 = reindex_5m_ffill_limit(oi_5, "oi_now", limit=3)
    oi_5 = oi_5.rename(columns={"oi_now_imputed": "_imputed_oi_now"})

    # Funding: try per-exchange list; fallback to oi-weighted
    fund_rows = _timeslice_fetch(client, ep_fund_oiw, args.symbol, start_ms, end_ms, args.tf, None)
    fund_df = _safe_parse(FundingBar, fund_rows)
    if fund_df.empty:
        fund_rows = _timeslice_fetch(client, ep_fund_list, args.symbol, start_ms, end_ms, args.tf, args.exchange)
        fund_df = _safe_parse(FundingBar, fund_rows)
    fund_5 = reindex_5m_ffill_limit(fund_df, "funding_now", limit=3)
    fund_5 = fund_5.rename(columns={"funding_now_imputed": "_imputed_funding_now"})

    # Futures taker buy/sell → sum to 5m
    taker_rows = _timeslice_fetch(client, ep_taker_fut, args.symbol, start_ms, end_ms, args.tf, args.exchange)
    taker_df = _safe_parse(TakerVolumeBar, taker_rows)
    taker_5 = resample_5m_sum(taker_df, ["taker_buy_usd", "taker_sell_usd"]) if not taker_df.empty else pd.DataFrame(columns=["ts", "taker_buy_usd", "taker_sell_usd"])  # type: ignore

    # Optionals: spot/perp aggregated takers
    spot_rows = _timeslice_fetch(client, ep_taker_spot_agg, args.symbol, start_ms, end_ms, args.tf, None)
    spot_df = _safe_parse(TakerVolumeBar, spot_rows)
    spot_5 = resample_5m_sum(spot_df, ["taker_buy_usd", "taker_sell_usd"]) if not spot_df.empty else pd.DataFrame(columns=["ts", "taker_buy_usd", "taker_sell_usd"])  # type: ignore
    if not spot_5.empty:
        spot_5 = spot_5.rename(columns={"taker_buy_usd": "spot_taker_buy_usd", "taker_sell_usd": "spot_taker_sell_usd"})

    futagg_rows = _timeslice_fetch(client, ep_taker_fut_agg, args.symbol, start_ms, end_ms, args.tf, None)
    futagg_df = _safe_parse(TakerVolumeBar, futagg_rows)
    futagg_5 = resample_5m_sum(futagg_df, ["taker_buy_usd", "taker_sell_usd"]) if not futagg_df.empty else pd.DataFrame(columns=["ts", "taker_buy_usd", "taker_sell_usd"])  # type: ignore
    if not futagg_5.empty:
        futagg_5 = futagg_5.rename(columns={"taker_buy_usd": "perp_taker_buy_usd", "taker_sell_usd": "perp_taker_sell_usd"})

    # Liquidation notional aggregated to 5m
    liq_rows = _timeslice_fetch(client, ep_liq, args.symbol, start_ms, end_ms, args.tf, args.exchange)
    liq_df = _safe_parse(LiquidationEvent, liq_rows)
    liq_5 = resample_5m_sum(liq_df, ["notional_usd"]).rename(columns={"notional_usd": "liq_notional_raw"}) if not liq_df.empty else pd.DataFrame(columns=["ts", "liq_notional_raw"])  # type: ignore

    # Compose
    base = price_5.copy()
    base = base.rename(columns={"volume_usd": "volume"})
    for piece in (oi_5, fund_5, taker_5, spot_5, futagg_5, liq_5):
        if not piece.empty:
            base = pd.merge(base, piece, on="ts", how="left")

    base["symbol"] = args.symbol

    # Fill any residual NaN in optional numeric columns with 0; exogenous are already ffilled with flags
    numeric_cols = [c for c in base.columns if c not in {"ts", "symbol"}]
    for c in numeric_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    for c in numeric_cols:
        base[c] = base[c].fillna(0.0)

    # Key uniqueness and sort
    ensure_unique_key(base, ["symbol", "ts"])
    base = base.sort_values(["symbol", "ts"]).reset_index(drop=True)

    # Daily partitioned write
    write_parquet_daily_files(base, args.out, args.symbol)

    # Emit simple sources map
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/p1_sources.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "endpoint", "notes"])
        w.writerow(["ohlcv", ep_price, f"interval={args.tf}, exchange={_normalize_exchange(args.exchange)}"])
        w.writerow(["funding_now", f"{ep_fund_oiw} OR {ep_fund_list}", "ffill<=3, flags in _imputed_funding_now"])
        w.writerow(["oi_now", ep_oi_agg, "resample last to 5m, ffill<=3, flags in _imputed_oi_now"])
        w.writerow(["taker_fut", ep_taker_fut, "sum to 5m (USD)"])
        w.writerow(["taker_spot_agg", ep_taker_spot_agg, "sum to 5m (USD)"])
        w.writerow(["taker_fut_agg", ep_taker_fut_agg, "sum to 5m (USD)"])
        w.writerow(["liq_notional_raw", ep_liq, "sum to 5m from events/heatmap"])


if __name__ == "__main__":
    app()
