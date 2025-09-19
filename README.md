# CoinGlass 15m BTCUSDT Pipeline

Small Python 3.11 pipeline to ingest CoinGlass v4 endpoints and build 15‑minute BTCUSDT Parquet with QA and DuckDB helpers.

## Quick Start
- Install: `pip install -e .[dev]`
- Auth: set `COINGLASS_API_KEY` in your environment.
- Ingest: `python ingest_cg.py ingest coinglass --conf conf/p1_inputs_cg.yaml`
- QA: `python qa_p1.py qa --glob "data/parquet/15m/BTCUSDT/**/*.parquet" --out reports/p1_qa_core.json`
- DuckDB view: `python qa_p1.py duckdb-view --glob "data/parquet/15m/BTCUSDT/**/*.parquet" --view bars_15m`

## Config (example)
`conf/p1_inputs_cg.yaml` should define:
```yaml
base_url: https://open-api-v4.coinglass.com/api
symbol: BTCUSDT
start: 2024-01-01T00:00:00Z
end: 2024-06-30T23:59:59Z
out_dir: data/parquet/15m/BTCUSDT
```

## Endpoints Used (v4)
- Futures price OHLC: `futures/price/history`
- OI aggregated OHLC: `futures/open-interest/aggregated-history` (close → oi_now)
- Funding (primary/fallback): `futures/funding-rate/oi-weight-history`, `.../vol-weight-history`
- Futures taker buy/sell: `futures/taker-buy-sell-volume/history`
- Aggregated taker (spot/perp): `spot/aggregated-taker-buy-sell-volume/history`, `futures/aggregated-taker-buy-sell-volume/history`
- Liquidations (coin aggregated): `futures/liquidation/heatmap/coin/model1`

## Features & Storage
- Features: rv_15m, cvd_perp_15m, perp_share_60m, oi_pctile_30d, funding_pctile_30d.
- Funding reindexed to 15m with ffill ≤3 bars, flag: `funding_now_imputed`.
- Parquet: partitioned by `y/m/d`, ZSTD level 3, key `[symbol, ts]` (UTC, 15m, right-closed/label=right).

## QA
- Reports last 180d: expected vs present bars, missing per column, imputation ratios (funding/OI), duplicates.
- Fails if gaps > 0.5% or any imputation > 5%.

## Testing
- Run: `pytest -q`
- Includes resampling correctness and key uniqueness tests.

## Notes
- No secrets committed; use `COINGLASS_API_KEY` for header `CG-API-KEY`.
- Adjust endpoints/params in `ingest_cg.py` if your tenant differs.
