# CoinGlass 15m BTCUSDT Pipeline

Small Python 3.11 pipeline to ingest CoinGlass v4 endpoints and build 15‑minute BTCUSDT Parquet with QA and DuckDB helpers.

## Quick Start
- Docker Compose (recommended)
  - Build: `docker compose build`
  - Set key: copy `secrets/.env.example` to `secrets/.env` and fill `COINGLASS_API_KEY`
  - Ingest: `docker compose run --rm ingest`
  - QA: `docker compose run --rm qa`
- DuckDB view: `docker compose run --rm duckdb_view`

## Incremental Ingestion
- Skips days that already have `_SUCCESS` and `MANIFEST.tsv` markers.
- Always refreshes the last N days (default `--refresh-tail 2`) to capture late data.
- Force full reload: add `--force --no-skip-present`.
- Debug requests/responses: use the `ingest_debug` service or add `--debug`.

Examples
- Refresh recent days only (defaults): `docker compose run --rm ingest`
- Force full 180‑day reload: `docker compose run --rm ingest --force --no-skip-present`
- Show request params and response keys: `docker compose run --rm ingest_debug`

## Local (optional)
- Install: `pip install -r requirements.txt -r requirements-dev.txt`
- Auth: export `COINGLASS_API_KEY`
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
- Lake layout: `data/parquet/15m/{symbol}/y=YYYY/m=MM/d=DD/part-YYYYMMDD.parquet` (one file/day, UTC, right-closed bars).
- Parquet format: ZSTD(3), dictionary on; keys `[symbol, ts]`. Partitions: `y/m/d`.
- Integrity per day: writes `MANIFEST.tsv` with sha256 and a `_SUCCESS` marker; re-writes by whole day for late data.
 - Incremental: detects already‑ingested days via markers and only re‑fetches missing or tail days.

## QA
- Reports last 180d: expected vs present bars, missing per column, imputation ratios (funding/OI), duplicates, NaN counts for derived features.
- Gates (configurable via `qa_targets`): gaps < 0.5%, impute < 5% per feature, no NaN after transforms.

## Testing
- Run: `pytest -q`
- Includes resampling correctness and key uniqueness tests.

## Notes
- No secrets committed; use `COINGLASS_API_KEY` for header `CG-API-KEY`.
- Adjust endpoints/params in `ingest_cg.py` if your tenant differs.

## DuckDB Catalog
- Catalog DB: `meta/duckdb/p1.duckdb` with view `bars_15m` over the partition glob.
- Create/update: `docker compose run --rm duckdb_view` (or `python qa_p1.py duckdb-view --glob ... --view bars_15m --db meta/duckdb/p1.duckdb`).
