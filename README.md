# CoinGlass 15m BTCUSDT Pipeline

Small Python 3.11 pipeline to ingest CoinGlass v4 endpoints and build 15‑minute BTCUSDT Parquet with QA and DuckDB helpers.

## Quick Start (5m default)
- Docker Compose (recommended)
  - Build: `docker compose build`
  - Set key: copy `secrets/.env.example` to `secrets/.env` and fill `COINGLASS_API_KEY`
  - Ingest 5m: `docker compose run --rm ingest_5m`
  - QA 5m: `docker compose run --rm qa_5m`
  - DuckDB view 5m: `docker compose run --rm duckdb_view_5m`

## Phase P2 – 5m Feature Builder
- Build features (from P1 5m bars):
  - Docker: `docker compose run --rm features_build`
  - Local: `python features_p2.py build --glob "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --out "data/features/5m/BTCUSDT" --warmup 8640 --winsor_low 0.01 --winsor_high 0.99`
- Validate acceptance (gaps≤0.5%, impute≤5%, no NaN):
  - Docker: `docker compose run --rm features_validate`
  - Local: `python features_p2.py validate --glob "data/features/5m/BTCUSDT/**/*.parquet" --qa reports/p2_qa.json`
- Benchmark (< 50 ms/bar):
  - Docker: `docker compose run --rm features_bench`
  - Local: `python features_p2.py bench --glob "data/features/5m/BTCUSDT/**/*.parquet" --report reports/p2_bench.csv`
- DuckDB view over features:
  - Docker: `docker compose run --rm duck_view_feat`
  - Local: `python duck_view.py create-view --glob "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --view feat_5m --db meta/duckdb/p1.duckdb`

Inputs/Outputs
- Input (P1): `data/parquet/5m/BTCUSDT/y=YYYY/m=MM/d=DD/part-*.parquet` with keys `[symbol, ts]`, UTC, 5m.
- Output (P2): `data/features/5m/BTCUSDT/y=YYYY/m=MM/d=DD/part-YYYYMMDD.parquet`, ZSTD(3), dictionary on, one file/day, MANIFEST.tsv + `_SUCCESS`.

Feature Columns (no NaN)
- ret_5m, ret_1h, ret_4h, hl_range, co_ret
- vol_z, oi_z, fund_now_z, funding_pctile_30d
- cvd_diff_z, perp_share_60m, liq60_z, rv_5m_z
- hour_of_week_sin, hour_of_week_cos
- _imputed_funding_now, _imputed_oi_now

Causal Transforms
- Warmup: 30 days (8640 bars) per symbol (drop before warmup).
- Winsorize causal (1–99%) and z-score causal on 30d past window.
- Only funding_now and oi_now may ffill≤3 bars (flags emitted).

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
- P1 5m: view `bars_5m` over `data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet`.
  - Create/update: `docker compose run --rm duckdb_view_5m` (or `python duckview.py create --glob ... --view bars_5m --db meta/duckdb/p1.duckdb`).
- P1 15m (legacy): view `bars_15m` over 15m lake via `docker compose run --rm duckdb_view`.
