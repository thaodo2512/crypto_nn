# CoinGlass BTCUSDT Pipeline (5m default)

Small Python 3.11 pipeline to ingest CoinGlass v4 endpoints and build 5‑minute BTCUSDT Parquet with QA and DuckDB helpers.

## Quick Start (5m default)
- Docker Compose (recommended)
  - Build: `docker compose build`
  - Set key: copy `secrets/.env.example` to `secrets/.env` and fill `COINGLASS_API_KEY`
  - Ingest 5m: `docker compose run --rm ingest_5m`
  - QA 5m: `docker compose run --rm qa_5m`
  - DuckDB view 5m: `docker compose run --rm duckdb_view_5m`
  - Labels P3: `make p3_label`
  - Sampling P4: `make p4_sampling` (rebuild image first)
  - Train P5: `make p5_train` (rebuild image first)
  - Policy P7: `docker compose run --rm p7_decide`
  - Export P8: `docker compose run --rm p8_export` (rebuild image first)
  - Explain P10: `python explain_p10.py api --port 8081`
  - Monitor P11: see Phase P11 commands below

## Phase P2 – 5m Feature Builder
- Build features (from P1 5m bars):
  - Docker: `docker compose run --rm features_build`
  - Local: `python features_p2.py build --glob "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --out "data/features/5m/BTCUSDT" --warmup 8640 --winsor_low 0.01 --winsor_high 0.99`
- Validate acceptance (gaps≤0.5%, impute≤5%, no NaN):
  - Docker: `docker compose run --rm features_validate`
  - Local: `python features_p2.py validate --glob "data/features/5m/BTCUSDT/**/*.parquet" --qa reports/p2_qa_5m_80d.json --schema reports/p2_feature_schema.json --horizon_days 80 --warmup 8640`
- Benchmark (< 50 ms/bar):
  - Docker: `docker compose run --rm features_bench`
  - Local: `python features_p2.py bench --glob "data/features/5m/BTCUSDT/**/*.parquet" --report reports/p2_bench_5m_80d.csv`
- DuckDB view over features:
  - Docker: `docker compose run --rm duck_view_feat`
  - Local: `python duck_view.py create-view --glob "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --view feat_5m --db meta/duckdb/p1.duckdb`

### P2 Acceptance Checker (one-shot summary)
- Docker (recommended): `docker compose run --rm p2_check`
  - Writes `reports/p2_check_5m_80d.json` and a wide CSV table `reports/p2_check_5m_80d_table.csv`.
  - Fails if: feature count not in [10,20], any NaN>0, any `_imputed_*` ratio>5%, or 5m gap ratio>0.5% (and if provided, ms_per_bar>50).
- Local equivalent:
  - `python p2_check.py run \
     --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
     --raw "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
     --bench-csv "reports/p2_bench_5m_80d.csv" \
     --out-json "reports/p2_check_5m_80d.json"`

## Phase P3 – Triple-Barrier Labels (5m)
- Build labels:
  - Docker: `docker compose run --rm labels_build` (or `make p3_label`)
  - Local: `python label_p3.py triple-barrier --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --raw "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --out "data/labels/5m/BTCUSDT" --tf 5m --k 1.2 --H 36 --atr_window 14`
- Validate labels:
  - Docker: `docker compose run --rm labels_validate`
  - Validator (acceptance gate + rulecheck): `docker compose run --rm p3_validate`
  - Local: `python label_p3.py validate --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --report reports/p3_qa_5m_80d.json`
- Sample:
  - Local: `python label_p3.py sample --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --n 10`

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

## Phase P4 – Sampling (IF gate + SMOTE, 5m)
- Overview
  - Isolation Forest gate on a 30‑day rolling window (q=0.995) selects “anomalous” timestamps → mask at `data/masks/ifgate_5m.parquet`.
  - SMOTE applies ONLY to TRAIN folds on LONG/SHORT windows (W=144 bars), never on WAIT or OOS/VAL.
  - Purged walk‑forward CV with 1‑day embargo between TRAIN and VAL/OOS.
  - Acceptance: WAIT share in TRAIN ≤ 60% for every fold; OOS untouched.
- Docker Compose:
  - IF gate: `docker compose run --rm p4_iforest`
  - SMOTE windows: `docker compose run --rm p4_smote`
  - Class mix report: `docker compose run --rm p4_classmix`
  - End-to-end: `docker compose run --rm p4_pipeline`
- Local commands
  - IF gate: `python cli_p4.py iforest-train --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --out data/masks/ifgate_5m.parquet --q 0.995 --rolling-days 30 --seed 42`
  - SMOTE windows: `python cli_p4.py smote-windows --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --mask data/masks/ifgate_5m.parquet --W 144 --out data/aug/train_smote --seed 42`
  - Report: `python cli_p4.py report-classmix --pre data/train/ --post data/aug/train_smote/ --out reports/p4_classmix.json`
- Outputs
  - Mask: `data/masks/ifgate_5m.parquet` (ts, symbol, keep, fold_id)
  - Augmented: `data/aug/train_smote/<fold>/train.parquet`
  - Report: `reports/p4_classmix.json`; Logs: `logs/p4_sampling.log`
- Note: Rebuild Docker image after adding P4 deps (`scikit-learn`, `imbalanced-learn`): `docker compose build`.

## Incremental Ingestion
- Skips days that already have `_SUCCESS` and `MANIFEST.tsv` markers.
- Always refreshes the last N days (default `--refresh-tail 2`) to capture late data.
- Force full reload: add `--force --no-skip-present`.
- Debug requests/responses: use the `ingest_debug` service or add `--debug`.
- Verbose logs: 5m ingester prints progress by default; add `--no-verbose` to silence.

Examples
- Refresh recent days only (defaults): `docker compose run --rm ingest`
- Force full 180‑day reload: `docker compose run --rm ingest --force --no-skip-present`
- Show request params and response keys: `docker compose run --rm ingest_debug`

## Local (optional)
- Install: `pip install -r requirements.txt -r requirements-dev.txt`
- Auth: export `COINGLASS_API_KEY`
- P1 (5m, 80d):
  - Ingest: `python ingest_cg_5m.py --symbol BTCUSDT --tf 5m --days 80 --out data/parquet/5m/BTCUSDT`
  - QA: `python qa_p1_5m.py --glob "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --out reports/p1_qa_core_5m_80d.json --days 80`
  - DuckDB view: `python duckview.py create --db meta/duckdb/p1.duckdb --glob "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --view bars_5m`
  
Legacy 15m (optional):
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

## Storage & Integrity
- P1 5m lake: `data/parquet/5m/{symbol}/y=YYYY/m=MM/d=DD/part-YYYYMMDD.parquet` (UTC, right‑closed 5m).
- ZSTD(3), dictionary on; keys `[symbol, ts]`; partitions `y/m/d`.
- Per day integrity: `MANIFEST.tsv` (sha256) and `_SUCCESS` marker; whole‑day rewrite semantics; incremental ingest supported.

## QA
- P1 5m 80d acceptance: expected_bars_80d=23040, gap_ratio<0.5%, NaN=0, imputed≤5% (funding/oi only).
- P2 5m acceptance: usable last 50d, ms_per_bar≤50, NaN=0, imputed≤5%.

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

## Phase P5 – Small NN Training (GRU)
- Docker: `docker compose run --rm p5_train`
- Local: `python cli_p5.py train --model gru --window 144 --cv walkforward --embargo 1D \
  --features "data/features/5m/BTCUSDT/**/part-*.parquet" --labels "data/labels/5m/BTCUSDT/**/part-*.parquet" \
  --out models/gru_5m --seed 42`

P5 Validation (artifacts)
- Docker: `docker compose run --rm p5_validate`
- Local: `python p5_validate.py run \
  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
  --labels   "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
  --folds artifacts/folds.json \
  --models "models/gru_5m/*/best.pt" \
  --oos-probs "artifacts/p5_oos_probs/fold*.parquet" \
  --train-log logs/p5_train.log \
  --metrics reports/p5_cv_metrics.json \
  --out-json reports/p5_validate.json --tz UTC --embargo 1D --window 144`
  - Notes: Validator now enforces OOS probabilities per fold (no missing folds); if `artifacts/folds.json` is absent it builds fallback folds but still requires per‑fold OOS.

Export OOS probabilities (for validation/calibration)
- Docker: `docker compose run --rm p5_oos_export`
- Local: `python p5_export_oos.py run \
  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
  --labels   "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
  --models-root models/gru_5m --out artifacts/p5_oos_probs --embargo 1D --folds 5 --window 144`
- Model: GRU(64,1), dropout=0.2, weight_decay=1e-4; loss: weighted CE with time-decay (≈0.98/day).
- CV: purged walk-forward with 1-day embargo, deterministic seed; checkpoints per fold under `models/gru_5m/<fold>/best.pt`.
- Metrics: `reports/p5_cv_metrics.json`; logs under `logs/p5_train.log`.
- Note: requires PyTorch; rebuild Docker image after adding P5 deps.

## Phase P6 – Calibration + Ensemble + τ Tuning
- Calibrate temps: `docker compose run --rm p6_calibrate`
- Ensemble by EV weights: `docker compose run --rm p6_ensemble`
- Tune τ: `docker compose run --rm p6_tune_threshold`
- Validate P6 artifacts: `docker compose run --rm p6_validate`
- Expected artifacts:
  - Calibration JSON: `models/calib.json` (per-fold temperature, ECE)
  - Ensemble JSON: `models/ensemble_5m.json` (non‑negative weights summing to 1)
  - Tuning summary: `reports/p6_oos_summary.json` (best_tau, EV curves) and `reports/p6_curves.png`
  - Inputs: VAL+OOS probs in `artifacts/p5_oos_probs/fold*.parquet`

## Phase P7 – EV Gating Policy
- Decide: `python policy_p7.py decide --probs artifacts/p6_calibrated.parquet --atr data/atr_5m.parquet --k-range-min 1.0 --k-range-max 1.5 --H 36 --out decisions/`
- Output (partitioned): `decisions/y=YYYY/m=MM/d=DD/part-YYYYMMDD.parquet` with columns ts, side, size, TP_px, SL_px, EV, reason.

## Phase P8 – ONNX FP16 Export + Parity
- Docker: `docker compose run --rm p8_export`
- Local: `python export_p8.py onnx --ckpt "models/gru_5m/*/best.pt" --fp16 --out export/model_5m_fp16.onnx --preproc conf/preproc_5m.yaml --calib models/ensemble_5m.json --window 144 --sample 16`
- Outputs: `export/model_5m_fp16.onnx`, `reports/p8_parity.json` (MSE, sha256), logs in `logs/p8_export.log`.
- Acceptance: parity MSE < 1e-3 and checksum logged.

## Train‑Only Stack (Offline P1→P8)
- Compose file: `docker-compose.train.yml` (profile `train`). No ports, no edge services.
- Environment: copy `scripts/env.train.example` → `scripts/env.train` and edit if needed.
  - Keys: `SYMS`, `TF` (5m), `WINDOW` (144), `H` (36), `DAYS`, `CUDA_VISIBLE_DEVICES`.
- Orchestrator (end‑to‑end with gates):
  - `bash scripts/train_full.sh`
  - Runs phases sequentially with acceptance gates; fails fast on any violation.
- Individual phases (compose):
  - `docker compose -f docker-compose.train.yml --profile train run --rm p1_ingest`
  - `docker compose -f docker-compose.train.yml --profile train run --rm p2_features`
  - `docker compose -f docker-compose.train.yml --profile train run --rm p3_label`
  - `docker compose -f docker-compose.train.yml --profile train run --rm p4_sampling`
  - `docker compose -f docker-compose.train.yml --profile train run --rm p5_train`   (GPU only here)
  - `docker compose -f docker-compose.train.yml --profile train run --rm p6_calibrate && \
     docker compose -f docker-compose.train.yml --profile train run --rm p6_thresholds`
  - `docker compose -f docker-compose.train.yml --profile train run --rm p8_export`
- Validators (acceptance gates): `scripts/validators/*_gate.py`
  - P1: gaps≤0.5%, NaN=0
  - P2: 10–20 features, NaN=0
  - P3: labels join features 1:1
  - P4: TRAIN WAIT≤60%
  - P5: CV=walkforward, embargo=1D, window=144, no divergence
  - P6: EV/trade>0 (95% CI), ECE≤10%
  - P8: MSE(probs)<1e‑3 and ONNX checksum present
- Makefile shortcuts:
  - `make print_env` – show effective SYMS/TF/WINDOW/H/DAYS
  - `make train_all` – run full offline pipeline with gates
  - `make p1` … `make p6`, `make p8` – run a single phase + gate

## Phase P10 – Explanations (IG + optional IF‑SHAP)
- Generate (parity‑guarded):
  - `python -m app.explain.cli run \
     --decision-id <id> \
     --window-npy /path/to/window.npy \
     --ckpt models/gru_5m/fold0/best.pt \
     --onnx export/model_5m_fp16.onnx \
     --steps 32 --topk 10 --target 1 [--baseline zeros|feature_means]`
  - Optional IF‑SHAP at alert time: add `--if-csv alerts.csv` (expects columns: id, alert=1).
- API (TTL 30 days):
  - `python -m app.explain.cli api --port 8081 --dir explain`
  - Endpoints: `GET /health`, `HEAD/GET /explain?id=<id>`
- GC: `python -m app.explain.cli gc --dir explain --ttl-days 30`
- JSON schema stored at `explain/<id>.json`:
  - `{ id, ts_unix, window_shape:[144,F], features:[...], topk:[{t,f,attr}], summary:{sum,l1,l2,max_abs}, if_shap:[{f,shap}]? }`
  - Atomic writes; expired files get 404 and are removed by GC.

## Phase P11 – Monitoring, Drift, Regimes, Retrain
- Drift + metrics: `python monitor_p11.py run --features "data/features/5m/**/part-*.parquet" --decisions "decisions/**/part-*.parquet" --out reports/`
- Regimes (CUSUM on rv_5m): `python monitor_p11.py regimes --rv "data/features/5m/**/part-*.parquet" --out reports/p11_regimes.json`
- Retrain trigger: `python monitor_p11.py retrain --config conf/retrain.yaml --out ops/retrain_trigger.json`
- Outputs: `reports/p11_drift.json`, `reports/p11_regimes.json`, `ops/retrain_trigger.json`; MLflow logs with regime_id, thresholds, EV summary.
