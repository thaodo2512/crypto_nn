# Repository Guidelines

## Project Structure
- Root CLI modules per phase: `ingest_cg_5m.py` (P1), `features_p2.py` (P2), `label_p3.py` (P3),
  `cli_p4.py` (P4), `cli_p5.py` (P5/P6), `export_p8.py` (P8), `policy_p7.py` (P7),
  `service_p9.py` (P9), `explain_p10.py` (P10), `monitor_p11.py` (P11).
- Utilities: `utils_cg.py`, transforms in `transforms.py`, sampling in `smote_train.py`, folds in `folds.py`.
- Tests: `tests/test_*.py` per phase. Data lakes under `data/`, models under `models/`, export under `export/`.
- Compose services in `docker-compose.yml`; Make targets in `Makefile`.

## Development & Testing
- Python 3.11. Install deps: `pip install -r requirements.txt -r requirements-dev.txt`.
- Run tests: `pytest -q` (set `PYTHONPATH=/app` when using Docker run).
- Formatting: Black; linting: Ruff; type-hints encouraged; no secrets in repo.

## Key Workflows (5m default)
- P1 (ingest/QA/view): `docker compose run --rm ingest_5m | qa_5m | duckdb_view_5m`.
- P2 (features): `docker compose run --rm features_build | features_validate | features_bench`.
- P2 checker (summary gate): `docker compose run --rm p2_check` (emits `reports/p2_check_5m_80d.json`).
- P3 (labels): `docker compose run --rm labels_build` then `labels_validate` or `p3_validate`.
- P4 (sampling): `docker compose run --rm p4_iforest | p4_smote | p4_classmix` (or `p4_pipeline`).
- P5 (train): `docker compose run --rm p5_train` (writes models under `models/gru_5m`).
- P5 (OOS export): `docker compose run --rm p5_oos_export` (writes `artifacts/p5_oos_probs/fold*.parquet`).
- P5 (validate): `docker compose run --rm p5_validate` (checks ckpts, OOS probs, CV integrity, window shape, loss/time-decay hints).
- P6 (calibrate/ensemble/tune τ): `p6_calibrate`, `p6_ensemble`, `p6_tune_threshold` compose services.
- P7 (policy): `python policy_p7.py decide --probs ... --atr ... --out decisions/`.
- P8 (export ONNX FP16): `docker compose run --rm p8_export`.
- P9 (service): `python service_p9.py api --onnx export/model_5m_fp16.onnx --port 8080`.
- P10 (explain):
  - Generate: `python -m app.explain.cli run --decision-id <id> --window-npy <win.npy> --ckpt models/gru_5m/fold0/best.pt --onnx export/model_5m_fp16.onnx [--if-csv alerts.csv]`
  - API: `python -m app.explain.cli api --port 8081 --dir explain` (or `docker compose run --rm -p 8081:8081 p10_explain_api`)
  - GC: `python -m app.explain.cli gc --dir explain --ttl-days 30`
- P11 (monitor): `python monitor_p11.py run ...` and `regimes` then `retrain`.

## Data & Storage
- Parquet partitioning: `y=YYYY/m=MM/d=DD/part-YYYYMMDD.parquet`. Keys `[symbol, ts]` unique, UTC, right‑closed.
- Integrity: per-day `MANIFEST.tsv` + `_SUCCESS`. Feature/label lake under `data/`; decisions under `decisions/`.

## Code Style & Conventions
- 4-space indent, line length ≤ 100; `snake_case` names; `PascalCase` classes.
- Keep implementations deterministic (seed=42) where applicable; tests must not rely on network or secrets.

## PR & Commit
- Conventional commits (feat/fix/chore/docs/test/refactor). Include tests and README/Compose updates for new CLIs.
- Do not lower test coverage or relax acceptance gates without explicit approval.
