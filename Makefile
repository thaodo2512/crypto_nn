.PHONY: p1_5m_ingest p1_5m_qa p1_5m_view

p1_5m_ingest:
	python ingest_cg_5m.py ingest --symbol BTCUSDT --tf 5m --days 180 --out data/parquet/5m/BTCUSDT

p1_5m_qa:
	python qa_p1_5m.py qa --glob "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --out reports/p1_qa_core_5m.json

p1_5m_view:
	python duckview.py create --db meta/duckdb/p1.duckdb --glob "data/parquet/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" --view bars_5m

.PHONY: p3_label
p3_label:
	python label_p3.py triple-barrier --tf 5m --k 1.2 --H 36 \
	  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --out "data/labels/5m/BTCUSDT"
	python label_p3.py validate \
	  --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --report "reports/p3_qa_5m_80d.json"

.PHONY: p4_sampling
p4_sampling:
	python cli_p4.py iforest-train \
	  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --out data/masks/ifgate_5m.parquet --q 0.995 --rolling-days 30 --seed 42
	python cli_p4.py smote-windows \
	  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --mask data/masks/ifgate_5m.parquet --W 144 --out data/aug/train_smote --seed 42
	python cli_p4.py report-classmix \
	  --pre data/train/ --post data/aug/train_smote/ --out reports/p4_classmix.json

.PHONY: p5_train
p5_train:
	python cli_p5.py train --model gru --window 144 --cv walkforward --embargo 1D \
	  --features "data/features/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --labels "data/labels/5m/BTCUSDT/y=*/m=*/d=*/part-*.parquet" \
	  --out "models/gru_5m" --seed 42 --folds 5

# --- Docker (PC / Jetson) ---
.PHONY: build-pc up-pc down-pc build-jetson up-jetson down-jetson package-model package-data deploy-model deploy-data jetson-health jetson-logs

build-pc:
	docker compose -f docker-compose.pc.yml build

up-pc:
	docker compose -f docker-compose.pc.yml up -d

down-pc:
	docker compose -f docker-compose.pc.yml down -v --remove-orphans

build-jetson:
	docker compose -f docker-compose.jetson.yml build

up-jetson:
	docker compose -f docker-compose.jetson.yml up -d

down-jetson:
	docker compose -f docker-compose.jetson.yml down -v --remove-orphans

package-model:
	bash scripts/pack_model.sh

package-data:
	bash scripts/pack_data.sh data/features/5m/BTCUSDT

deploy-model:
	bash scripts/push_ssh.sh --bundle "$(MODEL_BUNDLE)" --target models

deploy-data:
	bash scripts/push_ssh.sh --bundle "$(DATA_BUNDLE)" --target data

jetson-health:
	bash scripts/healthcheck.sh

jetson-logs:
	ssh -i "$(JETSON_SSH_KEY)" -o StrictHostKeyChecking=no "$(JETSON_USER)@$(JETSON_HOST)" 'sudo journalctl -u app --no-pager -n 200'
print_env:
	@echo SYMS=$(SYMS)
	@echo TF=$(TF)
	@echo WINDOW=$(WINDOW)
	@echo H=$(H)
	@echo DAYS=$(DAYS)

train_all:
	bash scripts/train_full.sh

p1:
	docker compose -f docker-compose.train.yml --profile train run --rm p1_ingest
	docker compose -f docker-compose.train.yml --profile train run --rm p1_ingest python scripts/validators/p1_gate.py

p2:
	docker compose -f docker-compose.train.yml --profile train run --rm p2_features
	docker compose -f docker-compose.train.yml --profile train run --rm p1_ingest python scripts/validators/p2_gate.py

p3:
	docker compose -f docker-compose.train.yml --profile train run --rm p3_label
	docker compose -f docker-compose.train.yml --profile train run --rm p1_ingest python scripts/validators/p3_gate.py

p4:
	docker compose -f docker-compose.train.yml --profile train run --rm p4_sampling || true
	docker compose -f docker-compose.train.yml --profile train run --rm p1_ingest python scripts/validators/p4_gate.py

p5:
	docker compose -f docker-compose.train.yml --profile train run --rm p5_train
	docker compose -f docker-compose.train.yml --profile train run --rm p1_ingest python scripts/validators/p5_gate.py

p6:
	docker compose -f docker-compose.train.yml --profile train run --rm p6_calibrate
	docker compose -f docker-compose.train.yml --profile train run --rm p6_ensemble
	docker compose -f docker-compose.train.yml --profile train run --rm p6_thresholds
	docker compose -f docker-compose.train.yml --profile train run --rm p1_ingest python scripts/validators/p6_gate.py

p8:
	docker compose -f docker-compose.train.yml --profile train run --rm p8_export
	docker compose -f docker-compose.train.yml --profile train run --rm p1_ingest python scripts/validators/p8_gate.py
