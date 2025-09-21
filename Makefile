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

.PHONY: clean-artifacts clean-old
clean-artifacts:
	bash scripts/cleanup.sh --purge-artifacts --yes

# Usage: make clean-old SYM=BTCUSDT TF=5m KEEP_DAYS=60 WHAT=features|raw|both
clean-old:
	@if [ "$(WHAT)" = "raw" ]; then \
	  KEEP_DAYS=$(KEEP_DAYS) SYM=$(SYM) TF=$(TF) bash scripts/cleanup.sh --prune-raw --yes; \
	elif [ "$(WHAT)" = "both" ]; then \
	  KEEP_DAYS=$(KEEP_DAYS) SYM=$(SYM) TF=$(TF) bash scripts/cleanup.sh --prune-features --prune-raw --yes; \
	else \
	  KEEP_DAYS=$(KEEP_DAYS) SYM=$(SYM) TF=$(TF) bash scripts/cleanup.sh --prune-features --yes; \
	fi

.PHONY: keep-best-model
keep-best-model:
	SYM=$(SYM) YES=1 bash scripts/cleanup.sh --keep-best-model --yes

#############################################
# GCP ephemeral training (gcloud + Make)
#############################################

# Defaults (override via env when invoking make)
GCP_PROJECT ?= valiant-epsilon-472304-r9
GCP_REGION  ?= us-central1
GCP_ZONE    ?= us-central1-c
GCP_NAME    ?= train-$(shell date +%Y%m%d-%H%M%S)
GCP_MACHINE ?= c2-standard-4
GCP_DISK    ?= 50
GCP_IMG     ?= projects/ubuntu-os-cloud/global/images/ubuntu-minimal-2504-plucky-amd64-v20250911
GCP_IP_NAME ?= $(GCP_NAME)-ip
GCP_SA      ?= 137846157442-compute@developer.gserviceaccount.com

GCP_SCOPES = https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append

GCP_NET_IF_BASE = network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default
GCP_NET_IF_WITH_IP = $(GCP_NET_IF_BASE),address=$(GCP_IP_NAME)

GCP_CREATE_FLAGS = \
 --project=$(GCP_PROJECT) \
 --zone=$(GCP_ZONE) \
 --machine-type=$(GCP_MACHINE) \
 --maintenance-policy=MIGRATE \
 --provisioning-model=STANDARD \
 --service-account=$(GCP_SA) \
 --scopes=$(GCP_SCOPES) \
 --min-cpu-platform="Intel Cascade Lake" \
 --tags=ssh \
 --create-disk=auto-delete=yes,boot=yes,device-name=$(GCP_NAME),image=$(GCP_IMG),mode=rw,size=$(GCP_DISK),type=pd-balanced \
 --no-shielded-secure-boot \
 --shielded-vtpm \
 --shielded-integrity-monitoring \
 --labels=goog-ec-src=vm_add-gcloud \
 --reservation-affinity=any \
 --threads-per-core=2 \
 --visible-core-count=2 \
 --metadata-from-file=startup-script=scripts/startup.sh

.PHONY: gcp-create-ip gcp-create gcp-create-with-ip gcp-wait gcp-push gcp-train gcp-tail gcp-pull gcp-destroy gcp-release-ip gcp-ssh gcp-tensorboard gcp-docker-build gcp-wait-train gcp-one

gcp-create-ip:
	@set -euxo pipefail; \
	if gcloud compute addresses describe "$(GCP_IP_NAME)" --project="$(GCP_PROJECT)" --region="$(GCP_REGION)" >/dev/null 2>&1; then \
	  echo "Static IP already exists: $(GCP_IP_NAME)"; \
	else \
	  gcloud compute addresses create "$(GCP_IP_NAME)" --project="$(GCP_PROJECT)" --region="$(GCP_REGION)"; \
	fi

gcp-create:
	@set -euxo pipefail; \
	if gcloud compute instances describe "$(GCP_NAME)" --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" >/dev/null 2>&1; then \
	  echo "Instance already exists: $(GCP_NAME)"; \
	else \
	  gcloud compute instances create "$(GCP_NAME)" \
	    $(GCP_CREATE_FLAGS) \
	    --network-interface=$(GCP_NET_IF_BASE); \
	fi

gcp-create-with-ip: gcp-create-ip
	@set -euxo pipefail; \
	if gcloud compute instances describe "$(GCP_NAME)" --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" >/dev/null 2>&1; then \
	  echo "Instance already exists: $(GCP_NAME)"; \
	else \
	  gcloud compute instances create "$(GCP_NAME)" \
	    $(GCP_CREATE_FLAGS) \
	    --network-interface=$(GCP_NET_IF_WITH_IP); \
	fi

gcp-wait:
	@set -euxo pipefail; \
	echo "Waiting for READY_FOR_REPO in serial console..."; \
	for i in $$(seq 1 120); do \
	  if gcloud compute instances get-serial-port-output "$(GCP_NAME)" --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" --port=1 --start=0 2>/dev/null | grep -q 'READY_FOR_REPO'; then \
	    echo "Instance is ready"; exit 0; \
	  fi; \
	  sleep 5; \
	done; echo "Timeout waiting for READY_FOR_REPO"; exit 1

.PHONY: gcp-wait-verbose gcp-serial
gcp-wait-verbose:
	@set -euxo pipefail; \
	echo "Waiting for READY_FOR_REPO with verbose serial output..."; \
	for i in $$(seq 1 120); do \
	  OUT=$$(gcloud compute instances get-serial-port-output "$(GCP_NAME)" --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" --port=1 --start=0 2>/dev/null || true); \
	  echo "--- [Attempt $$i] Last 120 lines of serial ---"; echo "$$OUT" | tail -n 120; \
	  echo "-------------------------------------------"; \
	  if echo "$$OUT" | grep -q 'READY_FOR_REPO'; then echo "Instance is ready"; exit 0; fi; \
	  sleep 5; \
	done; echo "Timeout waiting for READY_FOR_REPO"; exit 1

gcp-serial:
	@set -euxo pipefail; \
	LINES=$${LINES:-200}; \
	gcloud compute instances get-serial-port-output "$(GCP_NAME)" --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" --port=1 --start=0 | tail -n $$LINES

gcp-push:
	@set -euxo pipefail; \
	tar -czf /tmp/repo.tgz --exclude-from=.gcloudignore -C . .; \
	gcloud compute scp --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" /tmp/repo.tgz "$(GCP_NAME)":~/repo.tgz; \
	gcloud compute ssh --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" "$(GCP_NAME)" --command="mkdir -p ~/repo && tar -xzf ~/repo.tgz -C ~/repo && rm -f ~/repo.tgz"

gcp-docker-build:
	@set -euxo pipefail; \
	gcloud compute ssh --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" "$(GCP_NAME)" --command="bash -lc 'cd ~/repo; if [ -f Dockerfile ]; then docker build -t repo/train:latest .; elif [ -f docker/pc.Dockerfile ]; then docker build -t repo/train:latest -f docker/pc.Dockerfile .; else echo \"No Dockerfile found; skipping build\"; fi'"

gcp-train:
	@set -euxo pipefail; \
	gcloud compute ssh --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" "$(GCP_NAME)" --command="bash -lc 'sudo groupadd -f docker; sudo usermod -aG docker $$USER || true; if command -v tmux >/dev/null 2>&1; then tmux new -d -s train \"sg docker -c \\\"bash ~/repo/scripts/train_compose.sh\\\"\"; else nohup sg docker -c \"bash ~/repo/scripts/train_compose.sh\" > ~/train.log 2>&1 < /dev/null & fi'"; \
	echo "Training started (docker compose). Use make gcp-tail to follow logs."

gcp-wait-train:
	@set -euxo pipefail; \
	TIMEOUT_SEC=$${GCP_TIMEOUT:-7200}; ITER=$$(( (TIMEOUT_SEC + 9) / 10 )); \
	echo "Waiting for /work/artifacts-*.tgz to appear (timeout $$TIMEOUT_SEC s) ..."; \
	for i in $$(seq 1 $$ITER); do \
	  if gcloud compute ssh --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" "$(GCP_NAME)" --command="bash -lc 'ls /work/artifacts-*.tgz >/dev/null 2>&1'"; then \
	    echo "Training finished (artifact tarball present)."; exit 0; \
	  fi; \
	  sleep 10; \
	done; echo "Timeout waiting for training completion"; exit 1

gcp-tail:
	@set -euxo pipefail; \
	gcloud compute ssh --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" "$(GCP_NAME)" --command="tail -n +1 -f ~/train.log"

gcp-pull:
	@set -euxo pipefail; \
	mkdir -p artifacts; \
	(gcloud compute scp --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" --recurse "$(GCP_NAME)":/work/artifacts ./artifacts || true); \
	(gcloud compute scp --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" "$(GCP_NAME)":/work/artifacts-*.tgz ./artifacts || true); \
	echo "Artifacts copied to ./artifacts (if present)."

gcp-destroy:
	@set -euxo pipefail; \
	gcloud compute instances delete "$(GCP_NAME)" --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" --quiet || true

gcp-release-ip:
	@set -euxo pipefail; \
	gcloud compute addresses delete "$(GCP_IP_NAME)" --project="$(GCP_PROJECT)" --region="$(GCP_REGION)" --quiet || true

gcp-ssh:
	@set -euxo pipefail; \
	gcloud compute ssh --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" "$(GCP_NAME)"

gcp-tensorboard:
	@set -euxo pipefail; \
	echo "Forwarding local 6006 → remote 6006. Ctrl+C to stop."; \
	gcloud compute ssh --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" "$(GCP_NAME)" -- -L 6006:localhost:6006

# One-shot end-to-end: create → wait → push → (optional docker build) → train → wait → pull → (optional destroy)
# Controls:
#  - GCP_USE_IP=1 to reserve & attach static IP
#  - GCP_DOCKER_BUILD=1 to build Docker image remotely if Dockerfile exists (default 0)
#  - GCP_KEEP_VM=1 to skip teardown at the end (default 0)
#  - GCP_TIMEOUT=SECONDS to bound training wait (default 7200)
gcp-one:
	@set -euxo pipefail; \
	if [ "$${GCP_USE_IP:-0}" = "1" ]; then $(MAKE) gcp-create-with-ip; else $(MAKE) gcp-create; fi; \
	$(MAKE) gcp-wait; \
	$(MAKE) gcp-push; \
	if [ "$${GCP_DOCKER_BUILD:-0}" = "1" ]; then $(MAKE) gcp-docker-build; fi; \
	$(MAKE) gcp-train; \
	$(MAKE) gcp-wait-train; \
	$(MAKE) gcp-pull; \
	if [ "$${GCP_KEEP_VM:-0}" != "1" ]; then $(MAKE) gcp-destroy; if [ "$${GCP_USE_IP:-0}" = "1" ]; then $(MAKE) gcp-release-ip; fi; fi

.PHONY: gcp-one-remote gcp-one-multi
gcp-one-remote:
	@set -euxo pipefail; \
	if [ "$${GCP_USE_IP:-0}" = "1" ]; then $(MAKE) gcp-create-with-ip; else $(MAKE) gcp-create; fi; \
	$(MAKE) gcp-wait; \
	$(MAKE) gcp-push; \
	if [ "$${GCP_DOCKER_BUILD:-0}" = "1" ]; then $(MAKE) gcp-docker-build; fi; \
	$(MAKE) gcp-train-remote; \
	$(MAKE) gcp-wait-train; \
	$(MAKE) gcp-pull; \
	if [ "$${GCP_KEEP_VM:-0}" != "1" ]; then $(MAKE) gcp-destroy; if [ "$${GCP_USE_IP:-0}" = "1" ]; then $(MAKE) gcp-release-ip; fi; fi

gcp-one-multi:
	@set -euxo pipefail; \
	if [ "$${GCP_USE_IP:-0}" = "1" ]; then $(MAKE) gcp-create-with-ip; else $(MAKE) gcp-create; fi; \
	$(MAKE) gcp-wait; \
	$(MAKE) gcp-push; \
	if [ "$${GCP_DOCKER_BUILD:-0}" = "1" ]; then $(MAKE) gcp-docker-build; fi; \
	$(MAKE) gcp-train-remote-multi; \
	$(MAKE) gcp-wait-train; \
	$(MAKE) gcp-pull; \
	if [ "$${GCP_KEEP_VM:-0}" != "1" ]; then $(MAKE) gcp-destroy; if [ "$${GCP_USE_IP:-0}" = "1" ]; then $(MAKE) gcp-release-ip; fi; fi

# ---- Custom remote training with parameters (no need to open interactive SSH) ----
# Pass parameters as Make vars: SYMS, TF, WINDOW, H, DAYS, QUICK
# Example (single symbol, 80 days):
#   SYMS=ETHUSDT QUICK=0 DAYS=80 make gcp-train-remote GCP_NAME=test2
# Example (multi-symbol):
#   SYMS="BTCUSDT,ETHUSDT" QUICK=0 DAYS=80 make gcp-train-remote-multi GCP_NAME=test2

.PHONY: gcp-train-remote gcp-train-remote-multi
gcp-train-remote:
	@set -euxo pipefail; \
	REMOTE_ENV="SYMS='$(SYMS)' TF='$(TF)' WINDOW='$(WINDOW)' H='$(H)' DAYS='$(DAYS)' QUICK='$(QUICK)'"; \
	gcloud compute ssh --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" "$(GCP_NAME)" --command="bash -lc 'sudo groupadd -f docker; sudo usermod -aG docker $$USER || true; if command -v tmux >/dev/null 2>&1; then tmux new -d -s train \"sg docker -c \\\"$$REMOTE_ENV bash ~/repo/scripts/train_compose.sh\\\"\"; else nohup sg docker -c \"$$REMOTE_ENV bash ~/repo/scripts/train_compose.sh\" > ~/train.log 2>&1 < /dev/null & fi'"; \
	echo "Remote training (compose) started with $$REMOTE_ENV"

gcp-train-remote-multi:
	@set -euxo pipefail; \
	REMOTE_ENV="SYMS='$(SYMS)' TF='$(TF)' WINDOW='$(WINDOW)' H='$(H)' DAYS='$(DAYS)' QUICK='$(QUICK)'"; \
	gcloud compute ssh --tunnel-through-iap --project="$(GCP_PROJECT)" --zone="$(GCP_ZONE)" "$(GCP_NAME)" --command="bash -lc 'sudo groupadd -f docker; sudo usermod -aG docker $$USER || true; if command -v tmux >/dev/null 2>&1; then tmux new -d -s train \"sg docker -c \\\"$$REMOTE_ENV bash ~/repo/scripts/train_multi.sh\\\"\"; else nohup sg docker -c \"$$REMOTE_ENV bash ~/repo/scripts/train_multi.sh\" > ~/train.log 2>&1 < /dev/null & fi'"; \
	echo "Remote multi-symbol training started with $$REMOTE_ENV"
SHELL := /bin/bash
