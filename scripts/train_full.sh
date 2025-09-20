#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

# Load env
if [[ -f scripts/.env.train ]]; then
  set -a; source scripts/.env.train; set +a
elif [[ -f scripts/env.train ]]; then
  set -a; source scripts/env.train; set +a
elif [[ -f scripts/env.train.example ]]; then
  set -a; source scripts/env.train.example; set +a
fi

log() { echo "[train] $*"; }

run_service() {
  local svc="$1"; shift || true
  log "Running service: ${svc}"
  docker compose -f docker-compose.train.yml --profile train run --rm "${svc}" "$@"
}

gate() {
  local phase="$1"; shift || true
  log "Validating gate: ${phase}"
  docker compose -f docker-compose.train.yml --profile train run --rm p1_ingest \
    python "scripts/validators/${phase}_gate.py"
}

mkdir -p logs reports artifacts export

# P1
run_service p1_ingest
gate p1

# P2
run_service p2_features
gate p2

# P3
run_service p3_label
gate p3

# P4
run_service p4_sampling || true
gate p4

# P5
run_service p5_train
gate p5

# P6
run_service p6_calibrate
run_service p6_ensemble
run_service p6_thresholds
gate p6

# P8
run_service p8_export
gate p8

log "Training pipeline completed. Artifacts:"
log " - models/: $(ls -1 models 2>/dev/null | wc -l) entries"
log " - reports/: $(ls -1 reports 2>/dev/null | wc -l) entries"
log " - export/:  $(ls -1 export 2>/dev/null | wc -l) entries"
