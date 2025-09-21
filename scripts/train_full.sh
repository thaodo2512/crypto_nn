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

if [[ "${BUILD_ONCE:-1}" == "1" ]]; then
  log "Building training image (once)..."
  docker compose -f docker-compose.train.yml --profile train build
fi

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

# Quick mode to reduce runtime
# When QUICK=1, force DAYS to a short horizon regardless of any pre-set DAYS
# to guarantee smaller data pulls and faster iteration.
if [[ "${QUICK:-0}" == "1" ]]; then
  export DAYS="${QUICK_DAYS:-7}"
  export SAMPLE="${QUICK_SAMPLE:-512}"
  log "QUICK=1 enabled: override DAYS=${DAYS} SAMPLE=${SAMPLE}"
fi

# P1
if [[ "${SKIP_P1:-0}" != "1" ]]; then
  run_service p1_ingest
  gate p1
fi

# P2
if [[ "${SKIP_P2:-0}" != "1" ]]; then
  run_service p2_features
  gate p2
fi

# P3
if [[ "${SKIP_P3:-0}" != "1" ]]; then
  run_service p3_label
  gate p3
fi

# P4
if [[ "${SKIP_P4:-0}" != "1" ]]; then
  run_service p4_sampling || true
  gate p4
fi

# P5
has_nvidia_smi() { command -v nvidia-smi >/dev/null 2>&1; }
has_nvidia_runtime() { docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -qi nvidia; }

if [[ "${SKIP_P5:-0}" != "1" ]]; then
  svc=p5_train
  if has_nvidia_smi; then
    svc=p5_train_gpu
  elif has_nvidia_runtime; then
    svc=p5_train_gpu_jetson
  fi
  log "Selecting P5 service: ${svc}"
  run_service "${svc}"
  gate p5
  # Auto-export OOS probabilities for Phase 6 inputs
  log "Exporting OOS probabilities for calibration/ensemble..."
  run_service p5_oos_export
fi

# P6
if [[ "${SKIP_P6:-0}" != "1" ]]; then
  run_service p6_calibrate
  run_service p6_ensemble
  run_service p6_thresholds
  gate p6
fi

# P8
if [[ "${SKIP_P8:-0}" != "1" ]]; then
  run_service p8_export
  gate p8
fi

log "Training pipeline completed. Artifacts:"
log " - models/: $(ls -1 models 2>/dev/null | wc -l) entries"
log " - reports/: $(ls -1 reports 2>/dev/null | wc -l) entries"
log " - export/:  $(ls -1 export 2>/dev/null | wc -l) entries"
