#!/usr/bin/env bash
set -euxo pipefail

LOG=~/train.log
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ART_ROOT="/work/artifacts"
RUN_DIR="${ART_ROOT}/run-${RUN_ID}"
TARBALL="/work/artifacts-${RUN_ID}.tgz"

{
  echo "=== Compose Train start: ${RUN_ID} (UTC) ==="
  echo "Host: $(hostname)"
  echo "Kernel: $(uname -a)"
} | tee -a "${LOG}"

cd ~/repo

# Ensure docker and compose are present (compose plugin installed in startup)
docker --version | tee -a "${LOG}"
docker compose version | tee -a "${LOG}"

# Build images if needed
docker compose -f docker-compose.train.yml --profile train build | tee -a "${LOG}"

# Run training (env flags supported). If SYMS contains a comma → multi-symbol runner.
RUN_ENV=(
  QUICK="${QUICK:-1}"
  QUICK_DAYS="${QUICK_DAYS:-7}"
  SKIP_P1="${SKIP_P1:-0}" SKIP_P2="${SKIP_P2:-0}" SKIP_P3="${SKIP_P3:-0}" SKIP_P4="${SKIP_P4:-0}"
  SKIP_P5="${SKIP_P5:-0}" SKIP_P6="${SKIP_P6:-0}" SKIP_P8="${SKIP_P8:-0}"
  TF="${TF:-5m}" WINDOW="${WINDOW:-144}" H="${H:-36}" DAYS="${DAYS:-90}" SYMS="${SYMS:-BTCUSDT}"
  BUILD_ONCE=0
)
if [[ "${SYMS:-}" == *,* ]]; then
  echo "[train] Detected multiple symbols: ${SYMS}" | tee -a "${LOG}"
  env "${RUN_ENV[@]}" bash scripts/train_multi.sh 2>&1 | tee -a "${LOG}"
else
  env "${RUN_ENV[@]}" bash scripts/train_full.sh 2>&1 | tee -a "${LOG}"
fi

# Package artifacts of interest
mkdir -p "${RUN_DIR}"
for p in models reports export logs; do
  if [ -d "$p" ]; then
    mkdir -p "${RUN_DIR}/$p"
    cp -a "$p"/. "${RUN_DIR}/$p/" || true
  fi
done
cp -f "${LOG}" "${RUN_DIR}/train.log" || true
printf "run_id=%s\nutc_end=%s\n" "${RUN_ID}" "$(date -u --iso-8601=seconds)" > "${RUN_DIR}/metadata.txt"

tar -czf "${TARBALL}" -C /work artifacts

logger -t train "TRAIN_DONE ${RUN_ID}"
echo "=== Compose Train done: ${RUN_ID} (UTC) ===" | tee -a "${LOG}"
