#!/usr/bin/env bash
set -euxo pipefail

LOG=~/train.log
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
ART_ROOT="/work/artifacts"
RUN_DIR="${ART_ROOT}/run-${RUN_ID}"
TARBALL="/work/artifacts-${RUN_ID}.tgz"

{
  echo "=== Train start: ${RUN_ID} (UTC) ==="
  echo "Host: $(hostname)"
  echo "Kernel: $(uname -a)"
} | tee -a "${LOG}"

cd ~/repo

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip wheel
if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
fi

echo "[train] Simulating workload (10s)..." | tee -a "${LOG}"
sleep 10
python - <<'PY' 2>&1 | tee -a "$LOG"
import time
print("[train] Writing sample metrics ...", flush=True)
for i in range(5):
    print(f"[train] step={i} loss={1.0/(i+1):.4f}", flush=True)
    time.sleep(1)
print("[train] Done.", flush=True)
PY

mkdir -p "${RUN_DIR}"
cp -f "${LOG}" "${RUN_DIR}/train.log" || true
printf "run_id=%s\nutc_start=%s\n" "${RUN_ID}" "$(date -u --iso-8601=seconds)" > "${RUN_DIR}/metadata.txt"

tar -czf "${TARBALL}" -C /work artifacts

logger -t train "TRAIN_DONE ${RUN_ID}"
echo "=== Train done: ${RUN_ID} (UTC) ===" | tee -a "${LOG}"

