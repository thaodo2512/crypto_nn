#!/usr/bin/env bash
set -euo pipefail

# Launch per-symbol training in parallel tmux sessions on the VM.
# Usage (env): SYMS="BTCUSDT,ETHUSDT" QUICK=0 DAYS=80 TF=5m WINDOW=144 H=36 bash scripts/train_parallel.sh

cd ~/repo

# Build once to avoid duplicated work across sessions
docker compose -f docker-compose.train.yml --profile train build

IFS=',' read -r -a SYM_ARR <<< "${SYMS:-BTCUSDT}"
for sym in "${SYM_ARR[@]}"; do
  sym_trim=$(echo "$sym" | xargs)
  [[ -z "$sym_trim" ]] && continue
  sess="train_${sym_trim}"
  log="$HOME/train_${sym_trim}.log"
  echo "[parallel] starting session $sess for $sym_trim → $log"
  # Run each session with docker group privileges via sg
  sg docker -c "tmux new -d -s $sess \"SYMS=$sym_trim QUICK=${QUICK:-1} DAYS=${DAYS:-90} TF=${TF:-5m} WINDOW=${WINDOW:-144} H=${H:-36} TRAIN_LOG=$log RUN_SUFFIX=$sym_trim bash ~/repo/scripts/train_compose.sh\""
done

echo "[parallel] tmux sessions:"
tmux ls || true

