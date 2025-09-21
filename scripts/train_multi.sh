#!/usr/bin/env bash
set -euo pipefail

# Usage: SYMS="BTCUSDT,ETHUSDT" TF=5m WINDOW=144 H=36 DAYS=90 bash scripts/train_multi.sh

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

IFS=',' read -r -a SYM_ARR <<< "${SYMS:-BTCUSDT}"
for sym in "${SYM_ARR[@]}"; do
  sym_trim=$(echo "$sym" | xargs)
  if [[ -z "$sym_trim" ]]; then continue; fi
  echo "[multi] ===== Training for symbol: $sym_trim ====="
  SYMS="$sym_trim" TF="${TF:-5m}" WINDOW="${WINDOW:-144}" H="${H:-36}" DAYS="${DAYS:-90}" \
    bash scripts/train_full.sh
done

echo "[multi] All symbols completed."

