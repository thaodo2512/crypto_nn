#!/usr/bin/env bash
set -euo pipefail

# Run P1→P8 on a cloud VM for a given symbol, then auto‑deploy to Jetson via SSH.
# Requires secrets/.env with Jetson settings and AUTO_DEPLOY=1.
#
# Usage:
#   bash scripts/cloud_train_and_deploy.sh BTCUSDT \
#     --quick-days 60 --val-bars 288 [--days 90]
#
# From your laptop via gcloud:
#   gcloud compute ssh <vm-name> --zone <zone> -- \
#     "cd /path/to/crypto_nn && git pull && bash scripts/cloud_train_and_deploy.sh BTCUSDT --quick-days 60 --val-bars 288"

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

SYM=""
QUICK_DAYS=60
VAL_BARS=288
DAYS=""

usage() {
  cat <<EOF >&2
cloud_train_and_deploy.sh <SYMBOL> [--quick-days N] [--val-bars N] [--days N]

Runs scripts/train_full.sh with QUICK=1, QUICK_DAYS, VAL_BARS and SYMS=<SYMBOL>.
Auto‑deploys to Jetson if secrets/.env has AUTO_DEPLOY=1 and Jetson vars.
EOF
}

if [[ $# -lt 1 ]]; then usage; exit 2; fi
SYM="$1"; shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick-days) QUICK_DAYS="$2"; shift 2;;
    --val-bars)   VAL_BARS="$2"; shift 2;;
    --days)       DAYS="$2"; shift 2;;
    -h|--help)    usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

echo "[cloud] SYM=$SYM QUICK_DAYS=$QUICK_DAYS VAL_BARS=$VAL_BARS DAYS=${DAYS:-<auto>}"

# Build once (idempotent)
docker compose -f docker-compose.train.yml --profile train build

# Run training+export; AUTO_DEPLOY=1 in secrets/.env triggers push to Jetson after P8
if [[ -n "$DAYS" ]]; then
  SYMS="$SYM" QUICK=1 QUICK_DAYS="$QUICK_DAYS" VAL_BARS="$VAL_BARS" DAYS="$DAYS" bash scripts/train_full.sh
else
  SYMS="$SYM" QUICK=1 QUICK_DAYS="$QUICK_DAYS" VAL_BARS="$VAL_BARS" bash scripts/train_full.sh
fi

echo "[cloud] Done. Artifacts in models/, export/, reports/."

