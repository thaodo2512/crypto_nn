#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-127.0.0.1}"
PORT="${2:-8080}"

echo "Checking /health on http://$HOST:$PORT/health"
curl -sf "http://$HOST:$PORT/health" || echo "WARN: /health not OK"

echo "Checking /score dryrun"
curl -sf -X POST -H 'Content-Type: application/json' \
  "http://$HOST:$PORT/score" \
  -d '{"dryrun": true, "window": []}' || echo "WARN: /score dryrun failed"

echo "Checking /decide sample"
curl -sf -X POST -H 'Content-Type: application/json' \
  "http://$HOST:$PORT/decide" \
  -d '{"dryrun": true, "window": []}' || echo "WARN: /decide failed"

LOG="/var/log/app/latency.log"
if [[ -f "$LOG" ]]; then
  echo "Latency (from $LOG):"
  awk '{print $NF}' "$LOG" | sort -n | awk 'NR==int(NR*0.5){p50=$1} END{printf("p50=%sms p99=%sms\n", p50+0, $1+0)}'
else
  echo "No latency log found at $LOG"
fi

