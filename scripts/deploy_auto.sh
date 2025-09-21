#!/usr/bin/env bash
set -euo pipefail

# Auto-pack and push inference artifacts (model + optional features data) to Jetson
# Requires .env with JETSON_HOST, JETSON_USER, JETSON_SSH_KEY, JETSON_REMOTE_ROOT

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

set -a; source .env 2>/dev/null || true; set +a

if [[ -z "${JETSON_HOST:-}" || -z "${JETSON_USER:-}" || -z "${JETSON_SSH_KEY:-}" || -z "${JETSON_REMOTE_ROOT:-}" ]]; then
  echo "[deploy] Jetson env not configured; skip (set JETSON_HOST/USER/SSH_KEY/REMOTE_ROOT in .env)." >&2
  exit 0
fi

SYM=${SYM:-$(echo "${SYMS:-BTCUSDT}" | cut -d',' -f1)}
TF=${TF:-5m}

# Resolve model + ensemble paths (symbol-specific preferred)
MODEL_ONNX="export/model_${TF}_${SYM}_fp16.onnx"
[[ -f "$MODEL_ONNX" ]] || MODEL_ONNX="export/model_5m_fp16.onnx"
ENSEMBLE_JSON="models/ensemble_5m_${SYM}.json"
[[ -f "$ENSEMBLE_JSON" ]] || ENSEMBLE_JSON="models/ensemble_5m.json"

echo "[deploy] Packaging model for $SYM @ $TF using: $MODEL_ONNX and $ENSEMBLE_JSON"
MODEL_BUNDLE=$(bash scripts/pack_model.sh "$MODEL_ONNX" "$ENSEMBLE_JSON" conf/barrier.yaml conf/runtime.yaml)
echo "[deploy] Model bundle: $MODEL_BUNDLE"

echo "[deploy] Uploading model bundle to Jetson..."
bash scripts/push_ssh.sh --bundle "$MODEL_BUNDLE" --target models

# Optional: package recent features for on-edge inspection (can be large)
DATA_SRC="data/features/${TF}/${SYM}"
if [[ -d "$DATA_SRC" ]]; then
  echo "[deploy] Packaging features data from $DATA_SRC"
  DATA_BUNDLE=$(bash scripts/pack_data.sh "$DATA_SRC")
  echo "[deploy] Data bundle: $DATA_BUNDLE"
  echo "[deploy] Uploading data bundle to Jetson..."
  bash scripts/push_ssh.sh --bundle "$DATA_BUNDLE" --target data || echo "[deploy] Data upload optional; continuing"
else
  echo "[deploy] Features dir not found ($DATA_SRC); skipping data bundle."
fi

echo "[deploy] Done."

