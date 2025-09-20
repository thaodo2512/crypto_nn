#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
OUT_DIR="$ROOT_DIR/export"
MODEL_ONNX="${1:-$ROOT_DIR/export/model_5m_fp16.onnx}"
ENSEMBLE_JSON="${2:-$ROOT_DIR/models/ensemble_5m.json}"
CONF_BARRIER="${3:-$ROOT_DIR/conf/barrier.yaml}"
CONF_RUNTIME="${4:-$ROOT_DIR/conf/runtime.yaml}"

mkdir -p "$OUT_DIR"
BUNDLE="$OUT_DIR/model_bundle.tar.zst"
SHAFILE="$OUT_DIR/model_bundle.sha256"

for f in "$MODEL_ONNX" "$ENSEMBLE_JSON" "$CONF_RUNTIME"; do
  if [[ ! -f "$f" ]]; then
    echo "ERR: missing file: $f" >&2
    exit 1
  fi
done

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

mkdir -p "$TMPDIR/models" "$TMPDIR/conf"
cp -v "$MODEL_ONNX" "$TMPDIR/models/model.onnx"
cp -v "$ENSEMBLE_JSON" "$TMPDIR/models/ensemble_5m.json" || true
cp -v "$CONF_RUNTIME" "$TMPDIR/conf/runtime.yaml"
if [[ -f "$CONF_BARRIER" ]]; then cp -v "$CONF_BARRIER" "$TMPDIR/conf/barrier.yaml"; fi

tar --zstd -cf "$BUNDLE" -C "$TMPDIR" .
sha256sum "$BUNDLE" | awk '{print $1}' > "$SHAFILE"
echo "$BUNDLE"

