#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: jetson_unpack.sh <bundle.tar.zst> <target_dir_in_remote_root> <expected_sha256>" >&2
  exit 2
fi

BUNDLE="$1"
TARGET="$2"   # models|data|conf
EXPECTED_SHA="$3"

REMOTE_ROOT="${JETSON_REMOTE_ROOT:-/opt/app}"

calc=$(sha256sum "$BUNDLE" | awk '{print $1}')
if [[ "$calc" != "$EXPECTED_SHA" ]]; then
  echo "ERR: sha256 mismatch: $calc != $EXPECTED_SHA" >&2
  exit 1
fi

TS=$(date +%Y%m%d%H%M%S)
VERSION_DIR="$REMOTE_ROOT/$TARGET/V_$TS"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

mkdir -p "$VERSION_DIR"
tar --zstd -xf "$BUNDLE" -C "$VERSION_DIR"

if [[ "$TARGET" == "models" ]]; then
  # Expect model.onnx under VERSION_DIR/models/model.onnx or at root
  NEW_ONNX=""
  if [[ -f "$VERSION_DIR/models/model.onnx" ]]; then NEW_ONNX="$VERSION_DIR/models/model.onnx"; fi
  if [[ -z "$NEW_ONNX" && -f "$VERSION_DIR/model.onnx" ]]; then NEW_ONNX="$VERSION_DIR/model.onnx"; fi
  if [[ -z "$NEW_ONNX" ]]; then echo "ERR: model.onnx not found in bundle" >&2; exit 1; fi
  ln -sfn "$NEW_ONNX" "$REMOTE_ROOT/models/current.onnx"
else
  ln -sfn "$VERSION_DIR" "$REMOTE_ROOT/$TARGET/current"
fi

# Try to reload service if systemd unit exists
if command -v systemctl >/dev/null 2>&1; then
  systemctl reload app 2>/dev/null || systemctl restart app 2>/dev/null || true
fi

# Health check
if command -v curl >/dev/null 2>&1; then
  curl -sf http://127.0.0.1:8080/health || {
    echo "WARN: health endpoint not ready" >&2
  }
fi

echo "Unpack OK: $TARGET -> $VERSION_DIR"

