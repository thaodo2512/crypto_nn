#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
SRC_DIR="${1:-$ROOT_DIR/data/features/5m/BTCUSDT}"
OUT_DIR="$ROOT_DIR/export"
mkdir -p "$OUT_DIR"
BUNDLE="$OUT_DIR/data_bundle.tar.zst"
SHAFILE="$OUT_DIR/data_bundle.sha256"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "ERR: data dir not found: $SRC_DIR" >&2
  exit 1
fi

tar --zstd -cf "$BUNDLE" -C "$SRC_DIR" .
sha256sum "$BUNDLE" | awk '{print $1}' > "$SHAFILE"
echo "$BUNDLE"

