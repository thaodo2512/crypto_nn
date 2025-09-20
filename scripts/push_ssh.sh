#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
set -a; source "$ROOT_DIR/.env" 2>/dev/null || true; set +a

TARGET=""
BUNDLE_PATH=""

usage() {
  echo "Usage: $0 --bundle <path.tar.zst> --target <models|data|conf>" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE_PATH="$2"; shift 2;;
    --target) TARGET="$2"; shift 2;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ -z "${JETSON_HOST:-}" || -z "${JETSON_USER:-}" || -z "${JETSON_SSH_KEY:-}" || -z "${JETSON_REMOTE_ROOT:-}" ]]; then
  echo "ERR: missing envs. Fill .env (JETSON_HOST, JETSON_USER, JETSON_SSH_KEY, JETSON_REMOTE_ROOT)." >&2
  exit 1
fi

if [[ -z "$BUNDLE_PATH" || -z "$TARGET" ]]; then usage; exit 2; fi
if [[ ! -f "$BUNDLE_PATH" ]]; then echo "ERR: bundle not found: $BUNDLE_PATH" >&2; exit 1; fi

SHA_FILE="${BUNDLE_PATH%.tar.zst}.sha256"
if [[ ! -f "$SHA_FILE" ]]; then echo "ERR: sha256 file missing: $SHA_FILE" >&2; exit 1; fi

REMOTE_TMP="$JETSON_REMOTE_ROOT/tmp"

mkdir -p "$ROOT_DIR/export"

echo "Pushing bundle → $JETSON_USER@$JETSON_HOST:$REMOTE_TMP/"
ssh -i "$JETSON_SSH_KEY" -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_HOST" "mkdir -p '$REMOTE_TMP' '$JETSON_REMOTE_ROOT/scripts'"

rsync -e "ssh -i $JETSON_SSH_KEY -o StrictHostKeyChecking=no" -av "$BUNDLE_PATH" "$SHA_FILE" "$JETSON_USER@$JETSON_HOST:$REMOTE_TMP/"

# Upload unpack script
scp -i "$JETSON_SSH_KEY" -o StrictHostKeyChecking=no "$ROOT_DIR/scripts/jetson_unpack.sh" "$JETSON_USER@$JETSON_HOST:$JETSON_REMOTE_ROOT/scripts/"

REMOTE_BUNDLE="$REMOTE_TMP/$(basename "$BUNDLE_PATH")"
EXPECTED_SHA=$(cat "$SHA_FILE")

echo "Running remote unpack..."
ssh -i "$JETSON_SSH_KEY" -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_HOST" \
  "bash '$JETSON_REMOTE_ROOT/scripts/jetson_unpack.sh' '$REMOTE_BUNDLE' '$TARGET' '$EXPECTED_SHA'"

echo "Done."

