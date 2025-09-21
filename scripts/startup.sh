#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip \
  git tmux rsync unzip \
  build-essential docker.io docker-compose-plugin

# Start Docker if present
systemctl start docker || true

mkdir -p /work/artifacts
chmod -R 0777 /work || true

echo "READY_FOR_REPO" >/dev/ttyS0 || true
logger -t startup "READY_FOR_REPO"

exit 0
