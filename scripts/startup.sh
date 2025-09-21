#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip \
  git tmux rsync unzip curl \
  build-essential docker.io || true

# Try to install compose via apt plugin first; fall back to GitHub binary if unavailable
if ! apt-get install -y --no-install-recommends docker-compose-plugin; then
  echo "[startup] docker-compose-plugin not in repo; installing Compose v2 binary" | tee /dev/ttyS0 || true
  COMPOSE_VER="v2.27.0"
  mkdir -p /usr/local/lib/docker/cli-plugins
  ARCH=$(uname -m)
  case "$ARCH" in
    x86_64|amd64) SUFFIX=linux-x86_64 ;;
    aarch64|arm64) SUFFIX=linux-aarch64 ;;
    *) SUFFIX=linux-x86_64 ;;
  esac
  curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
    "https://github.com/docker/compose/releases/download/${COMPOSE_VER}/docker-compose-${SUFFIX}"
  chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

# Start Docker if present
systemctl start docker || true
docker --version || true
docker compose version || true

mkdir -p /work/artifacts
chmod -R 0777 /work || true

echo "READY_FOR_REPO" >/dev/ttyS0 || true
logger -t startup "READY_FOR_REPO"

exit 0
