#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "[startup] begin" | tee /dev/ttyS0 || true

# Base tools for keyrings and downloads
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg

# Add Docker official APT repo (Docker CE + Compose v2 plugin)
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

ARCH=$(dpkg --print-architecture)
CODENAME=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
  | tee /etc/apt/sources.list.d/docker.list >/dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Project runtime deps
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip \
  git tmux rsync unzip build-essential

# Start Docker and print versions
systemctl start docker || true
docker --version | tee /dev/ttyS0 || true
docker compose version | tee /dev/ttyS0 || true

# Workspace for artifacts
mkdir -p /work/artifacts
chmod -R 0777 /work || true

echo "READY_FOR_REPO" >/dev/ttyS0 || true
logger -t startup "READY_FOR_REPO"

exit 0
