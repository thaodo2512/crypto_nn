# Deployment Guide (PC/Cloud + Jetson)

This repo ships two deployments:
- PC/Cloud (P0–P8, P11 ops)
- Jetson Orin Nano (P9–P10 runtime)

## 1) Fill environment
Copy `.env.sample` → `.env` and set:

```
JETSON_HOST=192.168.1.50
JETSON_USER=deploy
JETSON_SSH_KEY=~/.ssh/jetson_deploy
JETSON_REMOTE_ROOT=/opt/app
MODEL_BUNDLE=export/model_bundle.tar.zst
DATA_BUNDLE=export/data_bundle.tar.zst
```

## 2) Build + boot PC stack
```
make build-pc
make up-pc
```

## 3) Train / Export → Package model bundle
Train + export using your preferred workflow (see README). Then package:
```
make package-model
```
This creates `export/model_bundle.tar.zst` and `export/model_bundle.sha256`.

## 4) Deploy model to Jetson
```
make deploy-model
```
This copies the bundle via SSH and runs `scripts/jetson_unpack.sh` remotely to verify sha256, unpack to a versioned folder, and atomically update:
```
/opt/app/models/current.onnx -> /opt/app/models/V_YYYYmmddHHMM/model.onnx
```
The service is reloaded and a health probe is issued.

## 5) Build + boot Jetson edge service
```
make build-jetson
make up-jetson
```
The edge service binds to host network and exposes `:8080` endpoints.

## 6) Health check and logs
```
make jetson-health
make jetson-logs
```
`scripts/healthcheck.sh` probes `/health`, `/score` (dryrun) and `/decide`, then prints p50/p99 if `/var/log/app/latency.log` exists.

## Data bundle (optional)
Create and deploy a data bundle if the edge requires local snapshots:
```
make package-data
make deploy-data
```
This unpacks under `/opt/app/data/V_YYYYmmddHHMM` and atomically updates `/opt/app/data/current` symlink.

## Notes
- On Jetson, the compose file uses `network_mode: host`, GPU env flags, and `ipc: host` for low latency.
- The remote unpack script is idempotent and verifies SHA256 before changing symlinks. Rollback hooks can be added as needed.

