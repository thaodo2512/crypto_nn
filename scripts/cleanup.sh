#!/usr/bin/env bash
set -euo pipefail

# Cleanup helper for storage: remove regenerable artifacts and prune old partitions.
#
# Usage (dry run by default):
#   SYM=BTCUSDT TF=5m KEEP_DAYS=60 bash scripts/cleanup.sh --purge-artifacts         # delete artifacts, masks, smote, logs, temp reports
#   SYM=BTCUSDT TF=5m KEEP_DAYS=60 bash scripts/cleanup.sh --prune-features --yes     # prune features/labels older than KEEP_DAYS
#   SYM=BTCUSDT TF=5m KEEP_DAYS=90 bash scripts/cleanup.sh --prune-raw --yes          # prune raw lake older than KEEP_DAYS
#   SYM=BTCUSDT               bash scripts/cleanup.sh --keep-best-model --yes         # keep only best fold checkpoint for SYM
#
# Env vars:
#   SYM        - symbol (default: first of SYMS or BTCUSDT)
#   SYMS       - comma list; first entry used if SYM not set
#   TF         - timeframe (default: 5m)
#   KEEP_DAYS  - days to keep when pruning (default: 60)
#   YES=1 or --yes to actually delete; otherwise dry-run prints actions only

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

SYM=${SYM:-$(echo "${SYMS:-BTCUSDT}" | cut -d',' -f1)}
TF=${TF:-5m}
KEEP_DAYS=${KEEP_DAYS:-60}
YES=${YES:-0}

PURGE_ARTIFACTS=0
PRUNE_FEATURES=0
PRUNE_RAW=0
KEEP_BEST_MODEL=0

usage() {
  cat <<EOF >&2
Cleanup storage safely (dry-run by default)

Flags:
  --purge-artifacts    Remove regenerable artifacts (artifacts/, masks, SMOTE, logs, temp reports)
  --prune-features     Prune features/labels older than KEEP_DAYS
  --prune-raw          Prune raw lake older than KEEP_DAYS
  --keep-best-model    Keep only best fold checkpoint for SYM
  --yes                Execute deletions (otherwise dry-run)

Env:
  SYM=${SYM}  TF=${TF}  KEEP_DAYS=${KEEP_DAYS}

Examples:
  SYM=BTCUSDT TF=5m KEEP_DAYS=60 bash scripts/cleanup.sh --purge-artifacts --yes
  SYM=BTCUSDT TF=5m KEEP_DAYS=60 bash scripts/cleanup.sh --prune-features --prune-raw --yes
  SYM=BTCUSDT bash scripts/cleanup.sh --keep-best-model --yes
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --purge-artifacts) PURGE_ARTIFACTS=1; shift;;
    --prune-features)  PRUNE_FEATURES=1; shift;;
    --prune-raw)       PRUNE_RAW=1; shift;;
    --keep-best-model) KEEP_BEST_MODEL=1; shift;;
    --yes)             YES=1; shift;;
    -h|--help)         usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

say() { echo "[cleanup] $*"; }
do_rm() {
  local path="$1"
  if [[ "$YES" == "1" ]]; then rm -rf "$path"; else say "DRY-RUN rm -rf $path"; fi
}

du_list() {
  say "Disk usage (top-level):"
  du -h -d 1 data 2>/dev/null | sort -h || true
  du -h -d 1 artifacts 2>/dev/null | sort -h || true
  du -h -d 1 models 2>/dev/null | sort -h || true
  du -h -d 1 export 2>/dev/null | sort -h || true
}

prune_partitions() {
  local root="$1"   # e.g., data/features/5m/BTCUSDT
  local keep_days="$2"
  [[ -d "$root" ]] || return 0
  local cutoff
  cutoff=$(date -u -d "-${keep_days} days" +%Y-%m-%d 2>/dev/null || date -v-"${keep_days}"d +%Y-%m-%d)
  find "$root" -type d -name 'd=*' | while read -r d; do
    local y m dd ds
    y=$(echo "$d" | sed -n "s|.*y=\([0-9]\{4\}\).*|\1|p")
    m=$(echo "$d" | sed -n "s|.*m=\([0-9]\{2\}\).*|\1|p")
    dd=$(basename "$d" | cut -d= -f2)
    [[ -n "$y$m$dd" ]] || continue
    ds="$y-$m-$dd"
    if [[ "$ds" < "$cutoff" ]]; then
      do_rm "$d"
    fi
  done
}

say "SYM=$SYM TF=$TF KEEP_DAYS=$KEEP_DAYS YES=$YES"
du_list

if [[ "$PURGE_ARTIFACTS" == "1" ]]; then
  say "Purging regenerable artifacts (dry-run unless --yes)"
  do_rm "artifacts/p5_oos_probs/$SYM"
  do_rm "data/aug/train_smote/$SYM"
  do_rm "data/masks/$TF/$SYM"
  do_rm "logs/*"
  # Temp reports that can be recreated
  do_rm "reports/p2_bench_*.csv"
  do_rm "reports/p4_classmix*.json"
  do_rm "reports/p6_curves.png"
  do_rm "reports/p6_oos_summary*.json"
  do_rm "reports/p8_parity.json"
fi

if [[ "$PRUNE_FEATURES" == "1" ]]; then
  say "Pruning features/labels older than ${KEEP_DAYS} days (dry-run unless --yes)"
  prune_partitions "data/features/$TF/$SYM" "$KEEP_DAYS"
  prune_partitions "data/labels/$TF/$SYM" "$KEEP_DAYS"
fi

if [[ "$PRUNE_RAW" == "1" ]]; then
  say "Pruning raw lake older than ${KEEP_DAYS} days (dry-run unless --yes)"
  prune_partitions "data/parquet/$TF/$SYM" "$KEEP_DAYS"
fi

if [[ "$KEEP_BEST_MODEL" == "1" ]]; then
  say "Keeping only best-weight fold checkpoint for $SYM (dry-run unless --yes)"
  python - <<'PY'
import json, os, shutil, glob, sys
sym=os.environ.get('SYM','BTCUSDT')
ens=f"models/ensemble_5m_{sym}.json"
try:
    d=json.load(open(ens))
    w=d.get('weights',{})
    best=str(max(w, key=lambda k: float(w[k])) if w else '0')
except Exception:
    print("[cleanup] WARN: ensemble JSON missing or unreadable; keeping all folds")
    sys.exit(0)
for p in glob.glob(f"models/gru_5m/{sym}/*"):
    fold=os.path.basename(p)
    if fold!=best:
        if os.environ.get('YES','0')=='1':
            shutil.rmtree(p, ignore_errors=True)
            print(f"[cleanup] removed {p}")
        else:
            print(f"[cleanup] DRY-RUN remove {p}")
print(f"[cleanup] kept fold {best}")
PY
fi

du_list
say "Done."

