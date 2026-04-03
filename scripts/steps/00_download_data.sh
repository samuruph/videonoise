#!/usr/bin/env bash
# Step 00 — Download or generate real reference videos.
#
# Usage:
#   bash scripts/steps/00_download_data.sh                  # synthetic (fast)
#   bash scripts/steps/00_download_data.sh --dataset davis  # DAVIS 2017 (~2 GB, ~90 sequences)
#
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"

DATASET="${1:-synthetic}"   # synthetic | davis

echo "================================================================"
echo "  Step 00 — Download real reference videos"
echo "  Dataset : $DATASET"
echo "  Output  : $DATA_REAL"
echo "================================================================"

python -m videonoise.scripts.download_data \
    --dataset "$DATASET" \
    --output  "$DATA_REAL" \
    --n       "$N_VIDEOS"

echo "  Done. Videos saved to $DATA_REAL"
