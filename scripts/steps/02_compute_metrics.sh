#!/usr/bin/env bash
# Step 02 — Compute correlation, spectral, and quality metrics.
#
# Reads  : $DATA_REAL  and  $DATA_GEN/<model>_<noise>/
# Writes : results/real/metrics.json
#          results/<model>_<noise>/metrics.json
#
# Run standalone:
#   bash scripts/steps/02_compute_metrics.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"

GEN_KEY="${MODEL}_${NOISE_TYPE}"
GEN_DIR="${DATA_GEN}${GEN_KEY}/"
REAL_OUT="${RESULTS}real/"
GEN_OUT="${RESULTS}${GEN_KEY}/"
mkdir -p "$REAL_OUT" "$GEN_OUT"

echo "================================================================"
echo "  Step 02 — Compute metrics"
echo "  Real      : $DATA_REAL  →  ${REAL_OUT}metrics.json"
echo "  Generated : $GEN_DIR  →  ${GEN_OUT}metrics.json"
echo "================================================================"

echo "  [1/2] Real videos..."
if [ -f "${REAL_OUT}metrics.json" ]; then
    echo "  [skip] ${REAL_OUT}metrics.json already exists"
else
    python -m videonoise.scripts.compute_metrics \
        --input      "$DATA_REAL" \
        --output     "${REAL_OUT}metrics.json" \
        --max_frames "$MAX_FRAMES" \
        --resize     $RESIZE
fi

echo "  [2/2] Generated videos..."
if [ -f "${GEN_OUT}metrics.json" ]; then
    echo "  [skip] ${GEN_OUT}metrics.json already exists"
else
    python -m videonoise.scripts.compute_metrics \
        --input      "$GEN_DIR" \
        --output     "${GEN_OUT}metrics.json" \
        --max_frames "$MAX_FRAMES" \
        --resize     $RESIZE
fi

echo "  Done. Metrics saved to ${RESULTS}<run_key>/metrics.json"
