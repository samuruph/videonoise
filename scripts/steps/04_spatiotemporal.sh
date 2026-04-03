#!/usr/bin/env bash
# Step 04 — Spatio-temporal analysis: PCA modes and 3D power spectrum.
#
# Writes : results/real/spatiotemporal/
#          results/<model>_<noise>/spatiotemporal/
#
# Run standalone:
#   bash scripts/steps/04_spatiotemporal.sh
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
echo "  Step 04 — Spatio-temporal analysis"
echo "  Real      : $DATA_REAL  →  ${REAL_OUT}spatiotemporal/"
echo "  Generated : $GEN_DIR  →  ${GEN_OUT}spatiotemporal/"
echo "================================================================"

echo "  [1/2] Real videos..."
python -m videonoise.scripts.spatiotemporal_analysis \
    --input      "$DATA_REAL" \
    --output     "${REAL_OUT}spatiotemporal/" \
    --max_frames "$MAX_FRAMES" \
    --resize     $RESIZE \
    --pixel_pca

echo "  [2/2] Generated videos..."
python -m videonoise.scripts.spatiotemporal_analysis \
    --input      "$GEN_DIR" \
    --output     "${GEN_OUT}spatiotemporal/" \
    --max_frames "$MAX_FRAMES" \
    --resize     $RESIZE \
    --pixel_pca

echo "  Done. ST plots saved to ${RESULTS}<run_key>/spatiotemporal/"
