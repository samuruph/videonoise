#!/usr/bin/env bash
# Step 03 — Run noise inversion and analyse the recovered latents.
#
# Inverts both real and generated videos using a pixel-domain DDPM schedule.
# Outputs statistics: mean, std, skewness, kurtosis, KL-div, KS test, power spectrum.
#
# Writes : results/real/noise_stats.json
#          results/<model>_<noise>/noise_stats.json
#
# Run standalone:
#   bash scripts/steps/03_noise_inversion.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"

if [ -n "${GEN_DIR_OVERRIDE:-}" ]; then
    GEN_DIR="${GEN_DIR_OVERRIDE%/}/"
    _GEN_BASE=$(basename "${GEN_DIR_OVERRIDE%/}")
else
    _GEN_BASE="${MODEL}_${NOISE_TYPE}"
    GEN_DIR="${DATA_GEN}${_GEN_BASE}/"
fi
GEN_KEY="${_GEN_BASE}__${SETTINGS_SUFFIX}"
REAL_OUT="${RESULTS}${REAL_KEY}/"
GEN_OUT="${RESULTS}${GEN_KEY}/"
mkdir -p "$REAL_OUT" "$GEN_OUT"

echo "================================================================"
echo "  Step 03 — Noise inversion"
echo "  Real      : $DATA_REAL  →  ${REAL_OUT}noise_stats.json"
echo "  Generated : $GEN_DIR  →  ${GEN_OUT}noise_stats.json"
echo "================================================================"

echo "  [1/2] Inverting real videos..."
python -m videonoise.scripts.noise_inversion \
    --input      "$DATA_REAL" \
    --output     "${REAL_OUT}noise_stats.json" \
    --max_frames "$MAX_FRAMES" \
    --resize     $RESIZE

echo "  [2/2] Inverting generated videos..."
python -m videonoise.scripts.noise_inversion \
    --input      "$GEN_DIR" \
    --output     "${GEN_OUT}noise_stats.json" \
    --max_frames "$MAX_FRAMES" \
    --resize     $RESIZE

echo "  Done. Noise stats saved to ${RESULTS}<run_key>/noise_stats.json"
