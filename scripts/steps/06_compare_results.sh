#!/usr/bin/env bash
# Step 06 — Compare real vs generated and print a summary table.
#
# Reads  : results/real/metrics.json
#          results/<model>_<noise>/metrics.json
#          results/real/noise_stats.json          (optional)
#          results/<model>_<noise>/noise_stats.json (optional)
#
# Writes : results/<model>_<noise>/comparison/comparison_overview.png
#
# Run standalone:
#   bash scripts/steps/06_compare_results.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"

GEN_KEY="${MODEL}_${NOISE_TYPE}"
REAL_OUT="${RESULTS}real/"
GEN_OUT="${RESULTS}${GEN_KEY}/"
CMP_OUT="${GEN_OUT}comparison/"
mkdir -p "$CMP_OUT"

echo "================================================================"
echo "  Step 06 — Compare results"
echo "  Real          : ${REAL_OUT}metrics.json"
echo "  Generated     : ${GEN_OUT}metrics.json"
echo "  Output        : ${CMP_OUT}comparison_overview.png"
echo "================================================================"

# Resolve optional noise stats paths
NOISE_REAL_FLAG="";  [ -f "${REAL_OUT}noise_stats.json" ] && NOISE_REAL_FLAG="--noise_real ${REAL_OUT}noise_stats.json"
NOISE_GEN_FLAG="";   [ -f "${GEN_OUT}noise_stats.json"  ] && NOISE_GEN_FLAG="--noise_gen  ${GEN_OUT}noise_stats.json"

python -m videonoise.scripts.compare_results \
    --real       "${REAL_OUT}metrics.json" \
    --generated  "${GEN_OUT}metrics.json" \
    $NOISE_REAL_FLAG \
    $NOISE_GEN_FLAG \
    --output     "$CMP_OUT"

echo "  Done. Figure saved to ${CMP_OUT}comparison_overview.png"
echo ""
echo "  Open the notebook for interactive exploration:"
echo "    jupyter notebook notebooks/01_analysis.ipynb"
