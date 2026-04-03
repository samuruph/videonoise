#!/usr/bin/env bash
# Full experiment pipeline — runs all steps in order.
#
# Edit scripts/config.sh first, then:
#   conda activate videonoise
#   pip install -e .          # first time only
#   bash scripts/run_all.sh
#
# To run a single step independently:
#   bash scripts/steps/00_download_data.sh
#   bash scripts/steps/01_generate_videos.sh
#   bash scripts/steps/02_compute_metrics.sh
#   bash scripts/steps/03_noise_inversion.sh
#   bash scripts/steps/04_spatiotemporal.sh
#   bash scripts/steps/05_noise_init_ablation.sh
#   bash scripts/steps/06_compare_results.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STEPS=(
    scripts/steps/00_download_data.sh
    scripts/steps/01_generate_videos.sh
    scripts/steps/02_compute_metrics.sh
    scripts/steps/03_noise_inversion.sh
    scripts/steps/04_spatiotemporal.sh
    scripts/steps/06_compare_results.sh
)

TOTAL=${#STEPS[@]}
for i in "${!STEPS[@]}"; do
    STEP="${STEPS[$i]}"
    echo ""
    echo "########################################################"
    echo "  [$((i+1))/$TOTAL] Running $STEP"
    echo "########################################################"
    bash "$STEP"
done

echo ""
echo "========================================================"
echo "  All steps complete."
echo "  Results : results/"
echo "  Notebook: jupyter notebook notebooks/01_analysis.ipynb"
echo "========================================================"
