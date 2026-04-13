#!/usr/bin/env bash
# Full experiment pipeline — runs all steps in order.
#
# Edit scripts/config.yaml first, then:
#   conda activate videonoise
#   pip install -e .          # first time only
#   bash scripts/run_all.sh
#
# To run a single step independently:
#   bash scripts/steps/00_download_data.sh        # download real reference videos
#   bash scripts/steps/01_generate_videos.sh      # generate videos (model + noise)  [GPU]
#   # — OR without a GPU: —
#   bash scripts/steps/01b_download_generated.sh  # download pre-generated AI videos
#   bash scripts/steps/01c_download_matched.sh    # download matched (gen, real) pairs
#   bash scripts/steps/02_compute_metrics.sh      # correlation, spectral, quality metrics
#   bash scripts/steps/03_noise_inversion.sh      # DDIM inversion + noise stats
#   bash scripts/steps/04_spatiotemporal.sh       # PCA + 3D power spectrum
#   bash scripts/steps/05_noise_init_ablation.sh  # sweep all 5 noise types          [GPU]
#   bash scripts/steps/06_compare_results.sh      # summary table + comparison figure
#   bash scripts/steps/07_distribution_metrics.sh # FVD, cross-set LPIPS, CLIP score
#   bash scripts/steps/08_attention_analysis.sh   # attention map extraction          [GPU]
#   bash scripts/steps/09_paired_analysis.sh      # paired Δ analysis + Wilcoxon tests
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
    scripts/steps/05_noise_init_ablation.sh
    scripts/steps/06_compare_results.sh
    scripts/steps/07_distribution_metrics.sh
    scripts/steps/08_attention_analysis.sh
    scripts/steps/09_paired_analysis.sh
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
