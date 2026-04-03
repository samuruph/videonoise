#!/usr/bin/env bash
# Full experiment pipeline.
# Edit the variables below, then run:
#
#   conda activate videonoise
#   pip install -e .          # first time only
#   cd videonoise
#   bash scripts/run_experiments.sh

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

DATA_REAL="data/real/"
DATA_GEN="data/generated/"
RESULTS="results/"
MAX_FRAMES=32
RESIZE="128 128"          # W H

# -----------------------------------------------------------------------
# 1. Test data  (replace with real datasets / vn-generate when ready)
# -----------------------------------------------------------------------
echo "=== [1/6] Generating synthetic test videos ==="
python -m videonoise.scripts.download_data \
    --dataset synthetic --output "$DATA_REAL" --n_synthetic 8

echo "=== [2/6] Generating synthetic 'generated' placeholders ==="
python -m videonoise.scripts.download_data \
    --dataset synthetic --output "$DATA_GEN" --n_synthetic 8

# To download DAVIS:
#   python -m videonoise.scripts.download_data --dataset davis --output "$DATA_REAL"
#
# To generate with SVD (requires ~7 GB download):
#   python -m videonoise.scripts.generate_videos \
#       --noise_type gaussian --n 8 --output "$DATA_GEN/gaussian/"
#   python -m videonoise.scripts.generate_videos \
#       --noise_type ar1 --alpha 0.8 --n 8 --output "$DATA_GEN/ar1/"

# -----------------------------------------------------------------------
# 2. Metrics
# -----------------------------------------------------------------------
echo "=== [3/6] Metrics — real ==="
python -m videonoise.scripts.compute_metrics \
    --input "$DATA_REAL" \
    --output "${RESULTS}metrics/real.json" \
    --max_frames $MAX_FRAMES --resize $RESIZE

echo "=== [4/6] Metrics — generated ==="
python -m videonoise.scripts.compute_metrics \
    --input "$DATA_GEN" \
    --output "${RESULTS}metrics/generated.json" \
    --max_frames $MAX_FRAMES --resize $RESIZE

# -----------------------------------------------------------------------
# 3. Noise inversion
# -----------------------------------------------------------------------
echo "=== [5/6] Noise inversion ==="
python -m videonoise.scripts.noise_inversion \
    --input "$DATA_REAL" \
    --output "${RESULTS}noise_stats/real.json" \
    --max_frames $MAX_FRAMES --resize $RESIZE

python -m videonoise.scripts.noise_inversion \
    --input "$DATA_GEN" \
    --output "${RESULTS}noise_stats/generated.json" \
    --max_frames $MAX_FRAMES --resize $RESIZE

# -----------------------------------------------------------------------
# 4. Spatio-temporal analysis + noise init ablation
# -----------------------------------------------------------------------
echo "=== [6/6] Spatio-temporal + noise init ==="
python -m videonoise.scripts.spatiotemporal_analysis \
    --input "$DATA_REAL" \
    --output "${RESULTS}plots/real_st/" \
    --max_frames $MAX_FRAMES --resize $RESIZE --pixel_pca

python -m videonoise.scripts.spatiotemporal_analysis \
    --input "$DATA_GEN" \
    --output "${RESULTS}plots/gen_st/" \
    --max_frames $MAX_FRAMES --resize $RESIZE --pixel_pca

python -m videonoise.scripts.noise_init \
    --type all --shape 16 4 32 32 --alpha 0.8 \
    --output "${RESULTS}noise_init/"

# -----------------------------------------------------------------------
# 5. Summary comparison
# -----------------------------------------------------------------------
python -m videonoise.scripts.compare_results \
    --real     "${RESULTS}metrics/real.json" \
    --generated "${RESULTS}metrics/generated.json" \
    --noise_real "${RESULTS}noise_stats/real.json" \
    --noise_gen  "${RESULTS}noise_stats/generated.json" \
    --output "${RESULTS}plots/comparison/"

echo ""
echo "Done. Open notebooks/01_analysis.ipynb for interactive exploration."
