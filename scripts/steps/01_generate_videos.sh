#!/usr/bin/env bash
# Step 01 — Generate videos with a diffusion model.
#
# Edit MODEL and NOISE_TYPE in scripts/config.sh, then run:
#   bash scripts/steps/01_generate_videos.sh
#
# Examples:
#   MODEL=svd        NOISE_TYPE=gaussian   (baseline)
#   MODEL=modelscope NOISE_TYPE=ar1        (temporal correlation)
#   MODEL=cogvideox  NOISE_TYPE=perlin
#   MODEL=wan        NOISE_TYPE=blue
#
# Generated videos land in  $DATA_GEN/<noise_type>/
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"

# The Python CLI auto-creates  <DATA_GEN>/<MODEL>_<NOISE_TYPE>/
# so we just pass the root data/generated/ directory here.
echo "================================================================"
echo "  Step 01 — Generate videos"
echo "  Model     : $MODEL"
echo "  Noise     : $NOISE_TYPE"
echo "  Videos    : $N_VIDEOS  x  $NUM_FRAMES frames  ($NUM_STEPS steps)"
echo "  Output    : ${DATA_GEN}${MODEL}_${NOISE_TYPE}/"
echo "================================================================"

MODEL_ID_FLAG="";  [ -n "${MODEL_ID:-}" ]     && MODEL_ID_FLAG="--model_id $MODEL_ID"
T2I_FLAG="";       [ -n "${T2I_MODEL_ID:-}" ] && T2I_FLAG="--t2i_model_id $T2I_MODEL_ID"

python -m videonoise.scripts.generate_videos \
    --model       "$MODEL" \
    $MODEL_ID_FLAG \
    $T2I_FLAG \
    --prompt      "$ACTIVE_PROMPT" \
    --noise_type  "$NOISE_TYPE" \
    --alpha       "$ALPHA" \
    --sigma       "$SIGMA" \
    --n           "$N_VIDEOS" \
    --num_frames  "$NUM_FRAMES" \
    --num_steps   "$NUM_STEPS" \
    --seed        "$SEED" \
    --output      "$DATA_GEN"

echo "  Done. Videos saved to ${DATA_GEN}${MODEL}_${NOISE_TYPE}/"
