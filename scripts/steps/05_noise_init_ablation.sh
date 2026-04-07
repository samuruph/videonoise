#!/usr/bin/env bash
# Step 05 — Noise initialization ablation.
#
# Generates videos for every noise type using the model in config.sh,
# then computes metrics for each condition so they can be compared.
#
# Writes :  data/generated/<model>_<noise>/      (one folder per noise type)
#           results/<model>_<noise>/metrics.json  (one folder per run key)
#
# Run standalone:
#   bash scripts/steps/05_noise_init_ablation.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"

NOISE_TYPES=("gaussian" "ar1" "spatial_lowpass" "blue" "perlin")

echo "================================================================"
echo "  Step 05 — Noise init ablation"
echo "  Model     : $MODEL"
echo "  Noises    : ${NOISE_TYPES[*]}"
echo "================================================================"

MODEL_ID_FLAG=""
[ -n "${MODEL_ID:-}" ] && MODEL_ID_FLAG="--model_id $MODEL_ID"

for NOISE in "${NOISE_TYPES[@]}"; do
    RUN_KEY="${MODEL}_${NOISE}__${SETTINGS_SUFFIX}"
    RUN_OUT="${RESULTS}${RUN_KEY}/"
    mkdir -p "$RUN_OUT"
    echo ""
    echo "  --- $RUN_KEY ---"

    echo "  Generating videos → data/generated/${RUN_KEY}/"
    python -m videonoise.scripts.generate_videos \
        --model       "$MODEL" \
        $MODEL_ID_FLAG \
        --prompt      "$ACTIVE_PROMPT" \
        --noise_type  "$NOISE" \
        --alpha       "$ALPHA" \
        --sigma       "$SIGMA" \
        --n           "$N_VIDEOS" \
        --num_frames  "$NUM_FRAMES" \
        --num_steps   "$NUM_STEPS" \
        --seed        "$SEED" \
        --output      "$DATA_GEN"

    echo "  Computing metrics → ${RUN_OUT}metrics.json"
    python -m videonoise.scripts.compute_metrics \
        --input      "${DATA_GEN}${RUN_KEY}/" \
        --output     "${RUN_OUT}metrics.json" \
        --max_frames "$MAX_FRAMES" \
        --resize     $RESIZE
done

# Visualise all noise types side-by-side (shapes only, no model needed)
echo ""
echo "  Visualising noise shapes → ${RESULTS}noise_init/"
python -m videonoise.scripts.noise_init \
    --type  all \
    --shape 16 4 32 32 \
    --alpha "$ALPHA" \
    --output "${RESULTS}noise_init/"

echo ""
echo "  Done. Each run saved to ${RESULTS}<model>_<noise>/"
