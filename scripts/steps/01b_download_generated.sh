#!/usr/bin/env bash
# Step 01b — Download pre-generated AI videos (alternative to step 01).
#
# Use this instead of step 01 when you do not have a GPU or do not want
# to wait for generation.  Downloads ~90 videos (matching DAVIS size) and
# places them in data/generated/<run_key>/ so all downstream steps work
# identically to generated videos.
#
# ── Backends ────────────────────────────────────────────────────────────────
#
#   pexels        Short clips from Pexels (free API key required).
#                 Best quality; clips are visually diverse.
#                 Get a free key in 30 s at: https://www.pexels.com/api/
#
#   hf_generated  AI-generated videos from HuggingFace Hub (no key needed).
#                 Downloads outputs from VBench evaluation sets:
#                   vbench_modelscope  (default)
#                   vbench_lavie
#                   vbench_cogvideo
#                   vbench_videocrafter
#
# ── Usage examples ──────────────────────────────────────────────────────────
#
#   # HuggingFace — ModelScope outputs (no key, default)
#   bash scripts/steps/01b_download_generated.sh
#
#   # HuggingFace — VideoCrafter2 outputs
#   bash scripts/steps/01b_download_generated.sh hf_generated vbench_videocrafter
#
#   # Pexels — nature clips (free key required)
#   PEXELS_KEY=your_key bash scripts/steps/01b_download_generated.sh pexels "nature landscape"
#
#   # Pexels — people walking
#   PEXELS_KEY=your_key bash scripts/steps/01b_download_generated.sh pexels "people walking"
#
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"

BACKEND="${1:-hf_generated}"     # hf_generated | pexels
OPTION="${2:-vbench_modelscope}" # hf_source key  OR  pexels query string

echo "================================================================"
echo "  Step 01b — Download pre-generated AI videos"
echo "  Backend : $BACKEND"
echo "  Option  : $OPTION"
echo "  N       : $N_VIDEOS  videos"
echo "================================================================"

if [ "$BACKEND" = "pexels" ]; then
    # ── Pexels ────────────────────────────────────────────────────────────
    KEY="${PEXELS_KEY:-${3:-}}"
    if [ -z "$KEY" ]; then
        echo ""
        echo "  [error] Pexels API key required."
        echo "  Get a free key at https://www.pexels.com/api/ then run:"
        echo "    PEXELS_KEY=your_key bash scripts/steps/01b_download_generated.sh pexels \"$OPTION\""
        exit 1
    fi
    SLUG=$(echo "$OPTION" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | cut -c1-20)
    OUT_DIR="${DATA_GEN}pexels_${SLUG}/"
    echo "  Output  : $OUT_DIR"
    python -m videonoise.scripts.download_data \
        --dataset  pexels \
        --api_key  "$KEY" \
        --query    "$OPTION" \
        --n        "$N_VIDEOS" \
        --output   "$OUT_DIR"

elif [ "$BACKEND" = "hf_generated" ]; then
    # ── HuggingFace Hub ───────────────────────────────────────────────────
    HF_SOURCE="$OPTION"
    # Derive a tidy run key from the hf_source name
    # e.g. vbench_modelscope → hf_modelscope
    RUN_KEY="hf_$(echo "$HF_SOURCE" | sed 's/vbench_//')"
    OUT_DIR="${DATA_GEN}${RUN_KEY}/"
    echo "  HF source : $HF_SOURCE"
    echo "  Output    : $OUT_DIR"
    python -m videonoise.scripts.download_data \
        --dataset    hf_generated \
        --hf_source  "$HF_SOURCE" \
        --n          "$N_VIDEOS" \
        --output     "$OUT_DIR"

else
    echo "[error] Unknown backend '$BACKEND'. Use: hf_generated | pexels"
    exit 1
fi

echo ""
echo "  Done."
echo "  Videos in : $OUT_DIR"
echo ""
echo "  You can now run step 02 directly — point it at this folder:"
echo "    MODEL=hf MODEL_ID= NOISE_TYPE=generated bash scripts/steps/02_compute_metrics.sh"
echo ""
echo "  Or set in config.yaml:"
echo "    model: hf"
echo "    noise_type: $(basename "$OUT_DIR" | sed 's/hf_//')"
