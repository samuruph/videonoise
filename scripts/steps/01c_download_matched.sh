#!/usr/bin/env bash
# Step 01c — Download N matched pairs of (generated, real) videos.
#
# DEFAULT (RECOMMENDED): DeepAction v1
#   faridlab/deepaction_v1  (HuggingFace, CC BY 4.0, no login/GPU needed)
#   Real and generated videos share the same action class and filename,
#   giving exact 1-1 semantic correspondence.
#
# FALLBACK: VBench 2.0 (HF Hub) + Internet Archive
#   Use --source vbench if you need camera-motion / VBench2-style prompts.
#   Match quality is weaker (keyword search on IA).
#
# Output:
#   data/matched_pairs/<source>_<model>/
#     generated/video_000.mp4 …
#     real/video_000.mp4      …
#     pairs.json              ← action_class / prompt, match type
#
# ── Usage examples ──────────────────────────────────────────────────────────
#
#   # 50 pairs with CogVideoX5B outputs (default)
#   bash scripts/steps/01c_download_matched.sh
#
#   # 100 pairs
#   bash scripts/steps/01c_download_matched.sh --n 100
#
#   # Different AI model
#   bash scripts/steps/01c_download_matched.sh --gen_model RunwayML
#   # Available: BDAnimateDiffLightning CogVideoX5B RunwayML StableDiffusion Veo VideoPoet
#
#   # ALL models — downloads generated_<model>/ for each model, same N real videos
#   bash scripts/steps/01c_download_matched.sh --gen_model all
#
#   # VBench2 + Internet Archive fallback (weaker matching)
#   bash scripts/steps/01c_download_matched.sh --source vbench
#
#   # VBench2 filtered to one category
#   bash scripts/steps/01c_download_matched.sh --source vbench \
#       --category_filter Camera_Motion --model_filter CogVideo
#
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"

# Defaults
N="${N_VIDEOS:-50}"
SOURCE="deepaction"         # deepaction | vbench
GEN_MODEL="CogVideoX5B"
MODEL_FILTER=""
CATEGORY_FILTER=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)               N="$2";               shift 2 ;;
        --source)          SOURCE="$2";          shift 2 ;;
        --gen_model)       GEN_MODEL="$2";       shift 2 ;;
        --model_filter)    MODEL_FILTER="$2";    shift 2 ;;
        --category_filter) CATEGORY_FILTER="$2"; shift 2 ;;
        --output)          OUTPUT="$2";          shift 2 ;;
        *) echo "[warn] Unknown arg: $1"; shift ;;
    esac
done

echo "================================================================"
echo "  Step 01c — Download matched (generated, real) video pairs"
echo "  Source     : $SOURCE"
echo "  N pairs    : $N"
echo "================================================================"

if [ "$SOURCE" = "deepaction" ]; then
    echo "  Gen model  : $GEN_MODEL"
    echo "  Dataset    : faridlab/deepaction_v1  (exact action-class matching)"
    echo "================================================================"

    EXTRA_ARGS="--gen_model $GEN_MODEL"
    [ -n "$OUTPUT" ] && EXTRA_ARGS="$EXTRA_ARGS --output $OUTPUT"

    python -m videonoise.scripts.download_data \
        --dataset deepaction_pairs \
        --n       "$N" \
        $EXTRA_ARGS

elif [ "$SOURCE" = "vbench" ]; then
    echo "  Model filter   : ${MODEL_FILTER:-any}"
    echo "  Category filter: ${CATEGORY_FILTER:-balanced across all}"
    echo "  Real backend   : Internet Archive  (keyword search, weaker matching)"
    echo "================================================================"

    EXTRA_ARGS=""
    [ -n "$MODEL_FILTER"    ] && EXTRA_ARGS="$EXTRA_ARGS --model_filter $MODEL_FILTER"
    [ -n "$CATEGORY_FILTER" ] && EXTRA_ARGS="$EXTRA_ARGS --category_filter $CATEGORY_FILTER"
    [ -n "$OUTPUT"          ] && EXTRA_ARGS="$EXTRA_ARGS --output $OUTPUT"

    python -m videonoise.scripts.download_data \
        --dataset matched_pairs \
        --n       "$N" \
        $EXTRA_ARGS
else
    echo "[error] Unknown --source '$SOURCE'. Use: deepaction | vbench"
    exit 1
fi

echo ""
echo "  Done. Pairs saved under data/matched_pairs/"
echo ""
echo "  To analyse these videos, run:"
echo "    bash scripts/steps/02_compute_metrics.sh"
