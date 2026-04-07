#!/usr/bin/env bash
# Step 01c — Download N matched pairs of (generated, real) videos.
#
# Each pair shares the same semantic prompt:
#   Generated : VBench 2.0 (Vchitect/VBench-2.0_sampled_videos, HF Hub, no key)
#   Real      : Internet Archive search with extracted prompt keywords (no key)
#
# Output:
#   data/matched_pairs/<model>_<category>/
#     generated/video_000.mp4 …
#     real/video_000.mp4      …
#     pairs.json              ← cross-reference: prompt, category, IA query
#
# ── Usage examples ──────────────────────────────────────────────────────────
#
#   # 50 pairs, balanced across all VBench2 categories (RECOMMENDED)
#   bash scripts/steps/01c_download_matched.sh
#
#   # 100 pairs
#   bash scripts/steps/01c_download_matched.sh --n 100
#
#   # Only Camera_Motion category, CogVideo model
#   bash scripts/steps/01c_download_matched.sh \
#       --category_filter Camera_Motion --model_filter CogVideo
#
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"

# Parse optional overrides passed as --key value pairs
N="${N_VIDEOS:-50}"
MODEL_FILTER=""
CATEGORY_FILTER=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)             N="$2";               shift 2 ;;
        --model_filter)  MODEL_FILTER="$2";    shift 2 ;;
        --category_filter) CATEGORY_FILTER="$2"; shift 2 ;;
        --output)        OUTPUT="$2";          shift 2 ;;
        *) echo "[warn] Unknown arg: $1"; shift ;;
    esac
done

echo "================================================================"
echo "  Step 01c — Download matched (generated, real) video pairs"
echo "  N pairs    : $N"
echo "  Model      : ${MODEL_FILTER:-any}"
echo "  Category   : ${CATEGORY_FILTER:-balanced across all}"
echo "================================================================"

EXTRA_ARGS=""
[ -n "$MODEL_FILTER"    ] && EXTRA_ARGS="$EXTRA_ARGS --model_filter $MODEL_FILTER"
[ -n "$CATEGORY_FILTER" ] && EXTRA_ARGS="$EXTRA_ARGS --category_filter $CATEGORY_FILTER"
[ -n "$OUTPUT"          ] && EXTRA_ARGS="$EXTRA_ARGS --output $OUTPUT"

python -m videonoise.scripts.download_data \
    --dataset matched_pairs \
    --n       "$N" \
    $EXTRA_ARGS

echo ""
echo "  Done. Pairs saved under data/matched_pairs/"
echo ""
echo "  To analyse these videos, run:"
echo "    bash scripts/steps/02_compute_metrics.sh"
