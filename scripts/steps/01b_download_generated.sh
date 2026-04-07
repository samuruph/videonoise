#!/usr/bin/env bash
# Step 01b — Download pre-generated / stock videos (alternative to step 01).
#
# Use this instead of step 01 when you do not have a GPU or do not want
# to wait for generation.  Downloads N_VIDEOS clips (matching DAVIS size) and
# places them in data/generated/<run_key>/ so all downstream steps work
# identically to self-generated videos.
#
# ── Backends ────────────────────────────────────────────────────────────────
#
#   archive        Free videos from archive.org (Internet Archive).  NO KEY NEEDED.
#                 Public-domain and CC-licensed clips on any topic.
#
#   hf_generated  AI-generated evaluation videos from HuggingFace.  NO KEY NEEDED.
#                 All sources below are verified freely accessible (no login).
#                 Sources (pass as OPTION):
#                   vbench2    (default) — 30k CogVideo outputs, VBench 2.0
#                   cogvideox  — CogVideoX-2b / 5b outputs (78 videos)
#                   wan        — Wan T2V baseline outputs (121 videos)
#
#   pexels        Short clips from Pexels.  FREE API KEY REQUIRED.
#                 Get a free key in 30 s at: https://www.pexels.com/api/
#
# ── Usage examples ──────────────────────────────────────────────────────────
#
#   # Internet Archive — nature clips, no key (RECOMMENDED, easiest)
#   bash scripts/steps/01b_download_generated.sh
#
#   # Internet Archive — city scenes
#   bash scripts/steps/01b_download_generated.sh archive "city"
#
#   # HuggingFace — VBench 2.0 CogVideo outputs (default, 30k videos)
#   bash scripts/steps/01b_download_generated.sh hf_generated vbench2
#
#   # HuggingFace — CogVideoX-2b / 5b outputs (78 videos)
#   bash scripts/steps/01b_download_generated.sh hf_generated cogvideox
#
#   # HuggingFace — Wan T2V baseline outputs (121 videos)
#   bash scripts/steps/01b_download_generated.sh hf_generated wan
#
#   # Pexels — nature clips (free key required)
#   PEXELS_KEY=your_key bash scripts/steps/01b_download_generated.sh pexels "nature"
#
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"

BACKEND="${1:-archive}"          # archive | hf_generated | pexels
OPTION="${2:-}"                  # query string (archive/pexels) OR hf_source key

# Set per-backend defaults for OPTION so the banner and branches are consistent
case "$BACKEND" in
    hf_generated) OPTION="${OPTION:-vbench2}" ;;
    *)            OPTION="${OPTION:-nature}"  ;;
esac

echo "================================================================"
echo "  Step 01b — Download pre-generated / stock videos"
echo "  Backend : $BACKEND"
echo "  Option  : $OPTION"
echo "  N       : $N_VIDEOS  videos"
echo "================================================================"

if [ "$BACKEND" = "archive" ]; then
    # ── Internet Archive (no key) ─────────────────────────────────────────
    SLUG=$(echo "$OPTION" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | cut -c1-20)
    OUT_DIR="${DATA_GEN}archive_${SLUG}/"
    echo "  Output  : $OUT_DIR"
    python -m videonoise.scripts.download_data \
        --dataset archive \
        --query   "$OPTION" \
        --n       "$N_VIDEOS" \
        --output  "$OUT_DIR"

elif [ "$BACKEND" = "hf_generated" ]; then
    # ── HuggingFace Hub (no key) ──────────────────────────────────────────
    HF_SOURCE="$OPTION"
    SLUG=$(echo "$HF_SOURCE" | tr '_' '-')
    OUT_DIR="${DATA_GEN}hf_${SLUG}/"
    echo "  HF source : $HF_SOURCE"
    echo "  Output    : $OUT_DIR"
    python -m videonoise.scripts.download_data \
        --dataset   hf_generated \
        --hf_source "$HF_SOURCE" \
        --n         "$N_VIDEOS" \
        --output    "$OUT_DIR"

elif [ "$BACKEND" = "pexels" ]; then
    # ── Pexels (free key required) ────────────────────────────────────────
    KEY="${PEXELS_KEY:-}"
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
        --dataset pexels \
        --api_key "$KEY" \
        --query   "$OPTION" \
        --n       "$N_VIDEOS" \
        --output  "$OUT_DIR"

else
    echo "[error] Unknown backend '$BACKEND'."
    echo "  Use: archive | hf_generated | pexels"
    exit 1
fi

echo ""
echo "  Done. Videos saved to $OUT_DIR"
echo ""
echo "  To analyse these videos, update config.yaml:"
RUN_KEY=$(basename "${OUT_DIR%/}")
echo "    model: $(echo "$RUN_KEY" | cut -d_ -f1)"
echo "    noise_type: $(echo "$RUN_KEY" | cut -d_ -f2-)"
echo ""
echo "  Then run:"
echo "    bash scripts/steps/02_compute_metrics.sh"
