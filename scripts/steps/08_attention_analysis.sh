#!/usr/bin/env bash
# Step 08 — Attention map extraction and analysis.
# Requires model_id to be set in scripts/config.yaml and a GPU.
#   --n_videos  : number of videos to process (default: 3)
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.attention_analysis --config scripts/config.yaml \
    --n_videos 3
