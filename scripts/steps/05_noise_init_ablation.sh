#!/usr/bin/env bash
# Step 05 — Noise initialization ablation (gaussian / ar1 / spatial_lowpass / blue / perlin).
# Edit params below to change settings for this run.
# Any param not listed here is read from scripts/config.yaml.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python scripts/steps/noise_ablation.py \
    --model      hf  \
    --n_videos   10  \
    --num_frames 32  \
    --num_steps  50
