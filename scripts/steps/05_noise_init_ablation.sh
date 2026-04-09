#!/usr/bin/env bash
# Step 05 — Noise initialization ablation (gaussian / ar1 / spatial_lowpass / blue / perlin).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.noise_ablation \
    --model      hf  \
    --n_videos   10  \
    --num_frames 32  \
    --num_steps  50
