#!/usr/bin/env bash
# Step 05 — Noise initialization ablation (gaussian / ar1 / spatial_lowpass / blue / perlin).
# Output always goes to data/generated/<model>_<noise>/ regardless of data_gen in config.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.noise_ablation \
    --model      wan  \
    --data_gen   data/generated/ \
    --n_videos   10  \
    --num_steps  50
