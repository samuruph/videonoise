#!/usr/bin/env bash
# Step 01 — Generate videos with a diffusion model.
# --model      svd | modelscope | cogvideox | wan
# --noise_type gaussian | ar1 | spatial_lowpass | blue | perlin
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.generate_videos \
    --model      hf       \
    --noise_type gaussian \
    --n_videos   10       \
    --num_frames 32       \
    --num_steps  50       \
    --seed       42
