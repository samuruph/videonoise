#!/usr/bin/env bash
# Step 01 — Generate videos with a diffusion model.
# Edit params below to change settings for this run.
# Any param not listed here is read from scripts/config.yaml.
#
# --model      svd | modelscope | cogvideox | wan
# --noise_type gaussian | ar1 | spatial_lowpass | blue | perlin
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python scripts/steps/generate_videos.py \
    --model      hf       \
    --noise_type gaussian \
    --n_videos   10       \
    --num_frames 32       \
    --num_steps  50       \
    --seed       42
