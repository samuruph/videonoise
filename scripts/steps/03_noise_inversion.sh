#!/usr/bin/env bash
# Step 03 — Noise inversion (recover latent noise + statistics).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.noise_inversion \
    --max_frames 32  \
    --resize     512 768
