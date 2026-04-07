#!/usr/bin/env bash
# Step 03 — Noise inversion (recover latent noise + statistics).
# Edit params below to change settings for this run.
# Any param not listed here is read from scripts/config.yaml.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python scripts/steps/noise_inversion.py \
    --max_frames 32  \
    --resize     512 768
