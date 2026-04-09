#!/usr/bin/env bash
# Step 04 — Spatio-temporal analysis (PCA + 3D power spectrum).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.spatiotemporal_analysis \
    --max_frames 32  \
    --resize     512 768
