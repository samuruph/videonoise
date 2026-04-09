#!/usr/bin/env bash
# Step 02 — Compute video metrics (correlation, spectral, quality).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.compute_metrics \
    --max_frames 32  \
    --resize     512 768
