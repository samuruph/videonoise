#!/usr/bin/env bash
# Step 02 — Compute video metrics (correlation, spectral, quality).
# Edit params below to change settings for this run.
# Any param not listed here is read from scripts/config.yaml.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python scripts/steps/compute_metrics.py \
    --max_frames 32  \
    --resize     512 768
