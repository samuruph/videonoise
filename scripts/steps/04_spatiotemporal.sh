#!/usr/bin/env bash
# Step 04 — Spatio-temporal analysis (PCA + 3D power spectrum).
# Edit params below to change settings for this run.
# Any param not listed here is read from scripts/config.yaml.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python scripts/steps/spatiotemporal.py \
    --max_frames 32  \
    --resize     512 768
