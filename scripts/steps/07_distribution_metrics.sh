#!/usr/bin/env bash
# Step 07 — Dataset-level distribution metrics: FVD, cross-set LPIPS, CLIP score.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.distribution_metrics
