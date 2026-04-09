#!/usr/bin/env bash
# Visualise metrics — works after step 02 alone.
# --mode  real | gen | both
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.viz_metrics \
    --mode both
