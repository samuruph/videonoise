#!/usr/bin/env bash
# Intermediate visualisation — works after step 02 alone.
# --mode  real | gen | both   (default: both)
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python scripts/steps/viz.py \
    --mode both
