#!/usr/bin/env bash
# Step 00 — Download real reference videos.
# --dataset  synthetic | davis | vbench2 | ucf101
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.download_data \
    --dataset synthetic
