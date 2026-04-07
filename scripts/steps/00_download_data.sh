#!/usr/bin/env bash
# Step 00 — Download real reference videos.
# --dataset  synthetic | davis | vbench2 | ucf101   (default: synthetic)
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python scripts/steps/download_data.py \
    --dataset synthetic
