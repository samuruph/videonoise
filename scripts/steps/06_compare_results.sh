#!/usr/bin/env bash
# Step 06 — Compare real vs. generated and save a summary figure.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.compare_results
