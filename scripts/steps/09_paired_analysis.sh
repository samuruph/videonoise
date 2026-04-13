#!/usr/bin/env bash
# Step 09 — Paired video analysis (matched real vs. generated).
# Requires matched filenames between real and generated video sets.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python -m videonoise.scripts.paired_analysis
