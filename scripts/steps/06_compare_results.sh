#!/usr/bin/env bash
# Step 06 — Compare real vs. generated and save a summary figure.
# Reads results from the paths derived in config.yaml (no compute params needed).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

python scripts/steps/compare_results.py
