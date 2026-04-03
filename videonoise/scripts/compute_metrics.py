"""
CLI: compute all video metrics on a folder of .mp4 files.

Usage:
    python -m videonoise.scripts.compute_metrics --input data/real/ --output results/metrics/real.json
    python -m videonoise.scripts.compute_metrics --input data/generated/ --output results/metrics/generated.json
"""
from videonoise.metrics import _cli

if __name__ == "__main__":
    _cli()
