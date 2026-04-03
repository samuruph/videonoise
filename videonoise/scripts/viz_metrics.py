"""
CLI: visualise a single metrics JSON (real or generated).

Usage:
    python -m videonoise.scripts.viz_metrics results/metrics/real.json
    python -m videonoise.scripts.viz_metrics results/metrics/wan_gaussian.json --label "Wan gaussian"
    vn-viz results/metrics/real.json
"""
from videonoise.analysis.plots import _cli_viz

if __name__ == "__main__":
    _cli_viz()
