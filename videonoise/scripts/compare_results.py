"""
CLI: load saved metric JSONs and produce comparison plots + summary table.

Usage:
    python -m videonoise.scripts.compare_results \\
        --real results/metrics/real.json \\
        --generated results/metrics/generated.json \\
        --noise_real results/noise_stats/real.json \\
        --noise_gen results/noise_stats/generated.json \\
        --output results/plots/comparison/
"""
from videonoise.analysis.plots import _cli

if __name__ == "__main__":
    _cli()
