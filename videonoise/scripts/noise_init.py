"""
CLI: generate and visualise different noise initialisations.

Usage:
    python -m videonoise.scripts.noise_init --type all --shape 16 4 32 32 --output results/noise_init/
    python -m videonoise.scripts.noise_init --type ar1 --alpha 0.9 --output results/noise_init/
"""
from videonoise.noise.generators import _cli

if __name__ == "__main__":
    _cli()
