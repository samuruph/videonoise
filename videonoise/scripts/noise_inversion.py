"""
CLI: run noise inversion on a folder of .mp4 files.

Usage:
    python -m videonoise.scripts.noise_inversion --input data/real/ --output results/noise_stats/real.json
    python -m videonoise.scripts.noise_inversion --input data/generated/ --output results/noise_stats/generated.json
    python -m videonoise.scripts.noise_inversion --input data/generated/ --output results/noise_stats/generated.json \\
        --model stabilityai/stable-video-diffusion-img2vid-xt
"""
from videonoise.noise.inversion import _cli

if __name__ == "__main__":
    _cli()
