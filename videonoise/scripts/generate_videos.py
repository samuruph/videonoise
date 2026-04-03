"""
CLI: generate videos with a diffusion model using custom noise initialisations.

Usage:
    python -m videonoise.scripts.generate_videos --model svd        --noise_type gaussian --n 5 --output data/generated/svd_gaussian/
    python -m videonoise.scripts.generate_videos --model modelscope --noise_type ar1 --alpha 0.8 --n 5 --output data/generated/modelscope_ar1/
    python -m videonoise.scripts.generate_videos --model cogvideox  --noise_type perlin --n 5 --output data/generated/cogvideox_perlin/
    python -m videonoise.scripts.generate_videos --model wan        --noise_type blue   --n 5 --output data/generated/wan_blue/

Models:  svd | modelscope | cogvideox | wan
Noises:  gaussian | ar1 | spatial_lowpass | blue | perlin
"""
from videonoise.diffusion.pipelines import _cli

if __name__ == "__main__":
    _cli()
