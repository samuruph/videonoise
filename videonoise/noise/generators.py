"""
Noise initialization strategies for video diffusion experiments.

Generators:
    gaussian    — i.i.d. N(0, 1) baseline
    ar1         — temporally correlated AR(1)
    spatial_lp  — spatially low-pass filtered Gaussian
    blue        — high-pass (blue) noise
    perlin      — fractal / Perlin-like via spectral synthesis
"""
import argparse
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def gaussian_noise(shape: tuple, **_) -> torch.Tensor:
    """Pure i.i.d. N(0, 1)."""
    return torch.randn(*shape)


def ar1_noise(shape: tuple, alpha: float = 0.5, **_) -> torch.Tensor:
    """
    Temporally correlated AR(1):
        ε_t = α · ε_{t-1} + √(1 − α²) · z_t

    Args:
        shape: (T, ...) — first dimension is time.
        alpha: Temporal correlation in [0, 1).
    """
    T = shape[0]
    eps = torch.zeros(*shape)
    z = torch.randn(*shape)
    eps[0] = z[0]
    scale = (1 - alpha ** 2) ** 0.5
    for t in range(1, T):
        eps[t] = alpha * eps[t - 1] + scale * z[t]
    return eps


def spatial_lowpass_noise(shape: tuple, sigma: float = 2.0, **_) -> torch.Tensor:
    """
    Spatially low-pass filtered Gaussian noise.

    Args:
        shape: (T, C, H, W).
        sigma: Gaussian blur std in pixels.
    """
    noise = torch.randn(*shape)
    T, C, H, W = shape
    ksize = int(6 * sigma + 1) | 1  # ensure odd
    for t in range(T):
        for c in range(C):
            frame = noise[t, c].numpy()
            noise[t, c] = torch.from_numpy(
                cv2.GaussianBlur(frame, (ksize, ksize), sigma)
            )
    return (noise - noise.mean()) / (noise.std() + 1e-8)


def blue_noise(shape: tuple, **_) -> torch.Tensor:
    """High-frequency (blue) noise = white − low-pass."""
    white = torch.randn(*shape)
    low   = spatial_lowpass_noise(shape, sigma=3.0)
    out   = white - low
    return (out - out.mean()) / (out.std() + 1e-8)


def perlin_noise(
    shape: tuple,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    **_,
) -> torch.Tensor:
    """
    Fractal / Perlin-like noise via spectral synthesis (sum of filtered Gaussians).

    Args:
        shape:       (T, C, H, W).
        octaves:     Number of frequency octaves.
        persistence: Amplitude decay per octave.
        lacunarity:  Frequency growth per octave.
    """
    _, _, _, W = shape
    result = torch.zeros(*shape)
    amplitude = 1.0
    frequency = 1.0
    total_amp = 0.0
    for _ in range(octaves):
        sigma = max(0.5, W / (2 * frequency))
        result += amplitude * spatial_lowpass_noise(shape, sigma=sigma)
        total_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    result = result / total_amp
    return (result - result.mean()) / (result.std() + 1e-8)


NOISE_GENERATORS: dict[str, Callable] = {
    "gaussian":   gaussian_noise,
    "ar1":        ar1_noise,
    "spatial_lp": spatial_lowpass_noise,
    "blue":       blue_noise,
    "perlin":     perlin_noise,
}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def noise_stats(eps: torch.Tensor) -> dict:
    """Mean, std, skewness, kurtosis of a noise tensor."""
    x = eps.flatten().numpy()
    return {
        "mean":     float(x.mean()),
        "std":      float(x.std()),
        "skewness": float(scipy_stats.skew(x)),
        "kurtosis": float(scipy_stats.kurtosis(x)),
    }


def temporal_correlation_noise(eps: torch.Tensor) -> list[float]:
    """Pearson correlation between consecutive time slices of a noise tensor."""
    T = eps.shape[0]
    corrs = []
    for t in range(T - 1):
        a = eps[t].flatten().numpy()
        b = eps[t + 1].flatten().numpy()
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return corrs


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_noise_comparison(
    noises: dict[str, torch.Tensor],
    output_dir: str,
    frame_idx: int = 0,
    channel: int = 0,
) -> None:
    """Save a grid comparing noise types: spatial map (top) + histogram (bottom)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n = len(noises)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    x_ref = np.linspace(-4, 4, 200)
    gauss_ref = np.exp(-x_ref ** 2 / 2) / np.sqrt(2 * np.pi)

    for col, (name, eps) in enumerate(noises.items()):
        frame = eps[frame_idx, channel].numpy()
        axes[0, col].imshow(frame, cmap="RdBu_r", vmin=-3, vmax=3)
        axes[0, col].set_title(name)
        axes[0, col].axis("off")

        axes[1, col].hist(eps.flatten().numpy(), bins=60, density=True, alpha=0.7, color="steelblue")
        axes[1, col].plot(x_ref, gauss_ref, "r--", label="N(0,1)")
        axes[1, col].set_xlim(-4, 4)
        axes[1, col].set_xlabel("value")
        if col == 0:
            axes[1, col].set_ylabel("density")
        axes[1, col].legend(fontsize=8)

    plt.tight_layout()
    out = str(Path(output_dir) / "noise_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Console-script entry point  (vn-noise-init)
# ---------------------------------------------------------------------------

def _cli() -> None:
    from videonoise.utils import save_json

    parser = argparse.ArgumentParser(description="Generate and compare noise initializations")
    parser.add_argument("--type", choices=list(NOISE_GENERATORS) + ["all"], default="all")
    parser.add_argument("--shape", type=int, nargs=4, default=[16, 4, 32, 32], metavar=("T", "C", "H", "W"))
    parser.add_argument("--alpha",   type=float, default=0.8)
    parser.add_argument("--sigma",   type=float, default=2.0)
    parser.add_argument("--octaves", type=int,   default=4)
    parser.add_argument("--output",  default="results/noise_init/")
    args = parser.parse_args()

    shape  = tuple(args.shape)
    kwargs = {"alpha": args.alpha, "sigma": args.sigma, "octaves": args.octaves}
    types  = list(NOISE_GENERATORS) if args.type == "all" else [args.type]

    Path(args.output).mkdir(parents=True, exist_ok=True)
    results, noises = {}, {}
    for nt in types:
        print(f"Generating {nt} noise {shape}...")
        eps = NOISE_GENERATORS[nt](shape, **kwargs)
        noises[nt] = eps
        corrs = temporal_correlation_noise(eps)
        results[nt] = {
            "stats":                noise_stats(eps),
            "temporal_corr_mean":   float(np.mean(corrs)),
            "temporal_corr_std":    float(np.std(corrs)),
        }
        s = results[nt]
        print(f"  mean={s['stats']['mean']:.4f}  std={s['stats']['std']:.4f}  "
              f"temp_corr={s['temporal_corr_mean']:.4f}")

    save_json(results, str(Path(args.output) / "noise_init_stats.json"))
    visualize_noise_comparison(noises, args.output)
