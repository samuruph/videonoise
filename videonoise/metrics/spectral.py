"""
Spectral analysis: 2D radial power spectrum, 3D frequency analysis, plots.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from videonoise.io.video import to_grayscale


def spatial_power_spectrum(video: torch.Tensor) -> dict:
    """
    Radially-averaged 2D power spectrum, averaged across all frames.

    Returns:
        Dict with radial_profile (list), spectral_slope, and spectral_r2
        (goodness of fit of the log-log power-law).
    """
    gray = to_grayscale(video).squeeze(1)  # (T, H, W)
    T, H, W = gray.shape

    spectra = [
        (torch.fft.rfft2(gray[t]).abs() ** 2).numpy()
        for t in range(T)
    ]
    mean_spectrum = np.mean(spectra, axis=0)

    cy, cx = H // 2, W // 2
    ys, xs = np.mgrid[0:H, 0:mean_spectrum.shape[1]]
    radii = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).astype(int)
    max_r = min(cy, cx)

    radial_profile = np.array([
        mean_spectrum[radii == r].mean() if (radii == r).any() else 0.0
        for r in range(1, max_r)
    ])

    freqs = np.arange(1, max_r)
    slope, _, r_value, _, _ = stats.linregress(
        np.log(freqs + 1e-8), np.log(radial_profile + 1e-8)
    )
    return {
        "spectral_slope":   float(slope),
        "spectral_r2":      float(r_value ** 2),
        "radial_profile":   radial_profile.tolist(),
    }


def frequency_analysis(video: torch.Tensor) -> dict:
    """
    3D FFT energy in temporal vs. spatial frequency bands.

    Returns:
        Dict with total_energy, temporal_low_freq_ratio, spatial_low_freq_ratio.
    """
    gray = to_grayscale(video).squeeze(1).numpy()  # (T, H, W)
    T, H, W = gray.shape
    f3d = np.abs(np.fft.fftn(gray)) ** 2
    total = f3d.sum()
    return {
        "total_energy":             float(total),
        "temporal_low_freq_ratio":  float(f3d[:T // 4].sum() / total),
        "spatial_low_freq_ratio":   float(f3d[:, :H // 4, :W // 4].sum() / total),
    }


def plot_3d_spectrum(video: torch.Tensor, output_dir: str, name: str) -> None:
    """Save marginal 1D power spectra along temporal and spatial axes."""
    gray = to_grayscale(video).squeeze(1).numpy()
    T, H, W = gray.shape
    f3d = np.abs(np.fft.fftn(gray)) ** 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, spec, label in zip(
        axes,
        [f3d.sum(axis=(1, 2))[:T // 2],
         f3d.sum(axis=(0, 2))[:H // 2],
         f3d.sum(axis=(0, 1))[:W // 2]],
        ["Temporal freq", "Spatial freq (H)", "Spatial freq (W)"],
    ):
        ax.semilogy(spec + 1e-12)
        ax.set_xlabel(label)
        ax.set_ylabel("Power")
        ax.set_title(f"{label} spectrum")
    plt.suptitle(name)
    plt.tight_layout()
    out = str(Path(output_dir) / f"{name}_3d_spectrum.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")
