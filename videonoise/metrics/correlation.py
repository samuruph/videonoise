"""
Correlation metrics: frame-to-frame, temporal ACF, 3D spatio-temporal.
"""
import numpy as np
import torch
from scipy import stats

from videonoise.io.video import to_grayscale


def frame_correlation(video: torch.Tensor) -> dict:
    """
    Pearson and Spearman correlation between each pair of consecutive frames.

    Args:
        video: (T, C, H, W) float tensor in [0, 1].

    Returns:
        Dict with mean/std of Pearson and Spearman coefficients.
    """
    gray = to_grayscale(video).squeeze(1)  # (T, H, W)
    T = gray.shape[0]
    pearson_vals, spearman_vals = [], []
    for t in range(T - 1):
        a = gray[t].flatten().numpy()
        b = gray[t + 1].flatten().numpy()
        pearson_vals.append(float(np.corrcoef(a, b)[0, 1]))
        spearman_vals.append(float(stats.spearmanr(a, b).statistic))
    return {
        "pearson_mean": float(np.mean(pearson_vals)),
        "pearson_std":  float(np.std(pearson_vals)),
        "spearman_mean": float(np.mean(spearman_vals)),
        "spearman_std":  float(np.std(spearman_vals)),
    }


def temporal_autocorrelation(video: torch.Tensor, max_lag: int = 10) -> dict:
    """
    Mean temporal ACF across all spatial pixels, for lags 1 … max_lag.

    Args:
        video:   (T, C, H, W) float tensor.
        max_lag: Maximum lag to compute.

    Returns:
        Dict mapping "lag_k" → mean ACF value.
    """
    gray = to_grayscale(video).squeeze(1)  # (T, H, W)
    T = gray.shape[0]
    pixels = gray.reshape(T, -1).numpy()  # (T, N)
    pixels = pixels - pixels.mean(axis=0, keepdims=True)
    var = (pixels ** 2).sum(axis=0) + 1e-8

    return {
        f"lag_{lag}": float(((pixels[:T - lag] * pixels[lag:]).sum(axis=0) / var).mean())
        for lag in range(1, min(max_lag + 1, T))
    }


def temporal_mutual_information(
    video: torch.Tensor, max_lag: int = 5, n_bins: int = 32
) -> dict:
    """
    Mutual information between pixel intensities at lag k (k = 1 … max_lag).

    Captures non-linear temporal dependence not visible in ACF.
    Estimated via 2D histogram joint entropy.

    Args:
        video:   (T, C, H, W) float tensor in [0, 1].
        max_lag: Maximum lag to compute.
        n_bins:  Number of histogram bins per axis.

    Returns:
        Dict mapping "lag_k" → MI in bits.
    """
    gray = to_grayscale(video).squeeze(1)  # (T, H, W)
    T = gray.shape[0]
    pixels = gray.reshape(T, -1).numpy()   # (T, N_pixels)

    # Sample at most 4096 random pixel positions for speed
    N = pixels.shape[1]
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(4096, N), replace=False)
    pixels = pixels[:, idx]  # (T, n_sample)

    result = {}
    for lag in range(1, min(max_lag + 1, T)):
        x = pixels[:T - lag].flatten()
        y = pixels[lag:].flatten()
        joint, _, _ = np.histogram2d(x, y, bins=n_bins, range=[[0, 1], [0, 1]])
        joint = joint / (joint.sum() + 1e-12)
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        px_py = px * py
        mask = joint > 0
        mi = float(np.sum(joint[mask] * np.log2(joint[mask] / (px_py[mask] + 1e-12))))
        result[f"lag_{lag}"] = max(0.0, mi)
    return result


def spatiotemporal_correlation_3d(video: torch.Tensor) -> dict:
    """
    3D FFT of the grayscale video cube; reports low- vs. high-frequency energy ratio.

    Args:
        video: (T, C, H, W) float tensor.

    Returns:
        Dict with low_freq_energy_ratio and high_freq_energy_ratio.
    """
    gray = to_grayscale(video).squeeze(1).numpy()  # (T, H, W)
    T, H, W = gray.shape
    power = np.abs(np.fft.fftn(gray)) ** 2
    total = power.sum()
    low_mask = np.zeros_like(power, dtype=bool)
    low_mask[:T // 4, :H // 4, :W // 4] = True
    low_ratio = float(power[low_mask].sum() / total)
    return {
        "low_freq_energy_ratio":  low_ratio,
        "high_freq_energy_ratio": 1.0 - low_ratio,
    }
