"""
Perceptual and pixel-level quality metrics: SSIM, PSNR, optical flow, statistics.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from skimage.metrics import structural_similarity as ski_ssim

from videonoise.io.video import to_grayscale


def temporal_ssim(video: torch.Tensor) -> dict:
    """
    SSIM between consecutive frames, averaged over time.

    Returns:
        Dict with temporal_ssim_mean/std and temporal_consistency_score (alias for mean).
    """
    gray = to_grayscale(video).squeeze(1).numpy()  # (T, H, W)
    T = gray.shape[0]
    ssim_vals = [
        float(ski_ssim(gray[t], gray[t + 1], data_range=1.0))
        for t in range(T - 1)
    ]
    mean = float(np.mean(ssim_vals))
    return {
        "temporal_ssim_mean":        mean,
        "temporal_ssim_std":         float(np.std(ssim_vals)),
        "temporal_consistency_score": mean,
    }


def frame_psnr(video: torch.Tensor) -> dict:
    """PSNR between consecutive grayscale frames."""
    gray = to_grayscale(video).squeeze(1)  # (T, H, W)
    T = gray.shape[0]
    psnr_vals = []
    for t in range(T - 1):
        mse = F.mse_loss(gray[t], gray[t + 1]).item()
        psnr_vals.append(float(10 * np.log10(1.0 / (mse + 1e-12))))
    return {
        "psnr_mean": float(np.mean(psnr_vals)),
        "psnr_std":  float(np.std(psnr_vals)),
    }


def frame_statistics(video: torch.Tensor) -> dict:
    """Basic pixel statistics (mean, std, skewness, kurtosis) across the whole video."""
    gray = to_grayscale(video).squeeze(1).numpy()
    flat = gray.flatten()
    return {
        "mean":     float(flat.mean()),
        "std":      float(flat.std()),
        "skewness": float(stats.skew(flat)),
        "kurtosis": float(stats.kurtosis(flat)),
    }


def optical_flow_stats(video: torch.Tensor) -> dict:
    """
    Dense Farneback optical flow between consecutive frames.

    Returns:
        Dict with mean, std, p95, p99 of per-pixel flow magnitude.
    """
    gray = (to_grayscale(video).squeeze(1).numpy() * 255).astype(np.uint8)
    T = gray.shape[0]
    magnitudes = []
    for t in range(T - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray[t], gray[t + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitudes.append(mag)

    all_mag = np.concatenate([m.flatten() for m in magnitudes])
    return {
        "flow_mag_mean": float(all_mag.mean()),
        "flow_mag_std":  float(all_mag.std()),
        "flow_mag_p95":  float(np.percentile(all_mag, 95)),
        "flow_mag_p99":  float(np.percentile(all_mag, 99)),
    }
