"""
Perceptual and pixel-level quality metrics: SSIM, PSNR, optical flow, statistics.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from skimage.metrics import structural_similarity as ski_ssim
from skimage.transform import rescale

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


def flow_direction_entropy(video: torch.Tensor) -> dict:
    """
    Shannon entropy of optical-flow angle histogram between consecutive frames.

    Uniform angle distribution (max entropy ≈ log2(36) ≈ 5.17 bits) indicates
    chaotic / unstructured motion; low entropy indicates coherent, directed motion.

    Returns:
        Dict with flow_direction_entropy_mean and flow_direction_entropy_std.
    """
    gray = (to_grayscale(video).squeeze(1).numpy() * 255).astype(np.uint8)
    T = gray.shape[0]
    entropies = []
    for t in range(T - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray[t], gray[t + 1], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        angles = np.arctan2(flow[..., 1], flow[..., 0])  # (-π, π)
        hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
        p = hist / (hist.sum() + 1e-8)
        H = float(-np.sum(p * np.log2(p + 1e-8)))
        entropies.append(H)
    return {
        "flow_direction_entropy_mean": float(np.mean(entropies)),
        "flow_direction_entropy_std":  float(np.std(entropies)),
    }


def lpips_temporal(video: torch.Tensor, net: str = "alex") -> dict:
    """
    LPIPS perceptual distance between consecutive frames (temporal consistency).

    Lower = more perceptually consistent across time.

    Returns:
        Dict with lpips_mean and lpips_std.
    """
    try:
        import lpips as lpips_lib
    except ImportError:
        raise ImportError("pip install lpips")

    loss_fn = lpips_lib.LPIPS(net=net, verbose=False)
    loss_fn.eval()

    # LPIPS expects (N, C, H, W) in [-1, 1]; video is (T, C, H, W) in [0, 1]
    frames = video.float() * 2 - 1
    # Ensure 3-channel input
    if frames.shape[1] == 1:
        frames = frames.expand(-1, 3, -1, -1)

    T = frames.shape[0]
    vals = []
    with torch.no_grad():
        for t in range(T - 1):
            d = loss_fn(frames[t:t + 1], frames[t + 1:t + 2]).item()
            vals.append(float(d))
    return {
        "lpips_mean": float(np.mean(vals)),
        "lpips_std":  float(np.std(vals)),
    }


def multiscale_temporal_ssim(video: torch.Tensor, n_scales: int = 4) -> dict:
    """
    Temporal SSIM computed at multiple spatial scales (each halved by 0.5^k).

    Captures coherence degradation from fine to coarse structure.
    Scales where the frame drops below 16px in any dim are skipped.

    Returns:
        Dict like {"scale_0": {"ssim_mean": float, "ssim_std": float}, ...}
    """
    gray = to_grayscale(video).squeeze(1).numpy()  # (T, H, W)
    result = {}
    for k in range(n_scales):
        scale_factor = 0.5 ** k
        if k == 0:
            scaled = gray
        else:
            scaled = np.stack([
                rescale(gray[t], scale_factor, anti_aliasing=True, channel_axis=None)
                for t in range(gray.shape[0])
            ])
        H, W = scaled.shape[1], scaled.shape[2]
        if H < 16 or W < 16:
            break
        ssim_vals = [
            float(ski_ssim(scaled[t], scaled[t + 1], data_range=1.0))
            for t in range(scaled.shape[0] - 1)
        ]
        result[f"scale_{k}"] = {
            "ssim_mean": float(np.mean(ssim_vals)),
            "ssim_std":  float(np.std(ssim_vals)),
        }
    return result
