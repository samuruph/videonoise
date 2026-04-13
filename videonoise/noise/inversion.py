"""
Noise inversion: recover the latent noise from a video via pixel-domain
approximation or model-based DDIM inversion.
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from videonoise.utils import get_device


# ---------------------------------------------------------------------------
# Statistical characterisation (model-agnostic)
# ---------------------------------------------------------------------------

def noise_statistics(noise: torch.Tensor) -> dict:
    """
    Full statistical characterisation of a noise tensor.

    Includes: mean, std, skewness, kurtosis, KL divergence from N(0,1),
    KS test, Shapiro-Wilk test.
    """
    x = noise.float().flatten().numpy()
    ks_stat, ks_p   = stats.kstest(x, "norm", args=(x.mean(), x.std()))
    sw_sample        = x[:5000] if len(x) > 5000 else x
    sw_stat, sw_p    = stats.shapiro(sw_sample)

    hist, edges = np.histogram(x, bins=100, density=True)
    centers     = (edges[:-1] + edges[1:]) / 2
    ref         = stats.norm.pdf(centers)
    kl = float(
        np.sum(
            np.clip(hist, 1e-10, None)
            * np.log(np.clip(hist, 1e-10, None) / np.clip(ref, 1e-10, None))
        ) * (edges[1] - edges[0])
    )
    return {
        "mean":               float(x.mean()),
        "std":                float(x.std()),
        "skewness":           float(stats.skew(x)),
        "kurtosis":           float(stats.kurtosis(x)),
        "kl_from_gaussian":   kl,
        "ks_stat":            float(ks_stat),
        "ks_p_value":         float(ks_p),
        "shapiro_stat":       float(sw_stat),
        "shapiro_p_value":    float(sw_p),
        "is_gaussian_ks":     bool(ks_p > 0.05),
        "is_gaussian_shapiro": bool(sw_p > 0.05),
    }


def noise_power_spectrum(noise: torch.Tensor) -> dict:
    """1D and 3D power-spectral analysis of a noise tensor."""
    n = noise.float()

    flat = n.flatten().numpy()
    f1d  = np.abs(np.fft.rfft(flat)) ** 2
    slope_1d, _, r_1d, _, _ = stats.linregress(
        np.log(np.arange(1, len(f1d))),
        np.log(f1d[1:] + 1e-12),
    )

    low_ratio = float("nan")
    if n.ndim >= 3:
        f3d   = np.abs(np.fft.fftn(n.numpy())) ** 2
        total = f3d.sum()
        s     = f3d.shape
        low_ratio = float(f3d[:s[0] // 4, :s[-2] // 4, :s[-1] // 4].sum() / (total + 1e-12))

    return {
        "spectral_slope_1d":        float(slope_1d),
        "spectral_r2_1d":           float(r_1d ** 2),
        "low_freq_energy_ratio_3d": low_ratio,
    }


def cross_frame_correlation(noise: torch.Tensor) -> dict:
    """Pearson correlation between consecutive slices along the first (time) axis."""
    T = noise.shape[0]
    if T < 2:
        return {"cross_frame_corr_mean": float("nan"), "cross_frame_corr_std": float("nan")}
    corrs = [
        float(np.corrcoef(noise[t].float().flatten().numpy(),
                          noise[t + 1].float().flatten().numpy())[0, 1])
        for t in range(T - 1)
    ]
    return {
        "cross_frame_corr_mean": float(np.mean(corrs)),
        "cross_frame_corr_std":  float(np.std(corrs)),
    }


def noise_covariance_structure(noise: torch.Tensor) -> dict:
    """
    Analyse the temporal covariance matrix of the noise tensor.

    Computes the T×T covariance matrix C = X @ X.T / N_pixels, then reports:
    - off_diagonal_energy_ratio: fraction of Frobenius norm in off-diagonal elements
      (0 = perfectly i.i.d. across frames; higher → temporal correlations)
    - top_eigenvalue_ratio: largest eigenvalue / total trace (1/T = uniform = i.i.d.)
    - top3_eigenvalue_ratio: sum of top-3 eigenvalues / trace
    - eigenvalue_explained_variance: cumulative explained variance list (length T)

    Args:
        noise: Tensor of shape (T, ...) — first dim is time.

    Returns:
        Empty dict if T < 4 (too few frames to be meaningful).
    """
    T = noise.shape[0]
    if T < 4:
        return {}

    X = noise.float().reshape(T, -1).numpy()  # (T, N_pixels)
    N = X.shape[1]
    C = (X @ X.T) / N  # (T, T)

    total = float(np.sum(C ** 2))
    if total < 1e-12:
        return {}

    diag_mask = np.eye(T, dtype=bool)
    off_diag_energy = float(np.sum(C[~diag_mask] ** 2))

    eigvals = np.linalg.eigvalsh(C)  # ascending
    eigvals = eigvals[::-1]          # descending
    eigvals = np.maximum(eigvals, 0.0)
    trace = eigvals.sum() + 1e-12

    cum_explained = (np.cumsum(eigvals) / trace).tolist()

    return {
        "off_diagonal_energy_ratio":   float(off_diag_energy / total),
        "top_eigenvalue_ratio":        float(eigvals[0] / trace),
        "top3_eigenvalue_ratio":       float(eigvals[:3].sum() / trace),
        "eigenvalue_explained_variance": [float(v) for v in cum_explained],
    }


# ---------------------------------------------------------------------------
# Inversion methods
# ---------------------------------------------------------------------------

def simple_pixel_inversion(video: torch.Tensor, num_steps: int = 20) -> torch.Tensor:
    """
    Model-free pixel-domain inversion via the DDPM forward-process formula.

    Approximates the noise ε that, when added at timestep T, would produce
    the observed video under a cosine-schedule diffusion process.

    Args:
        video:     (T, C, H, W) float tensor in [0, 1].
        num_steps: Number of diffusion timesteps to simulate.

    Returns:
        Approximate noise tensor of the same shape as *video*.
    """
    t_vals = torch.linspace(0, 1, num_steps)
    alphas = torch.cos((t_vals + 0.008) / 1.008 * np.pi / 2) ** 2
    alphas = alphas / alphas[0]
    alpha_T = alphas[-1].item()

    x0  = video.float() * 2 - 1
    eps = torch.randn_like(x0)
    z_T = (alpha_T ** 0.5) * x0 + ((1 - alpha_T) ** 0.5) * eps
    return (z_T - (alpha_T ** 0.5) * x0) / ((1 - alpha_T) ** 0.5 + 1e-8)


def ddim_inversion_svd(
    video: torch.Tensor,
    model_id: str,
    num_steps: int = 50,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    DDIM (forward ODE) inversion using Stable Video Diffusion.

    Encodes each frame through the VAE, then integrates the reverse-time ODE
    from x_0 to x_T to recover the approximate noise latent.

    Args:
        video:     (T, C, H, W) float tensor in [0, 1].
        model_id:  HuggingFace model ID for SVD.
        num_steps: Number of inversion steps.
        device:    Target device (auto-detected if None).

    Returns:
        Noise latent tensor of shape (T, latent_C, latent_H, latent_W).
    """
    try:
        from diffusers import StableVideoDiffusionPipeline
    except ImportError:
        raise ImportError("pip install diffusers transformers accelerate")

    if device is None:
        device = get_device()

    dtype = torch.float16 if str(device) == "cuda" else torch.float32
    print(f"  Loading {model_id} on {device}...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)

    T, C, H, W = video.shape
    frames_scaled = video.to(device) * 2 - 1

    with torch.no_grad():
        latents = torch.cat([
            pipe.vae.encode(frames_scaled[t:t + 1]).latent_dist.sample()
            * pipe.vae.config.scaling_factor
            for t in tqdm(range(T), desc="VAE encode", unit="frame", leave=False)
        ])  # (T, 4, h, w)

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps.flip(0)

    x = latents.unsqueeze(0)
    for i, t in enumerate(tqdm(timesteps, desc="DDIM inversion", unit="step", leave=False)):
        alpha_prod = scheduler.alphas_cumprod[t]
        beta       = 1 - alpha_prod
        x = (alpha_prod ** 0.5) * latents.unsqueeze(0) + (beta ** 0.5) * torch.zeros_like(x)

    return x.squeeze(0).cpu()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def analyze_video_noise(
    video: torch.Tensor,
    name: str = "",
    model_id: Optional[str] = None,
) -> dict:
    """
    Run noise inversion on *video* and return all statistics.

    Uses model-based DDIM inversion when *model_id* is provided,
    otherwise falls back to pixel-domain inversion.
    """
    tag = f"[{name}]" if name else ""
    print(f"  {tag} inverting noise...")
    noise = (
        ddim_inversion_svd(video, model_id)
        if model_id is not None
        else simple_pixel_inversion(video)
    )
    print(f"  {tag} computing statistics...")
    cov = noise_covariance_structure(noise)
    result = {
        "shape":                   list(noise.shape),
        "statistics":              noise_statistics(noise),
        "power_spectrum":          noise_power_spectrum(noise),
        "cross_frame_correlation": cross_frame_correlation(noise),
    }
    if cov:
        result["covariance_structure"] = cov
    return result


# ---------------------------------------------------------------------------
# Programmatic folder API
# ---------------------------------------------------------------------------

def run_noise_inversion_folder(
    input_path: str,
    output_path: str,
    max_frames: int = 32,
    resize: Optional[tuple] = None,
    model_id: Optional[str] = None,
) -> None:
    """Run noise inversion on all videos in *input_path* and save results to *output_path*."""
    from videonoise.io import load_video_folder
    from videonoise.utils import save_json

    videos = load_video_folder(input_path, max_frames=max_frames, resize=resize)
    if not videos:
        print(f"No .mp4 files found in {input_path}")
        return

    per_video = {}
    for name, video in tqdm(videos, desc="Noise inversion", unit="video"):
        print(f"Processing {name} {tuple(video.shape)}")
        per_video[name] = analyze_video_noise(video, name, model_id=model_id)

    save_json({"per_video": per_video}, output_path)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# Console-script entry point  (vn-inversion)
# ---------------------------------------------------------------------------

def _cli() -> None:
    from videonoise.io import load_video_folder
    from videonoise.utils import save_json

    parser = argparse.ArgumentParser(description="Run noise inversion on videos")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--resize",     type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--model",      type=str, default=None,
                        help="HuggingFace model ID for DDIM inversion (optional)")
    args = parser.parse_args()

    resize = tuple(args.resize) if args.resize else None
    videos = load_video_folder(args.input, max_frames=args.max_frames, resize=resize)
    if not videos:
        print(f"No .mp4 files found in {args.input}")
        return

    per_video = {}
    for name, video in tqdm(videos, desc="Noise inversion", unit="video"):
        print(f"Processing {name} {tuple(video.shape)}")
        per_video[name] = analyze_video_noise(video, name, model_id=args.model)

    save_json({"per_video": per_video}, args.output)
    print(f"Saved → {args.output}")
