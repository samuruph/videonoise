"""
High-level orchestrator: compute all metrics for a single video.
"""
import argparse

import torch
from tqdm import tqdm

from videonoise.metrics.correlation import (
    frame_correlation,
    temporal_autocorrelation,
    spatiotemporal_correlation_3d,
)
from videonoise.metrics.quality import (
    temporal_ssim,
    frame_psnr,
    frame_statistics,
    optical_flow_stats,
    flow_direction_entropy,
    lpips_temporal,
    multiscale_temporal_ssim,
)
from videonoise.metrics.correlation import temporal_mutual_information
from videonoise.metrics.spectral import spatial_power_spectrum

__all__ = [
    "compute_all_metrics",
    "frame_correlation",
    "temporal_autocorrelation",
    "temporal_mutual_information",
    "spatiotemporal_correlation_3d",
    "temporal_ssim",
    "frame_psnr",
    "frame_statistics",
    "optical_flow_stats",
    "flow_direction_entropy",
    "lpips_temporal",
    "multiscale_temporal_ssim",
    "spatial_power_spectrum",
]


def compute_all_metrics(video: torch.Tensor, name: str = "") -> dict:
    tag = f"[{name}]" if name else ""
    print(f"  {tag} correlation...")
    m = {
        "frame_correlation": frame_correlation(video),
        "temporal_acf": temporal_autocorrelation(video),
    }
    print(f"  {tag} temporal MI...")
    m["temporal_mi"] = temporal_mutual_information(video)
    print(f"  {tag} spectral...")
    m["spatial_power_spectrum"] = spatial_power_spectrum(video)
    m["spatiotemporal_3d"] = spatiotemporal_correlation_3d(video)
    print(f"  {tag} quality...")
    m["temporal_ssim"] = temporal_ssim(video)
    m["psnr"] = frame_psnr(video)
    m["frame_statistics"] = frame_statistics(video)
    print(f"  {tag} optical flow...")
    m["optical_flow"] = optical_flow_stats(video)
    m["flow_direction_entropy"] = flow_direction_entropy(video)
    print(f"  {tag} perceptual...")
    m["lpips_temporal"] = lpips_temporal(video)
    m["multiscale_ssim"] = multiscale_temporal_ssim(video)
    return m


# ---------------------------------------------------------------------------
# Console-script entry point  (vn-metrics)
# ---------------------------------------------------------------------------

def _cli() -> None:
    import numpy as np
    from videonoise.io import load_video_folder
    from videonoise.utils import save_json

    parser = argparse.ArgumentParser(description="Compute video metrics")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("W", "H"))
    args = parser.parse_args()

    resize = tuple(args.resize) if args.resize else None
    videos = load_video_folder(args.input, max_frames=args.max_frames, resize=resize)
    if not videos:
        print(f"No .mp4 files found in {args.input}")
        return

    per_video = {}
    for name, video in tqdm(videos, desc="Computing metrics", unit="video"):
        print(f"Processing {name} {tuple(video.shape)}")
        per_video[name] = compute_all_metrics(video, name)

    def _agg(results):
        agg = {}
        for key in next(iter(results.values())):
            sub = next(iter(results.values()))[key]
            if isinstance(sub, (int, float)):
                vals = [r[key] for r in results.values() if isinstance(r.get(key), (int, float))]
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            elif isinstance(sub, dict):
                agg[key] = {}
                for sk in sub:
                    sv = [r[key].get(sk) for r in results.values() if isinstance(r[key].get(sk), (int, float))]
                    if sv:
                        agg[key][sk] = {"mean": float(np.mean(sv)), "std": float(np.std(sv))}
        return agg

    save_json({"per_video": per_video, "aggregate": _agg(per_video)}, args.output)
    print(f"Saved → {args.output}")


def compute_metrics_for_folder(input_path: str, output_path: str, max_frames: int = 64, resize: tuple[int, int] | None = None) -> None:
    """Compute metrics for all videos in a folder and save JSON.

    This is the programmatic version used by other scripts.
    """
    import numpy as np
    from videonoise.io import load_video_folder
    from videonoise.utils import save_json

    videos = load_video_folder(input_path, max_frames=max_frames, resize=resize)
    if not videos:
        print(f"No .mp4 files found in {input_path}")
        return

    per_video = {}
    for name, video in tqdm(videos, desc="Computing metrics", unit="video"):
        print(f"Processing {name} {tuple(video.shape)}")
        per_video[name] = compute_all_metrics(video, name)

    def _agg(results):
        agg = {}
        for key in next(iter(results.values())):
            sub = next(iter(results.values()))[key]
            if isinstance(sub, (int, float)):
                vals = [r[key] for r in results.values() if isinstance(r.get(key), (int, float))]
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            elif isinstance(sub, dict):
                agg[key] = {}
                for sk in sub:
                    sv = [r[key].get(sk) for r in results.values() if isinstance(r[key].get(sk), (int, float))]
                    if sv:
                        agg[key][sk] = {"mean": float(np.mean(sv)), "std": float(np.std(sv))}
        return agg

    save_json({"per_video": per_video, "aggregate": _agg(per_video)}, output_path)
    print(f"Saved → {output_path}")
