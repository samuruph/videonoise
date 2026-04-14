"""
Dataset-level distribution metrics: FVD, cross-set LPIPS, CLIP score.

These require two video *sets* (real + generated), not individual videos.
They are NOT integrated into compute_all_metrics() — call them separately
via videonoise.scripts.distribution_metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_video_tensors(
    folder: str,
    max_frames: int,
    resize: Optional[tuple[int, int]],
    device: torch.device,
) -> list[torch.Tensor]:
    """Return list of (T, C, H, W) float tensors in [0, 1]."""
    from videonoise.io import load_video_folder
    videos = load_video_folder(folder, max_frames=max_frames, resize=resize)
    return [v.to(device) for _, v in videos]


def _frames_to_uint8(videos: list[torch.Tensor]) -> torch.Tensor:
    """Stack videos → (N, T, C, H, W) uint8."""
    min_T = min(v.shape[0] for v in videos)
    stacked = torch.stack([v[:min_T] for v in videos])  # (N, T, C, H, W)
    return (stacked * 255).clamp(0, 255).byte()


# ─────────────────────────────────────────────────────────────────────────────
# FVD
# ─────────────────────────────────────────────────────────────────────────────

def _extract_video_features(
    videos: list[torch.Tensor],
    device: torch.device,
    batch_size: int = 4,
    clip_len: int = 16,
) -> np.ndarray:
    """
    Extract spatiotemporal features from a list of (T, C, H, W) video tensors
    using torchvision's R3D-18 backbone (no extra pip installs required).

    Videos are resized to 112×112 (standard R3D input), temporally sampled to
    *clip_len* frames, and passed through the network with the classification
    head removed.  Returns an (N, 512) float64 feature matrix.
    """
    import torchvision.models.video as vm

    # Load R3D-18 with pretrained Kinetics-400 weights (cached after first use)
    weights = vm.R3D_18_Weights.KINETICS400_V1
    backbone = vm.r3d_18(weights=weights)
    backbone.fc = torch.nn.Identity()  # drop classifier → 512-d features
    backbone = backbone.to(device).eval()

    # R3D normalisation (ImageNet-style, applied per-frame)
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
    std  = torch.tensor([0.22803, 0.22145,  0.216989], device=device).view(1, 3, 1, 1, 1)

    feats: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(videos), batch_size):
            batch_clips = []
            for v in videos[start : start + batch_size]:
                # v: (T, C, H, W) float in [0,1]
                T = v.shape[0]
                # Temporal sampling → clip_len frames
                idx = torch.linspace(0, T - 1, clip_len).long()
                clip = v[idx].float()  # (clip_len, C, H, W)

                # Spatial resize to 112×112
                clip = torch.nn.functional.interpolate(
                    clip, size=(112, 112), mode="bilinear", align_corners=False,
                )
                # R3D expects (C, T, H, W)
                batch_clips.append(clip.permute(1, 0, 2, 3))

            # Stack → (B, C, T, H, W)
            x = torch.stack(batch_clips).to(device)
            x = (x - mean) / std
            f = backbone(x)  # (B, 512)
            feats.append(f.cpu().float().numpy())

    return np.concatenate(feats, axis=0).astype(np.float64)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                      mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Fréchet distance between two multivariate Gaussians."""
    from scipy import linalg

    diff = mu1 - mu2
    # Symmetric matrix square root of sigma1 @ sigma2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fvd(
    real_folder: str,
    gen_folder: str,
    max_frames: int = 32,
    resize: Optional[tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 4,
    clip_len: int = 16,
) -> float:
    """
    Fréchet Video Distance between real and generated video sets.

    Uses torchvision's R3D-18 (Kinetics-400 pretrained) as the feature
    extractor — no TensorFlow or extra pip installs required.

    Returns:
        FVD scalar (lower = more similar distributions).
    """
    if device is None:
        from videonoise.utils import get_device
        device = get_device()

    real_vids = _load_video_tensors(real_folder, max_frames, resize, device)
    gen_vids  = _load_video_tensors(gen_folder,  max_frames, resize, device)

    if len(real_vids) < 2 or len(gen_vids) < 2:
        raise ValueError(
            f"FVD requires ≥ 2 videos per set (got {len(real_vids)} real, {len(gen_vids)} gen)"
        )

    print(f"        Extracting R3D-18 features for {len(real_vids)} real videos...")
    real_feats = _extract_video_features(real_vids, device, batch_size, clip_len)
    print(f"        Extracting R3D-18 features for {len(gen_vids)} generated videos...")
    gen_feats  = _extract_video_features(gen_vids,  device, batch_size, clip_len)

    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = gen_feats.mean(0),  np.cov(gen_feats,  rowvar=False)

    return _frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-set LPIPS
# ─────────────────────────────────────────────────────────────────────────────

_LPIPS_MAX_SIDE = 256  # LPIPS is a perceptual metric; full-res is wasteful and slow


def _lpips_resize(frames: torch.Tensor) -> torch.Tensor:
    """Downscale (T, C, H, W) so the longer side ≤ _LPIPS_MAX_SIDE."""
    H, W = frames.shape[-2], frames.shape[-1]
    scale = _LPIPS_MAX_SIDE / max(H, W)
    if scale >= 1.0:
        return frames
    nH, nW = max(1, int(H * scale)), max(1, int(W * scale))
    return torch.nn.functional.interpolate(
        frames, size=(nH, nW), mode="bilinear", align_corners=False,
    )


def compute_lpips_between_sets(
    real_folder: str,
    gen_folder: str,
    max_frames: int = 32,
    resize: Optional[tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 16,
) -> dict:
    """
    Compute LPIPS metrics:
    - lpips_temporal_real: mean LPIPS between consecutive real frames (temporal consistency)
    - lpips_temporal_gen:  same for generated
    - lpips_real_vs_gen:   mean LPIPS between corresponding real/gen frames at each timestep
                           (only computed if real and gen video counts match)

    Frames are downscaled to ≤256px before LPIPS (perceptual metric — full res is wasteful).
    Frame pairs are batched for throughput.

    Returns:
        Dict with lpips_temporal_real, lpips_temporal_gen,
        and optionally lpips_real_vs_gen_mean.
    """
    try:
        import lpips as lpips_lib
    except ImportError:
        raise ImportError("pip install lpips")

    if device is None:
        from videonoise.utils import get_device
        device = get_device()

    loss_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(device)
    loss_fn.eval()

    real_vids = _load_video_tensors(real_folder, max_frames, resize, device)
    gen_vids  = _load_video_tensors(gen_folder,  max_frames, resize, device)

    def _temporal_lpips(vids: list[torch.Tensor], label: str) -> float:
        """Mean LPIPS between consecutive frames, batched across all pairs in a video."""
        all_vals: list[float] = []
        with torch.no_grad():
            for v in tqdm(vids, desc=f"LPIPS temporal {label}", unit="video", leave=False):
                frames = _lpips_resize(v.float() * 2 - 1)  # (T, C, H, W) in [-1,1]
                if frames.shape[1] == 1:
                    frames = frames.expand(-1, 3, -1, -1)
                T = frames.shape[0]
                # collect consecutive pairs into batches
                for start in range(0, T - 1, batch_size):
                    end = min(start + batch_size, T - 1)
                    a = frames[start:end]          # (B, C, H, W)
                    b = frames[start + 1:end + 1]
                    d = loss_fn(a, b)              # (B, 1, 1, 1)
                    all_vals.extend(d.squeeze().tolist()
                                    if d.numel() > 1 else [d.item()])
        return float(np.mean(all_vals)) if all_vals else float("nan")

    result = {
        "lpips_temporal_real": _temporal_lpips(real_vids, "real"),
        "lpips_temporal_gen":  _temporal_lpips(gen_vids, "gen"),
    }

    # Cross-set: only if counts match (matched pairs scenario)
    if len(real_vids) == len(gen_vids) and real_vids:
        cross_vals: list[float] = []
        min_T = min(
            min(r.shape[0] for r in real_vids),
            min(g.shape[0] for g in gen_vids),
        )
        with torch.no_grad():
            for rv, gv in tqdm(zip(real_vids, gen_vids), desc="LPIPS real↔gen",
                                total=len(real_vids), unit="pair", leave=False):
                rf = _lpips_resize(rv[:min_T].float() * 2 - 1)
                gf = _lpips_resize(gv[:min_T].float() * 2 - 1)
                if rf.shape[1] == 1:
                    rf = rf.expand(-1, 3, -1, -1)
                if gf.shape[1] == 1:
                    gf = gf.expand(-1, 3, -1, -1)
                for start in range(0, min_T, batch_size):
                    end = min(start + batch_size, min_T)
                    d = loss_fn(rf[start:end], gf[start:end])
                    cross_vals.extend(d.squeeze().tolist()
                                      if d.numel() > 1 else [d.item()])
        result["lpips_real_vs_gen_mean"] = float(np.mean(cross_vals))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLIP score
# ─────────────────────────────────────────────────────────────────────────────

def compute_clip_score(
    gen_folder: str,
    prompts_file: Optional[str] = None,
    max_frames: int = 32,
    resize: Optional[tuple[int, int]] = None,
    device: Optional[torch.device] = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> dict:
    """
    Mean CLIP cosine similarity between video frames and text prompts.

    Requires: pip install open-clip-torch

    Looks for prompts in (in order):
      1. The file path given in *prompts_file* (JSON dict {name: prompt} or CSV)
      2. captions.csv in gen_folder  (columns: filename, caption)
      3. prompts.json in gen_folder  (dict {stem: prompt})

    If no prompts file is found, returns {"clip_score_mean": nan, "skipped": True}.

    Returns:
        Dict with clip_score_mean, clip_score_std, n_videos.
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError("pip install open-clip-torch")

    if device is None:
        from videonoise.utils import get_device
        device = get_device()

    # ── load prompts ──────────────────────────────────────────────────────────
    prompts: dict[str, str] = {}

    def _try_load(path: str) -> dict[str, str]:
        p = Path(path)
        if not p.exists():
            return {}
        if p.suffix == ".json":
            import json
            with open(p) as f:
                return json.load(f)
        if p.suffix == ".csv":
            import csv
            out = {}
            with open(p) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fn = row.get("filename") or row.get("name") or ""
                    cap = row.get("caption") or row.get("prompt") or ""
                    if fn and cap:
                        out[Path(fn).stem] = cap
            return out
        return {}

    if prompts_file:
        prompts = _try_load(prompts_file)
    if not prompts:
        prompts = _try_load(str(Path(gen_folder) / "captions.csv"))
    if not prompts:
        prompts = _try_load(str(Path(gen_folder) / "prompts.json"))

    if not prompts:
        print("  [clip_score] No prompts file found — skipping CLIP score.")
        return {"clip_score_mean": float("nan"), "clip_score_std": float("nan"), "skipped": True}

    # ── load model ────────────────────────────────────────────────────────────
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    from videonoise.io import load_video_folder
    videos = load_video_folder(gen_folder, max_frames=max_frames, resize=resize)

    scores = []
    with torch.no_grad():
        for stem, video in tqdm(videos, desc="CLIP score", unit="video"):
            prompt = prompts.get(stem) or prompts.get(Path(stem).stem)
            if not prompt:
                continue
            text_tok = tokenizer([prompt]).to(device)
            text_feat = model.encode_text(text_tok)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            # sample up to 8 frames evenly
            T = video.shape[0]
            indices = np.linspace(0, T - 1, min(8, T), dtype=int)
            frame_scores = []
            for i in indices:
                # convert frame to PIL-like tensor for preprocess
                frame = video[i]  # (C, H, W) in [0, 1]
                frame_pil = _tensor_to_pil(frame)
                img_tensor = preprocess(frame_pil).unsqueeze(0).to(device)
                img_feat = model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                cos_sim = (img_feat @ text_feat.T).item()
                frame_scores.append(float(cos_sim))
            if frame_scores:
                scores.append(float(np.mean(frame_scores)))

    if not scores:
        return {"clip_score_mean": float("nan"), "clip_score_std": float("nan"), "n_videos": 0}

    return {
        "clip_score_mean": float(np.mean(scores)),
        "clip_score_std":  float(np.std(scores)),
        "n_videos":        len(scores),
    }


def _tensor_to_pil(tensor: torch.Tensor):
    """Convert (C, H, W) float [0,1] tensor to PIL Image."""
    from PIL import Image
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if arr.shape[0] == 1:
        arr = arr[0]
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr.transpose(1, 2, 0))
