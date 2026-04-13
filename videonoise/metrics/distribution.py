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

def compute_fvd(
    real_folder: str,
    gen_folder: str,
    max_frames: int = 32,
    resize: Optional[tuple[int, int]] = None,
    device: Optional[torch.device] = None,
) -> float:
    """
    Fréchet Video Distance between real and generated video sets.

    Requires: pip install frechet-video-distance

    Returns:
        FVD scalar (lower = more similar distributions).
    """
    try:
        from frechet_video_distance import frechet_video_distance
    except ImportError:
        raise ImportError(
            "pip install frechet-video-distance\n"
            "See: https://github.com/google-research/google-research/tree/master/frechet_video_distance"
        )

    if device is None:
        from videonoise.utils import get_device
        device = get_device()

    real_vids = _load_video_tensors(real_folder, max_frames, resize, device)
    gen_vids  = _load_video_tensors(gen_folder,  max_frames, resize, device)

    if len(real_vids) < 2 or len(gen_vids) < 2:
        raise ValueError(f"FVD requires ≥ 2 videos per set (got {len(real_vids)} real, {len(gen_vids)} gen)")

    real_uint8 = _frames_to_uint8(real_vids)  # (N, T, C, H, W)
    gen_uint8  = _frames_to_uint8(gen_vids)

    fvd = frechet_video_distance(real_uint8, gen_uint8)
    return float(fvd)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-set LPIPS
# ─────────────────────────────────────────────────────────────────────────────

def compute_lpips_between_sets(
    real_folder: str,
    gen_folder: str,
    max_frames: int = 32,
    resize: Optional[tuple[int, int]] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Compute LPIPS metrics:
    - lpips_temporal_real: mean LPIPS between consecutive real frames (temporal consistency)
    - lpips_temporal_gen:  same for generated
    - lpips_real_vs_gen:   mean LPIPS between corresponding real/gen frames at each timestep
                           (only computed if real and gen video counts match)

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

    def _temporal_lpips(vids: list[torch.Tensor]) -> float:
        vals = []
        with torch.no_grad():
            for v in vids:
                frames = v.float() * 2 - 1
                if frames.shape[1] == 1:
                    frames = frames.expand(-1, 3, -1, -1)
                for t in range(frames.shape[0] - 1):
                    d = loss_fn(frames[t:t + 1], frames[t + 1:t + 2]).item()
                    vals.append(float(d))
        return float(np.mean(vals)) if vals else float("nan")

    result = {
        "lpips_temporal_real": _temporal_lpips(real_vids),
        "lpips_temporal_gen":  _temporal_lpips(gen_vids),
    }

    # Cross-set: only if counts match (matched pairs scenario)
    if len(real_vids) == len(gen_vids) and real_vids:
        cross_vals = []
        min_T = min(
            min(r.shape[0] for r in real_vids),
            min(g.shape[0] for g in gen_vids),
        )
        with torch.no_grad():
            for rv, gv in zip(real_vids, gen_vids):
                rf = rv[:min_T].float() * 2 - 1
                gf = gv[:min_T].float() * 2 - 1
                if rf.shape[1] == 1:
                    rf = rf.expand(-1, 3, -1, -1)
                if gf.shape[1] == 1:
                    gf = gf.expand(-1, 3, -1, -1)
                for t in range(min_T):
                    d = loss_fn(rf[t:t + 1], gf[t:t + 1]).item()
                    cross_vals.append(float(d))
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
        for stem, video in videos:
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
