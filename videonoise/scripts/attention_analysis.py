"""
Step 08 — Attention map extraction and analysis for video diffusion models.

For GENERATED videos: extracts attention during generation pass AND inversion pass.
For REAL videos: extracts attention during inversion pass only.

Writes to:
  results/<gen_key>/attention/<video_name>/
    attn_self_overview.png
    attn_cross_overview.png
    attention_stats.json
  results/<gen_key>/attention/inversion_comparison.png
  results/<real_key>/attention/<video_name>/
    attn_self_overview.png
    attention_stats.json

All settings come from scripts/config.yaml. Override any value from the CLI:
    python -m videonoise.scripts.attention_analysis --n_videos 3 --num_steps 5
"""
from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from videonoise.config_loader import build_parser, load_config


def _model_type(model: str) -> str:
    """Infer model_type ('unet' or 'transformer') from model name."""
    transformer_models = {"cogvideox", "cogvideo", "wan", "open-sora", "latte"}
    for name in transformer_models:
        if name in model.lower():
            return "transformer"
    return "unet"


def _load_pipeline(cfg):
    """Load the generation pipeline based on cfg.model."""
    from videonoise.utils import get_device
    device = get_device()

    model_lower = cfg.model.lower()
    dtype_str   = "fp16" if str(device) == "cuda" else "fp32"
    import torch
    dtype = torch.float16 if dtype_str == "fp16" else torch.float32

    if "svd" in model_lower or "stable-video" in model_lower:
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            cfg.model_id, torch_dtype=dtype
        ).to(device)
    elif "modelscope" in model_lower or "damo" in model_lower:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            cfg.model_id, torch_dtype=dtype
        ).to(device)
    elif "cogvideox" in model_lower or "cogvideo" in model_lower:
        from diffusers import CogVideoXPipeline
        pipe = CogVideoXPipeline.from_pretrained(
            cfg.model_id, torch_dtype=dtype
        ).to(device)
    else:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            cfg.model_id, torch_dtype=dtype
        ).to(device)

    # Disable attention slicing — required before hooking
    try:
        pipe.disable_attention_slicing()
    except Exception:
        pass

    return pipe, device


def _run_inversion_with_hooks(pipe, video, model_type: str, num_steps: int, device):
    """Run simple pixel inversion and capture attention during the noise schedule forward pass."""
    import torch
    from videonoise.analysis.attention import (
        register_attention_hooks, remove_hooks, aggregate_attention_maps,
    )
    from videonoise.noise.inversion import simple_pixel_inversion

    storage, handles = register_attention_hooks(pipe, model_type)
    try:
        # Pixel-domain inversion: run UNet/transformer forward on noisy latents
        # Use vae encode + UNet forward to get attention during inversion
        video_dev = video.to(device)
        if hasattr(pipe, "vae"):
            import torch.nn.functional as F
            frames = video_dev.float() * 2 - 1
            T, C, H, W = frames.shape
            with torch.no_grad():
                latents = torch.cat([
                    pipe.vae.encode(frames[t:t + 1]).latent_dist.sample()
                    * pipe.vae.config.scaling_factor
                    for t in tqdm(range(T), desc="VAE encode", unit="frame", leave=False)
                ])
            # Denoise one step to trigger attention hooks
            scheduler = pipe.scheduler
            scheduler.set_timesteps(num_steps)
            t_mid = scheduler.timesteps[len(scheduler.timesteps) // 2]
            noise = torch.randn_like(latents)
            noisy = scheduler.add_noise(latents, noise, t_mid.unsqueeze(0))
            encoder_hidden = None
            if hasattr(pipe, "text_encoder"):
                dummy_tokens = torch.zeros(1, 77, dtype=torch.long, device=device)
                with torch.no_grad():
                    encoder_hidden = pipe.text_encoder(dummy_tokens)[0]
            with torch.no_grad():
                backbone = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
                if backbone is not None:
                    inp = noisy.unsqueeze(0) if noisy.ndim == 4 else noisy
                    kwargs = {"timestep": t_mid}
                    if encoder_hidden is not None:
                        kwargs["encoder_hidden_states"] = encoder_hidden
                    try:
                        backbone(inp, **kwargs)
                    except Exception:
                        pass
    finally:
        remove_hooks(handles)

    return aggregate_attention_maps(storage)


def _run_generation_with_hooks(pipe, cfg, model_type: str, device):
    """Run one generation pass and capture attention maps."""
    import torch
    from videonoise.analysis.attention import (
        register_attention_hooks, remove_hooks, aggregate_attention_maps,
    )

    storage, handles = register_attention_hooks(pipe, model_type)
    try:
        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        prompt = cfg.active_prompt
        H, W = cfg.resize  # (W, H) in config but use for generation
        kwargs = dict(
            num_inference_steps=cfg.num_steps,
            generator=generator,
        )
        if hasattr(pipe, "text_encoder"):
            kwargs["prompt"] = prompt
        try:
            pipe(**kwargs)
        except Exception as e:
            print(f"    [warn] Generation forward pass failed: {e}")
    finally:
        remove_hooks(handles)

    return aggregate_attention_maps(storage)


def main() -> None:
    parser = build_parser("Step 08 — Attention analysis")
    args = parser.parse_args()
    cfg  = load_config(args)

    from videonoise.analysis.attention import (
        plot_attention_maps, compute_attention_stats, compare_inversion_attention,
    )
    from videonoise.io import load_video_folder
    from videonoise.utils import get_device, save_json

    device = get_device()
    model_type = _model_type(cfg.model)

    print("=" * 64)
    print("  Step 08 — Attention analysis")
    print(f"  Model      : {cfg.model}  ({model_type})")
    print(f"  Real       : {cfg.data_real}  →  {cfg.real_out}/attention/")
    print(f"  Generated  : {cfg.gen_dir_resolved}  →  {cfg.gen_out}/attention/")
    print("=" * 64)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    if cfg.model_id is None:
        print("\n  [skip] model_id not set in config — cannot load pipeline.")
        print("         Set model_id in scripts/config.yaml to run attention analysis.")
        return

    print("\n  Loading pipeline...")
    try:
        pipe, device = _load_pipeline(cfg)
    except Exception as e:
        print(f"\n  [skip] Failed to load pipeline: {e}")
        return

    real_videos = load_video_folder(
        cfg.data_real, max_frames=cfg.max_frames, resize=cfg.resize
    )[:args.n_videos]

    gen_videos = load_video_folder(
        cfg.gen_dir_resolved, max_frames=cfg.max_frames, resize=cfg.resize
    )[:args.n_videos]

    all_real_stats: dict[str, dict] = {}
    all_gen_inv_stats: dict[str, dict] = {}

    # ── Real videos: inversion only ───────────────────────────────────────────
    if real_videos:
        print(f"\n  [A] Real videos — inversion attention ({len(real_videos)} videos)")
        for name, video in tqdm(real_videos, desc="Real attention", unit="video"):
            print(f"    {name}...")
            agg = _run_inversion_with_hooks(pipe, video, model_type, cfg.num_steps, device)
            plot_attention_maps(agg, cfg.real_out, "real-inversion", name)
            stats = compute_attention_stats(agg)
            all_real_stats[name] = stats

        save_json({"per_video": all_real_stats},
                  str(Path(cfg.real_out) / "attention" / "attention_stats.json"))

    # ── Generated videos: inversion + generation ─────────────────────────────
    if gen_videos:
        print(f"\n  [B] Generated videos — inversion attention ({len(gen_videos)} videos)")
        for name, video in tqdm(gen_videos, desc="Gen attention", unit="video"):
            print(f"    {name}...")
            agg = _run_inversion_with_hooks(pipe, video, model_type, cfg.num_steps, device)
            plot_attention_maps(agg, cfg.gen_out, "gen-inversion", name)
            stats = compute_attention_stats(agg)
            all_gen_inv_stats[name] = stats

        save_json({"per_video": all_gen_inv_stats},
                  str(Path(cfg.gen_out) / "attention" / "attention_inv_stats.json"))

        print(f"\n  [C] Generated videos — generation attention")
        all_gen_gen_stats: dict[str, dict] = {}
        agg_gen = _run_generation_with_hooks(pipe, cfg, model_type, device)
        if agg_gen:
            plot_attention_maps(agg_gen, cfg.gen_out, "gen-generation", "generation_pass",
                                prompt=cfg.active_prompt)
            all_gen_gen_stats["generation_pass"] = compute_attention_stats(agg_gen)
            save_json({"per_video": all_gen_gen_stats},
                      str(Path(cfg.gen_out) / "attention" / "attention_gen_stats.json"))

    # ── Compare real inversion vs generated inversion ─────────────────────────
    if all_real_stats and all_gen_inv_stats:
        print("\n  [D] Comparing real vs generated inversion attention...")
        # Aggregate across videos: mean stats per layer
        def _mean_stats(per_vid: dict[str, dict]) -> dict:
            all_keys = set()
            for v in per_vid.values():
                all_keys.update(v.keys())
            result = {}
            import numpy as np
            for k in sorted(all_keys):
                vals_entropy = [
                    per_vid[n][k]["entropy"]
                    for n in per_vid if k in per_vid[n]
                ]
                result[k] = {
                    "entropy": float(np.mean(vals_entropy)) if vals_entropy else float("nan")
                }
            return result

        real_agg = _mean_stats(all_real_stats)
        gen_agg  = _mean_stats(all_gen_inv_stats)
        compare_inversion_attention(real_agg, gen_agg, cfg.gen_out)

    print("\n  Attention analysis complete.")


if __name__ == "__main__":
    main()
