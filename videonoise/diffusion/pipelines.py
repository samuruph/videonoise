"""
Video diffusion model pipelines with unified API.

Supported models
----------------
svd         Stable Video Diffusion img2vid-xt  (image → video)
svd_t2v     SVD + SDXL conditioning            (text  → image → video)
modelscope  ModelScope Text-to-Video 1.7B      (text  → video)
cogvideox   CogVideoX-2b / 5b                  (text  → video)
wan         Wan2.1-T2V-1.3B                    (text  → video)

Custom noise initializations are injected at the latent level where
the diffusers API supports it.  SVD / svd_t2v have full support via
VAE encoding; other models use approximate latent shapes.
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from videonoise.noise.generators import NOISE_GENERATORS
from videonoise.utils import get_device, save_json


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "svd": {
        "default_id":  "stabilityai/stable-video-diffusion-img2vid-xt",
        "modality":    "image2video",
        "description": "Stable Video Diffusion img2vid-xt  (~14 GB, random conditioning image)",
    },
    "svd_t2v": {
        "default_id":  "stabilityai/stable-video-diffusion-img2vid-xt",
        "t2i_id":      "stabilityai/stable-diffusion-xl-base-1.0",
        "modality":    "text2video",
        "description": "SVD + SDXL: text → SDXL image → SVD video  (~21 GB total)",
    },
    "modelscope": {
        "default_id":  "damo-vilab/text-to-video-ms-1.7b",
        "modality":    "text2video",
        "description": "ModelScope Text-to-Video 1.7B  (~7 GB)",
    },
    "cogvideox": {
        "default_id":  "THUDM/CogVideoX-2b",
        "modality":    "text2video",
        "description": "CogVideoX-2b  49 frames at 8 fps  (~14 GB)",
    },
    "wan": {
        "default_id":  "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "modality":    "text2video",
        "description": "Wan2.1 T2V 1.3B  (~6 GB, requires diffusers>=0.32)",
    },
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pipeline(
    model_name: str,
    model_id: Optional[str],
    device: torch.device,
    t2i_model_id: Optional[str] = None,
):
    """Load the appropriate diffusers pipeline for *model_name*.

    For svd_t2v, also loads an SDXL text-to-image pipeline as conditioning backbone.
    Returns a plain pipeline for all other models, or a dict
    {"t2i": <sdxl_pipe>, "svd": <svd_pipe>} for svd_t2v.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {list(MODEL_REGISTRY)}")
    model_id = model_id or MODEL_REGISTRY[model_name]["default_id"]
    if model_name == "svd_t2v":
        t2i_id = t2i_model_id or MODEL_REGISTRY["svd_t2v"]["t2i_id"]
        return _load_svd_t2v(model_id, t2i_id, device)
    loaders = {
        "svd":        _load_svd,
        "modelscope": _load_modelscope,
        "cogvideox":  _load_cogvideox,
        "wan":        _load_wan,
    }
    return loaders[model_name](model_id, device)


def _load_svd(model_id, device):
    from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
    dtype = torch.float16 if str(device) == "cuda" else torch.float32
    print(f"Loading SVD ({model_id}) on {device} [{dtype}]...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype,
        variant="fp16" if str(device) == "cuda" else None,
    ).to(device)
    pipe.enable_attention_slicing(1)   # slice size = 1 (most aggressive)
    # pipe.enable_vae_slicing()        # decode one frame at a time (not supported in SVD)
    # pipe.enable_vae_tiling()         # tile large frames through VAE (not supported in SVD)
    return pipe


def _load_svd_t2v(svd_model_id, t2i_model_id, device):
    from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
    dtype = torch.float16 if str(device) == "cuda" else torch.float32
    print(f"Loading SDXL ({t2i_model_id}) for text → image conditioning on {device}...")
    t2i_pipe = StableDiffusionXLPipeline.from_pretrained(
        t2i_model_id, torch_dtype=dtype,
        variant="fp16" if str(device) == "cuda" else None,
        use_safetensors=True,
    ).to(device)
    t2i_pipe.enable_attention_slicing()
    print(f"Loading SVD ({svd_model_id}) for image → video on {device}...")
    svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
        svd_model_id, torch_dtype=dtype,
        variant="fp16" if str(device) == "cuda" else None,
    ).to(device)
    svd_pipe.enable_attention_slicing()
    return {"t2i": t2i_pipe, "svd": svd_pipe}


def _load_modelscope(model_id, device):
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    dtype = torch.float16 if str(device) == "cuda" else torch.float32
    print(f"Loading ModelScope ({model_id}) on {device} [{dtype}]...")
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.enable_attention_slicing()
    return pipe


def _load_cogvideox(model_id, device):
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
    dtype = torch.bfloat16 if str(device) == "cuda" else torch.float32
    print(f"Loading CogVideoX ({model_id}) on {device} [{dtype}]...")
    pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if str(device) == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def _load_wan(model_id, device):
    try:
        from diffusers.pipelines.wan.pipeline_wan import WanPipeline
    except ImportError:
        raise ImportError("Wan2.1 requires diffusers>=0.32: pip install -U diffusers")
    dtype = torch.bfloat16 if str(device) == "cuda" else torch.float32
    print(f"Loading Wan2.1 ({model_id}) on {device} [{dtype}]...")
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.enable_attention_slicing()
    return pipe


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_video(
    pipe,
    model_name: str,
    prompt_or_image,
    noise_type: str = "gaussian",
    noise_kwargs: Optional[dict] = None,
    num_frames: int = 16,
    num_steps: int = 25,
    seed: int = 42,
) -> list:
    """
    Unified generation for all supported models.

    Args:
        pipe:             Loaded pipeline from load_pipeline().
        model_name:       Key from MODEL_REGISTRY.
        prompt_or_image:  str prompt (text2video) or PIL.Image (image2video).
        noise_type:       Key from NOISE_GENERATORS.
        noise_kwargs:     Extra kwargs for the noise generator (alpha, sigma, …).
        num_frames:       Number of output frames.
        num_steps:        Denoising steps.
        seed:             Random seed.

    Returns:
        List of PIL Images (one per frame).
    """
    noise_kwargs = noise_kwargs or {}
    generators = {
        "svd":        _generate_svd,
        "svd_t2v":    _generate_svd_t2v,
        "modelscope": _generate_modelscope,
        "cogvideox":  _generate_cogvideox,
        "wan":        _generate_wan,
    }
    return generators[model_name](
        pipe, prompt_or_image, noise_type, noise_kwargs, num_frames, num_steps, seed
    )


def _make_latents(noise_type, shape, noise_kwargs, device, dtype):
    """Build a batched custom-noise latent tensor (batch dim = 1)."""
    noise = NOISE_GENERATORS[noise_type](shape, **noise_kwargs).to(device).to(dtype)
    return noise.unsqueeze(0)


def _generate_svd(pipe, image, noise_type, noise_kwargs, num_frames, num_steps, seed):
    device = pipe.device
    gen = torch.Generator(device=device).manual_seed(seed)

    if noise_type == "gaussian":
        return pipe(
            image=image, num_frames=num_frames, num_inference_steps=num_steps,
            decode_chunk_size=8, generator=gen,
        ).frames[0]

    # Encode conditioning image to get latent shape, then inject custom noise.
    with torch.no_grad():
        img_t = torch.from_numpy(np.array(image)).float() / 255.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).to(device)
        latent = pipe.vae.encode(img_t * 2 - 1).latent_dist.sample(gen)

    shape = (num_frames, latent.shape[1], latent.shape[2], latent.shape[3])
    latents = _make_latents(noise_type, shape, noise_kwargs, device, latent.dtype)
    return pipe(
        image=image, num_frames=num_frames, num_inference_steps=num_steps,
        decode_chunk_size=8, generator=gen, latents=latents,
    ).frames[0]


def _generate_svd_t2v(pipes, prompt, noise_type, noise_kwargs, num_frames, num_steps, seed):
    """Text → SDXL conditioning image → SVD video."""
    t2i_pipe = pipes["t2i"]
    svd_pipe = pipes["svd"]
    t2i_device = t2i_pipe.device

    print(f"  [svd_t2v] Generating conditioning image with SDXL: '{prompt[:60]}...'")
    gen_t2i = torch.Generator(device=t2i_device).manual_seed(seed)
    cond_image = t2i_pipe(
        prompt=prompt,
        num_inference_steps=30,
        generator=gen_t2i,
        height=576,
        width=1024,
    ).images[0]

    return _generate_svd(svd_pipe, cond_image, noise_type, noise_kwargs, num_frames, num_steps, seed)


def _generate_modelscope(pipe, prompt, noise_type, noise_kwargs, num_frames, num_steps, seed):
    # pipe.device is set by .to(device); fall back to unet for older diffusers
    device = getattr(pipe, "device", None) or pipe.unet.device
    gen = torch.Generator(device=str(device)).manual_seed(seed)
    kwargs = dict(
        prompt=prompt, num_frames=num_frames,
        num_inference_steps=num_steps, generator=gen,
    )
    if noise_type != "gaussian":
        # ModelScope latent layout: (batch, channels=4, frames, H//8, W//8)
        H, W = 256, 256
        shape = (4, num_frames, H // 8, W // 8)
        kwargs["latents"] = _make_latents(noise_type, shape, noise_kwargs, device, pipe.unet.dtype)

    out = pipe(**kwargs)
    return out.frames[0] if hasattr(out, "frames") else out.images


def _generate_cogvideox(pipe, prompt, noise_type, noise_kwargs, num_frames, num_steps, seed):
    gen = torch.Generator().manual_seed(seed)
    kwargs = dict(
        prompt=prompt, num_frames=num_frames,
        num_inference_steps=num_steps, generator=gen,
    )
    if noise_type != "gaussian":
        # CogVideoX 3D-VAE: spatial 8x + temporal 4x downsampling, 16 channels.
        # Latent layout expected by pipeline: (batch, frames, channels, H//8, W//8)
        H, W = 480, 720
        t = (num_frames - 1) // 4 + 1
        device = pipe.device
        dtype  = next(pipe.transformer.parameters()).dtype
        shape  = (16, t, H // 8, W // 8)
        latents = _make_latents(noise_type, shape, noise_kwargs, device, dtype)
        kwargs["latents"] = latents.permute(0, 2, 1, 3, 4)  # → (B, T, C, H, W)

    return pipe(**kwargs).frames[0]


def _generate_wan(pipe, prompt, noise_type, noise_kwargs, num_frames, num_steps, seed):
    gen = torch.Generator().manual_seed(seed)
    kwargs = dict(
        prompt=prompt, num_frames=num_frames,
        num_inference_steps=num_steps, generator=gen,
    )
    if noise_type != "gaussian":
        # Wan2.1 3D-VAE: spatial 8x + temporal 4x downsampling, 16 channels.
        H, W = 480, 832
        t = (num_frames - 1) // 4 + 1
        device = pipe.device
        dtype  = next(pipe.transformer.parameters()).dtype
        shape  = (16, t, H // 8, W // 8)
        kwargs["latents"] = _make_latents(noise_type, shape, noise_kwargs, device, dtype)

    return pipe(**kwargs).frames[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_frames_to_tensor(frames) -> torch.Tensor:
    """
    Convert pipeline output frames to a (T, C, H, W) float tensor in [0, 1].

    Accepts:
    - list of PIL Images  (any mode — converted to RGB)
    - list of numpy arrays  (H, W, C) uint8  → divided by 255
    - list of numpy arrays  (H, W, C) float  → kept as-is (already [0, 1])
    - numpy array  (T, H, W, C) in either range (e.g. TextToVideoSDPipeline output)
    """
    from PIL import Image as _PilImage
    arrs = []
    for f in frames:
        if isinstance(f, _PilImage.Image):
            arr = np.array(f.convert("RGB")).astype(np.float32) / 255.0
        else:
            arr = np.asarray(f)[..., :3]  # drop alpha if present
            if arr.dtype.kind == "f":
                arr = arr.astype(np.float32)        # already [0, 1]
            else:
                arr = arr.astype(np.float32) / 255.0  # uint8 → [0, 1]
        arrs.append(arr)
    return torch.from_numpy(np.stack(arrs, axis=0)).permute(0, 3, 1, 2)


def create_conditioning_image(H: int = 320, W: int = 576):
    """Create a random RGB PIL Image as a placeholder conditioning frame (SVD).

    Default is 320×576 (~18% of the original 576×1024 pixel count) which fits
    in MPS unified memory.  Pass H=576, W=1024 for full resolution on CUDA.
    """
    arr = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Console-script entry point  (vn-generate)
# ---------------------------------------------------------------------------

def _cli() -> None:
    import datetime
    from videonoise.io import save_video_cv2

    parser = argparse.ArgumentParser(
        description="Generate videos with a diffusion model and custom noise initialization",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY), default="svd",
        help=(
            "Model to use:\n" +
            "\n".join(f"  {k}: {v['description']}" for k, v in MODEL_REGISTRY.items())
        ),
    )
    parser.add_argument("--model_id",    default=None,
                        help="Override HuggingFace model ID (default: per-model)")
    parser.add_argument("--t2i_model_id", default=None,
                        help="Override SDXL model ID for svd_t2v (default: sdxl-base-1.0)")
    parser.add_argument("--prompt",     default="A beautiful natural landscape with flowing water",
                        help="Text prompt  [text2video models only]")
    parser.add_argument("--noise_type", choices=list(NOISE_GENERATORS), default="gaussian",
                        help="Noise initialization type")
    parser.add_argument("--alpha",      type=float, default=0.8,
                        help="AR(1) temporal coefficient  [ar1 noise]")
    parser.add_argument("--sigma",      type=float, default=2.0,
                        help="Spatial low-pass sigma  [spatial_lowpass noise]")
    parser.add_argument("--n",          type=int,   default=5,   help="Number of videos")
    parser.add_argument("--num_frames", type=int,   default=16,  help="Frames per video")
    parser.add_argument("--num_steps",  type=int,   default=25,  help="Denoising steps")
    parser.add_argument("--fps",        type=int,   default=8,   help="Output video FPS")
    parser.add_argument("--seed",       type=int,   default=42,  help="Base random seed")
    parser.add_argument(
        "--output", default="data/generated/",
        help=(
            "Root output directory (default: data/generated/).\n"
            "A subfolder  <model>_<noise_type>/  is always created inside it,\n"
            "e.g.  data/generated/modelscope_gaussian/"
        ),
    )
    args = parser.parse_args()

    device       = get_device()
    noise_kwargs = {"alpha": args.alpha, "sigma": args.sigma}
    run_key      = f"{args.model}_{args.noise_type}"
    out_dir      = Path(args.output) / run_key
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    model_id  = args.model_id or MODEL_REGISTRY[args.model]["default_id"]
    modality  = MODEL_REGISTRY[args.model]["modality"]

    print("=" * 60)
    print(f"  model      : {args.model}  ({model_id})")
    print(f"  noise      : {args.noise_type}", end="")
    if args.noise_type == "ar1":
        print(f"  alpha={args.alpha}", end="")
    if args.noise_type == "spatial_lowpass":
        print(f"  sigma={args.sigma}", end="")
    print()
    print(f"  frames     : {args.num_frames}  steps={args.num_steps}  fps={args.fps}")
    print(f"  n_videos   : {args.n}  base_seed={args.seed}")
    print(f"  output     : {out_dir}/")
    print("=" * 60)

    pipe     = load_pipeline(args.model, args.model_id, device,
                             t2i_model_id=args.t2i_model_id)
    videos   = []

    for i in range(args.n):
        seed = args.seed + i
        print(f"\n[{i+1}/{args.n}] seed={seed}")

        input_ = args.prompt if modality == "text2video" else create_conditioning_image()
        frames = generate_video(
            pipe, args.model, input_,
            noise_type=args.noise_type,
            noise_kwargs=noise_kwargs,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            seed=seed,
        )

        filename = f"video_{i:03d}_seed{seed}.mp4"
        out_path = str(out_dir / filename)
        save_video_cv2(pil_frames_to_tensor(frames), out_path, fps=args.fps)
        print(f"  → {out_path}")

        videos.append({
            "file":       filename,
            "seed":       seed,
            "index":      i,
            "prompt":     args.prompt if modality == "text2video" else None,
        })

    # Write a complete experiment record alongside the videos
    save_json(
        {
            "run_key":    run_key,
            "timestamp":  timestamp,
            "model":      args.model,
            "model_id":   model_id,
            "noise_type": args.noise_type,
            "noise_kwargs": {
                "alpha": args.alpha,
                "sigma": args.sigma,
            },
            "generation": {
                "num_frames": args.num_frames,
                "num_steps":  args.num_steps,
                "fps":        args.fps,
                "base_seed":  args.seed,
                "n_videos":   args.n,
            },
            "prompt": args.prompt if modality == "text2video" else None,
            "videos": videos,
        },
        str(out_dir / "metadata.json"),
    )
    print(f"\n  Metadata → {out_dir}/metadata.json")
    print(f"  Done. {args.n} video(s) saved to {out_dir}/")
