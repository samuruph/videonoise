"""
Load scripts/config.yaml and resolve all experiment paths.

All CLI argument definitions live here. To add a new config field:
  1. Add it to config.yaml
  2. Add it to the Config dataclass below
  3. Add it to build_parser() with the right type/help
  4. Apply the override in _apply_overrides()
  That's it — no other file needs to change.

Usage
-----
In a script with no step-specific args:

    from videonoise.config_loader import load_config
    cfg = load_config()

In a script that adds its own flags:

    from videonoise.config_loader import build_parser, load_config
    parser = build_parser("My step")
    parser.add_argument("--my_flag", ...)
    args = parser.parse_args()
    cfg  = load_config(args)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml


@dataclass
class Config:
    # ── raw ──────────────────────────────────────────────────────────────────
    data_real: str
    data_gen: str
    gen_dir: Optional[str]
    results: str
    max_frames: int
    resize: Tuple[int, int]         # (W, H)
    model: str
    model_id: Optional[str]
    t2i_model_id: Optional[str]
    noise_type: str
    alpha: float
    sigma: float
    n_videos: int
    num_frames: int
    num_steps: int
    seed: int
    active_prompt: str

    # ── derived ───────────────────────────────────────────────────────────────
    settings_suffix: str    # e.g. "32f_512x768"
    real_key: str           # e.g. "deepaction_all__real__32f_512x768"
    gen_key: str            # e.g. "generated_CogVideoX5B__32f_512x768"
    gen_dir_resolved: str   # actual path to generated videos (trailing /)
    real_out: str           # results/<real_key>/
    gen_out: str            # results/<gen_key>/


def build_parser(description: str = "") -> argparse.ArgumentParser:
    """Return a parser with every config key as an optional CLI override.

    All args default to None — only explicitly passed values override YAML.
    Step scripts may call parser.add_argument() to add step-specific flags.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="scripts/config.yaml",
                   help="Path to config YAML (default: scripts/config.yaml)")

    # ── paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_real", help="Real videos directory")
    p.add_argument("--data_gen",  help="Generated videos root directory")
    p.add_argument("--gen_dir",   help="Override: exact generated videos directory")
    p.add_argument("--results",   help="Results root directory")

    # ── video loading ─────────────────────────────────────────────────────────
    p.add_argument("--max_frames", type=int, help="Max frames to load per video")
    p.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                   help="Resize frames: W H")

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model",        help="Model key: svd | modelscope | cogvideox | wan")
    p.add_argument("--model_id",     help="HuggingFace model ID override")
    p.add_argument("--t2i_model_id", help="SDXL model ID override (svd_t2v only)")

    # ── noise ─────────────────────────────────────────────────────────────────
    p.add_argument("--noise_type", help="gaussian | ar1 | spatial_lowpass | blue | perlin")
    p.add_argument("--alpha", type=float, help="AR(1) temporal coefficient [ar1]")
    p.add_argument("--sigma", type=float, help="Spatial low-pass sigma [spatial_lowpass]")

    # ── generation ───────────────────────────────────────────────────────────
    p.add_argument("--n_videos",   type=int, help="Number of videos to generate")
    p.add_argument("--num_frames", type=int, help="Frames per video")
    p.add_argument("--num_steps",  type=int, help="Denoising steps")
    p.add_argument("--seed",       type=int, help="Base random seed")
    p.add_argument("--prompt",               help="Text prompt (text-to-video models)")

    return p


def _data_key(path_str: str) -> str:
    """Short human-readable key from a data path.

      data/matched_pairs/deepaction_all/real  →  deepaction_all__real
      data/real/DAVIS/Videos/480p             →  Videos__480p
    """
    parts = [p for p in Path(path_str.rstrip("/")).parts if p not in (".", "")]
    meaningful = [p for p in parts if p != "data"]
    if len(meaningful) >= 2:
        return "__".join(meaningful[-2:])
    return meaningful[0] if meaningful else "data"


def _apply_overrides(c: dict, args: argparse.Namespace) -> None:
    """Copy non-None CLI args into the raw config dict."""
    mapping = {
        "data_real": "data_real", "data_gen": "data_gen", "gen_dir": "gen_dir",
        "results": "results", "max_frames": "max_frames", "model": "model",
        "model_id": "model_id", "t2i_model_id": "t2i_model_id",
        "noise_type": "noise_type", "alpha": "alpha", "sigma": "sigma",
        "n_videos": "n_videos", "num_frames": "num_frames",
        "num_steps": "num_steps", "seed": "seed", "prompt": "active_prompt",
    }
    for arg_key, cfg_key in mapping.items():
        val = getattr(args, arg_key, None)
        if val is not None:
            c[cfg_key] = val
    if getattr(args, "resize", None) is not None:
        c["resize"] = list(args.resize)


def load_config(args: Optional[argparse.Namespace] = None) -> Config:
    """Load config.yaml and apply any CLI overrides.

    Pass a pre-parsed Namespace when the calling script added its own args.
    If args is None, build_parser().parse_args() is called automatically.
    """
    if args is None:
        args = build_parser().parse_args()

    with open(args.config) as f:
        c = yaml.safe_load(f)

    _apply_overrides(c, args)

    resize = c.get("resize", [128, 128])
    settings_suffix = f"{c['max_frames']}f_{resize[0]}x{resize[1]}"

    gen_dir = c.get("gen_dir") or None
    if gen_dir:
        gen_dir_resolved = gen_dir.rstrip("/") + "/"
        gen_base = Path(gen_dir.rstrip("/")).name
    else:
        gen_base = f"{c['model']}_{c['noise_type']}"
        gen_dir_resolved = c["data_gen"].rstrip("/") + "/" + gen_base + "/"

    results = c["results"].rstrip("/") + "/"
    real_key = _data_key(c["data_real"]) + "__" + settings_suffix
    gen_key  = gen_base + "__" + settings_suffix

    return Config(
        data_real=c["data_real"], data_gen=c["data_gen"], gen_dir=gen_dir,
        results=results, max_frames=int(c["max_frames"]), resize=tuple(resize),
        model=c["model"], model_id=c.get("model_id") or None,
        t2i_model_id=c.get("t2i_model_id") or None, noise_type=c["noise_type"],
        alpha=float(c["alpha"]), sigma=float(c["sigma"]),
        n_videos=int(c["n_videos"]), num_frames=int(c["num_frames"]),
        num_steps=int(c["num_steps"]), seed=int(c["seed"]),
        active_prompt=c["active_prompt"],
        settings_suffix=settings_suffix, real_key=real_key, gen_key=gen_key,
        gen_dir_resolved=gen_dir_resolved,
        real_out=results + real_key + "/",
        gen_out=results + gen_key + "/",
    )
