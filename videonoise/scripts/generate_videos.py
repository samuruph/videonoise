"""
Step 01 — Generate videos with a diffusion model.

All settings come from scripts/config.yaml. Override any value from the CLI:
    python -m videonoise.scripts.generate_videos --model modelscope --noise_type ar1 --n_videos 5
"""
import sys
from pathlib import Path

from videonoise.config_loader import load_config


def main() -> None:
    cfg = load_config()
    run_key = f"{cfg.model}_{cfg.noise_type}"
    out_dir = Path(cfg.data_gen) / run_key

    print("=" * 64)
    print("  Step 01 — Generate videos")
    print(f"  Model     : {cfg.model}  (id: {cfg.model_id or 'default'})")
    print(f"  Noise     : {cfg.noise_type}")
    print(f"  Videos    : {cfg.n_videos}  x  {cfg.num_frames} frames  ({cfg.num_steps} steps)")
    print(f"  Output    : {out_dir}/")
    print("=" * 64)

    sys.argv = [
        "vn-generate",
        "--model",      cfg.model,
        "--prompt",     cfg.active_prompt,
        "--noise_type", cfg.noise_type,
        "--alpha",      str(cfg.alpha),
        "--sigma",      str(cfg.sigma),
        "--n",          str(cfg.n_videos),
        "--num_frames", str(cfg.num_frames),
        "--num_steps",  str(cfg.num_steps),
        "--seed",       str(cfg.seed),
        "--output",     cfg.data_gen,
    ] + (["--model_id",     cfg.model_id]     if cfg.model_id     else []) \
      + (["--t2i_model_id", cfg.t2i_model_id] if cfg.t2i_model_id else [])

    from videonoise.diffusion.pipelines import _cli
    _cli()


if __name__ == "__main__":
    main()
