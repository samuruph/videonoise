#!/usr/bin/env python3
"""
Step 05 — Noise initialization ablation.

Generates videos for every noise type, then computes metrics for each.

Writes  : data/generated/<model>_<noise>/
          results/<model>_<noise>__<settings>/metrics.json

Override any config value from the command line:
    python scripts/steps/noise_ablation.py --model wan --n_videos 20
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.config_loader import load_config
from videonoise.metrics import compute_metrics_for_folder

NOISE_TYPES = ["gaussian", "ar1", "spatial_lowpass", "blue", "perlin"]


def main() -> None:
    cfg = load_config()

    print("=" * 64)
    print("  Step 05 — Noise initialization ablation")
    print(f"  Model     : {cfg.model}")
    print(f"  Noises    : {', '.join(NOISE_TYPES)}")
    print("=" * 64)

    for noise in NOISE_TYPES:
        run_key  = f"{cfg.model}_{noise}"
        run_out  = cfg.results + run_key + "__" + cfg.settings_suffix + "/"
        gen_dir  = cfg.data_gen.rstrip("/") + "/" + run_key + "/"

        print(f"\n  --- {run_key} ---")
        print(f"  Generating  →  {gen_dir}")

        sys.argv = [
            "vn-generate",
            "--model",      cfg.model,
            "--prompt",     cfg.active_prompt,
            "--noise_type", noise,
            "--alpha",      str(cfg.alpha),
            "--sigma",      str(cfg.sigma),
            "--n",          str(cfg.n_videos),
            "--num_frames", str(cfg.num_frames),
            "--num_steps",  str(cfg.num_steps),
            "--seed",       str(cfg.seed),
            "--output",     cfg.data_gen,
        ] + (["--model_id", cfg.model_id] if cfg.model_id else []) \
          + (["--t2i_model_id", cfg.t2i_model_id] if cfg.t2i_model_id else [])

        from videonoise.diffusion.pipelines import _cli as _gen
        _gen()

        metrics_json = Path(run_out) / "metrics.json"
        Path(run_out).mkdir(parents=True, exist_ok=True)
        print(f"  Metrics     →  {metrics_json}")
        if metrics_json.exists():
            print(f"  [skip] {metrics_json} already exists")
        else:
            compute_metrics_for_folder(gen_dir, str(metrics_json), cfg.max_frames, cfg.resize)

    # Noise shape visualisation
    noise_out = cfg.results + "noise_init/"
    print(f"\n  Visualising noise shapes  →  {noise_out}")
    sys.argv = [
        "vn-noise-init", "--type", "all",
        "--shape", "16", "4", "32", "32",
        "--alpha", str(cfg.alpha),
        "--output", noise_out,
    ]
    from videonoise.noise.generators import _cli as _noise
    _noise()

    print(f"\n  Done. Each run saved to {cfg.results}<model>_<noise>__<settings>/")


if __name__ == "__main__":
    main()
