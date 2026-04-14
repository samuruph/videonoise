"""
Step 05 — Noise initialization ablation.

Generates videos for every noise type (gaussian, ar1, spatial_lowpass, blue,
perlin) then computes metrics for each condition.

Writes  : data/generated/<model>_<noise>/
          results/<model>_<noise>__<settings>/metrics.json

All settings come from scripts/config.yaml. Override any value from the CLI:
    python -m videonoise.scripts.noise_ablation --model wan --n_videos 20
"""
import datetime
import sys
from pathlib import Path

from tqdm import tqdm

from videonoise.config_loader import load_config
from videonoise.metrics import compute_metrics_for_folder

NOISE_TYPES = ["gaussian", "ar1", "spatial_lowpass", "blue", "perlin"]


def _resolve_gen_model(cfg) -> tuple[str, str | None]:
    """Return (model_name, model_id_override) for the generation pipeline.

    config.yaml may use  model: hf  as a naming prefix for downloaded runs.
    When that is not a valid MODEL_REGISTRY key, fall back to model_id (e.g.
    "wan") as the model name.  Only forward model_id when it looks like a full
    HuggingFace repo path (contains "/"), otherwise let the registry default
    provide the correct repo ID.
    """
    from videonoise.diffusion.pipelines import MODEL_REGISTRY

    if cfg.model in MODEL_REGISTRY:
        gen_model = cfg.model
    elif cfg.model_id and cfg.model_id in MODEL_REGISTRY:
        gen_model = cfg.model_id
    else:
        raise ValueError(
            f"Cannot resolve a valid generation model from config: "
            f"model={cfg.model!r}, model_id={cfg.model_id!r}. "
            f"Set 'model' in config.yaml to one of: {list(MODEL_REGISTRY)}"
        )

    # Only pass model_id override when it is a full "owner/repo" path
    model_id_override = cfg.model_id if (cfg.model_id and "/" in cfg.model_id) else None
    return gen_model, model_id_override


def main() -> None:
    from videonoise.diffusion.pipelines import (
        MODEL_REGISTRY,
        generate_video,
        load_pipeline,
        pil_frames_to_tensor,
    )
    from videonoise.io import save_video_cv2
    from videonoise.noise.generators import _cli as _noise
    from videonoise.utils import get_device, save_json

    cfg = load_config()

    gen_model, model_id_override = _resolve_gen_model(cfg)
    run_prefix = cfg.model  # e.g. "wan" or "hf"
    modality = MODEL_REGISTRY[gen_model]["modality"]
    noise_kwargs = {"alpha": cfg.alpha, "sigma": cfg.sigma}

    print("=" * 64)
    print("  Step 05 — Noise initialization ablation")
    print(f"  Generation model : {gen_model}")
    print(f"  Run-key prefix   : {run_prefix}")
    print(f"  Noises           : {', '.join(NOISE_TYPES)}")
    print("=" * 64)

    # ------------------------------------------------------------------ #
    # Load the pipeline ONCE and reuse it across all noise conditions.    #
    # ------------------------------------------------------------------ #
    device = get_device()
    pipe = load_pipeline(gen_model, model_id_override, device,
                         t2i_model_id=cfg.t2i_model_id if cfg.t2i_model_id else None)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    model_id_used = model_id_override or MODEL_REGISTRY[gen_model]["default_id"]

    for noise in tqdm(NOISE_TYPES, desc="Noise ablation", unit="noise_type"):
        run_key = f"{run_prefix}_{noise}"
        run_out = Path(cfg.results) / (run_key + "__" + cfg.settings_suffix)
        gen_dir = Path(cfg.data_gen.rstrip("/")) / run_key

        gen_dir.mkdir(parents=True, exist_ok=True)
        run_out.mkdir(parents=True, exist_ok=True)

        print(f"\n  --- {run_key} ---")
        print(f"  Generating  →  {gen_dir}/")

        videos = []
        for i in range(cfg.n_videos):
            seed = cfg.seed + i
            print(f"  [{i+1}/{cfg.n_videos}] seed={seed}  noise={noise}")

            input_ = cfg.active_prompt if modality == "text2video" else None
            frames = generate_video(
                pipe, gen_model, input_,
                noise_type=noise,
                noise_kwargs=noise_kwargs,
                num_frames=cfg.num_frames,
                num_steps=cfg.num_steps,
                seed=seed,
            )

            filename = f"video_{i:03d}_seed{seed}.mp4"
            out_path = str(gen_dir / filename)
            save_video_cv2(pil_frames_to_tensor(frames), out_path, fps=8)
            print(f"    → {out_path}")

            videos.append({
                "file":   filename,
                "seed":   seed,
                "index":  i,
                "prompt": cfg.active_prompt if modality == "text2video" else None,
            })

        save_json(
            {
                "run_key":      run_key,
                "timestamp":    timestamp,
                "model":        gen_model,
                "model_id":     model_id_used,
                "noise_type":   noise,
                "noise_kwargs": noise_kwargs,
                "generation": {
                    "num_frames": cfg.num_frames,
                    "num_steps":  cfg.num_steps,
                    "fps":        8,
                    "base_seed":  cfg.seed,
                    "n_videos":   cfg.n_videos,
                },
                "prompt":  cfg.active_prompt if modality == "text2video" else None,
                "videos":  videos,
            },
            str(gen_dir / "metadata.json"),
        )

        metrics_json = run_out / "metrics.json"
        print(f"  Metrics     →  {metrics_json}")
        if not metrics_json.exists():
            compute_metrics_for_folder(str(gen_dir), str(metrics_json),
                                       cfg.max_frames, cfg.resize)

    # Noise shape visualisation (no model required)
    noise_out = cfg.results + "noise_init/"
    print(f"\n  Visualising noise shapes  →  {noise_out}")
    sys.argv = [
        "vn-noise-init", "--type", "all",
        "--shape", "16", "4", "32", "32",
        "--alpha", str(cfg.alpha),
        "--output", noise_out,
    ]
    _noise()


if __name__ == "__main__":
    main()
