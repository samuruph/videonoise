"""
Step 04 — Spatio-temporal analysis (PCA, 3D spectrum) for real and generated videos.

Reads   : data_real  and  gen_dir  (from config)
Writes  : results/<real_key>/spatiotemporal/
          results/<gen_key>/spatiotemporal/

All settings come from scripts/config.yaml. Override any value from the CLI:
    python -m videonoise.scripts.spatiotemporal_analysis --max_frames 16 --resize 256 256
"""
from pathlib import Path

from videonoise.config_loader import load_config
from videonoise.analysis.spatiotemporal import run_spatiotemporal_folder


def main() -> None:
    cfg = load_config()

    real_st = Path(cfg.real_out) / "spatiotemporal"
    gen_st  = Path(cfg.gen_out)  / "spatiotemporal"

    print("=" * 64)
    print("  Step 04 — Spatio-temporal analysis")
    print(f"  Real      : {cfg.data_real}  →  {real_st}/")
    print(f"  Generated : {cfg.gen_dir_resolved}  →  {gen_st}/")
    print("=" * 64)

    print("\n  [1/2] Real videos...")
    run_spatiotemporal_folder(
        cfg.data_real, str(real_st),
        max_frames=cfg.max_frames, resize=cfg.resize, use_pixel_pca=True,
    )

    print("\n  [2/2] Generated videos...")
    run_spatiotemporal_folder(
        cfg.gen_dir_resolved, str(gen_st),
        max_frames=cfg.max_frames, resize=cfg.resize, use_pixel_pca=True,
    )


if __name__ == "__main__":
    main()
