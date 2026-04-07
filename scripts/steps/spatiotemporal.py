#!/usr/bin/env python3
"""
Step 04 — Spatio-temporal analysis (PCA, 3D spectrum) for real and generated videos.

Reads   : data_real  and  gen_dir  (from config)
Writes  : results/<real_key>/spatiotemporal/
          results/<gen_key>/spatiotemporal/

Override any config value from the command line:
    python scripts/steps/spatiotemporal.py --max_frames 16 --resize 256 256
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.config_loader import load_config
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

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
