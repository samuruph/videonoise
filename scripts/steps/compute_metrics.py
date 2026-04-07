#!/usr/bin/env python3
"""
Step 02 — Compute video metrics for real and generated videos.

Reads   : data_real  and  gen_dir  (from config)
Writes  : results/<real_key>/metrics.json
          results/<gen_key>/metrics.json

Override any config value from the command line:
    python scripts/steps/compute_metrics.py --max_frames 16 --resize 256 256
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.config_loader import load_config
from videonoise.metrics import compute_metrics_for_folder


def main() -> None:
    cfg = load_config()

    real_json = Path(cfg.real_out) / "metrics.json"
    gen_json  = Path(cfg.gen_out)  / "metrics.json"

    print("=" * 64)
    print("  Step 02 — Compute metrics")
    print(f"  Real      : {cfg.data_real}")
    print(f"              → {real_json}")
    print(f"  Generated : {cfg.gen_dir_resolved}")
    print(f"              → {gen_json}")
    print("=" * 64)

    Path(cfg.real_out).mkdir(parents=True, exist_ok=True)
    Path(cfg.gen_out).mkdir(parents=True, exist_ok=True)

    print("\n  [1/2] Real videos...")
    if real_json.exists():
        print(f"  [skip] {real_json} already exists")
    else:
        compute_metrics_for_folder(cfg.data_real, str(real_json), cfg.max_frames, cfg.resize)

    print("\n  [2/2] Generated videos...")
    if gen_json.exists():
        print(f"  [skip] {gen_json} already exists")
    else:
        compute_metrics_for_folder(cfg.gen_dir_resolved, str(gen_json), cfg.max_frames, cfg.resize)

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
