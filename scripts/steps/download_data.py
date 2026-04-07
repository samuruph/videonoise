#!/usr/bin/env python3
"""
Step 00 — Download real reference videos.

Step-specific flag:
    --dataset  synthetic | davis | vbench2 | ucf101   (default: synthetic)

Override any config value from the command line:
    python scripts/steps/download_data.py --dataset davis --n_videos 50
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.config_loader import build_parser, load_config


def main() -> None:
    parser = build_parser("Download real reference videos (step 00)")
    parser.add_argument(
        "--dataset", default="synthetic",
        choices=["synthetic", "davis", "vbench2", "ucf101"],
        help="Dataset to download (default: synthetic)",
    )
    args = parser.parse_args()
    cfg  = load_config(args)

    print("=" * 64)
    print("  Step 00 — Download real reference videos")
    print(f"  Dataset : {args.dataset}")
    print(f"  Output  : {cfg.data_real}")
    print("=" * 64)

    sys.argv = [
        "vn-download",
        "--dataset", args.dataset,
        "--output",  cfg.data_real,
        "--n",       str(cfg.n_videos),
    ]
    from videonoise.scripts.download_data import main as _dl
    _dl()


if __name__ == "__main__":
    main()
