#!/usr/bin/env python3
"""
Intermediate visualisation — works after step 02 alone.

Produces 4–5 PNG panels per dataset in results/<key>/plots/:

  01_temporal_coherence.png  — TCS, Pearson, Spearman, PSNR
  02_motion_dynamics.png     — optical-flow stats + temporal ACF
  03_spectral.png            — power spectra, slope, R², 3D energy ratio
  04_frame_content.png       — frame-level pixel statistics
  05_noise_stats.png         — inverted-noise stats (only when noise_stats.json exists)

Step-specific flag:
    --mode  real | gen | both   (default: both)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.config_loader import build_parser, load_config
from videonoise.analysis.plots import plot_all_single, print_single_table
from videonoise.utils import load_json


def _viz_one(run_dir: str, label: str) -> None:
    metrics_json = Path(run_dir) / "metrics.json"
    noise_json   = Path(run_dir) / "noise_stats.json"
    plots_dir    = str(Path(run_dir) / "plots")

    if not metrics_json.exists():
        print(f"  [skip] {metrics_json} not found — run step 02 first")
        return

    print(f"  Visualising {label}  →  {plots_dir}/")
    results = load_json(str(metrics_json))
    noise   = load_json(str(noise_json)) if noise_json.exists() else None

    print_single_table(results, label)
    plot_all_single(results, label, plots_dir, noise_results=noise)


def main() -> None:
    parser = build_parser("Visualise metrics")
    parser.add_argument(
        "--mode", choices=["real", "gen", "both"], default="both",
        help="Which dataset(s) to visualise (default: both)",
    )
    args = parser.parse_args()
    cfg  = load_config(args)

    print("=" * 64)
    print(f"  Visualisation  (mode: {args.mode})")
    print("=" * 64)

    if args.mode in ("real", "both"):
        _viz_one(cfg.real_out, cfg.real_key)
    if args.mode in ("gen", "both"):
        _viz_one(cfg.gen_out, cfg.gen_key)

    print("\n  Done. Plots saved inside each results/<run_key>/plots/")


if __name__ == "__main__":
    main()
