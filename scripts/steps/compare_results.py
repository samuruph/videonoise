#!/usr/bin/env python3
"""
Step 06 — Compare real vs. generated metrics and produce a summary figure.

Reads   : results/<real_key>/metrics.json
          results/<gen_key>/metrics.json
          results/<real_key>/noise_stats.json   (optional)
          results/<gen_key>/noise_stats.json    (optional)
Writes  : results/<gen_key>/comparison/comparison_overview.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.config_loader import load_config
from videonoise.analysis.plots import comparison_figure, print_summary_table
from videonoise.utils import load_json


def main() -> None:
    cfg = load_config()

    real_metrics = Path(cfg.real_out) / "metrics.json"
    gen_metrics  = Path(cfg.gen_out)  / "metrics.json"
    real_noise   = Path(cfg.real_out) / "noise_stats.json"
    gen_noise    = Path(cfg.gen_out)  / "noise_stats.json"
    cmp_out      = Path(cfg.gen_out)  / "comparison"

    print("=" * 64)
    print("  Step 06 — Compare results")
    print(f"  Real      : {real_metrics}")
    print(f"  Generated : {gen_metrics}")
    print(f"  Output    : {cmp_out}/comparison_overview.png")
    print("=" * 64)

    _load = lambda p: load_json(str(p)) if p.exists() else {"per_video": {}}
    print_summary_table(_load(real_metrics), _load(gen_metrics), _load(real_noise), _load(gen_noise))

    cmp_out.mkdir(parents=True, exist_ok=True)
    comparison_figure(
        _load(real_metrics), _load(gen_metrics), _load(real_noise), _load(gen_noise),
        str(cmp_out / "comparison_overview.png"),
    )

    print(f"\n  Done. Figure saved to {cmp_out}/comparison_overview.png")


if __name__ == "__main__":
    main()
