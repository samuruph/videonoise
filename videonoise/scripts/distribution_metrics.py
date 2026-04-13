"""
Step 07 — Dataset-level distribution metrics: FVD, LPIPS cross-set, CLIP score.

Reads   : gen_dir  and  data_real  (from config)
Writes  : results/<gen_key>/distribution_metrics.json
          results/<gen_key>/plots/08_distribution_metrics.png

All settings come from scripts/config.yaml. Override any value from the CLI:
    python -m videonoise.scripts.distribution_metrics
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from videonoise.config_loader import load_config
from videonoise.utils import get_device


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _figure_08_distribution_metrics(metrics: dict, label: str, out: Path) -> None:
    """
    08 — Distribution Metrics
    ━━━━━━━━━━━━━━━━━━━━━━━
    Single summary figure showing FVD, LPIPS (3 variants), and CLIP score.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"08 — Distribution Metrics  |  {label}\n"
        "Dataset-level comparison between real and generated video distributions.",
        fontsize=10, fontweight="bold",
    )
    _GRID_KW = dict(color="lightgray", linestyle="--", linewidth=0.6, alpha=0.7)

    # Panel 0: FVD
    ax = axes[0]
    fvd = metrics.get("fvd", float("nan"))
    if np.isfinite(fvd):
        bar = ax.bar(["FVD"], [fvd], color="steelblue", alpha=0.8)
        ax.bar_label(bar, fmt="%.1f", fontsize=10)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_ylabel("FVD score")
    ax.set_title("Fréchet Video Distance\nLower = more similar distributions.\n"
                 "< 100 = good; > 500 = poor.", fontsize=8)
    ax.grid(axis="y", **_GRID_KW)

    # Panel 1: LPIPS (3 variants)
    ax = axes[1]
    lpips_keys = [
        ("lpips_temporal_real", "LPIPS\ntemporal\n(real)", "steelblue"),
        ("lpips_temporal_gen",  "LPIPS\ntemporal\n(gen)",  "tomato"),
        ("lpips_real_vs_gen_mean", "LPIPS\nreal↔gen",     "mediumseagreen"),
    ]
    x_pos = []
    x_labels = []
    colors = []
    heights = []
    for i, (key, lbl, color) in enumerate(lpips_keys):
        v = metrics.get(key, float("nan"))
        x_pos.append(i)
        x_labels.append(lbl)
        colors.append(color)
        heights.append(v if np.isfinite(v) else 0.0)
    bars = ax.bar(x_pos, heights, color=colors, alpha=0.8)
    for bar, h in zip(bars, heights):
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("LPIPS")
    ax.set_title("LPIPS Perceptual Distance\nLower = more perceptually similar.\n"
                 "Temporal: within-set; real↔gen: cross-set (matched pairs).", fontsize=8)
    ax.grid(axis="y", **_GRID_KW)

    # Panel 2: CLIP score
    ax = axes[2]
    clip_mean = metrics.get("clip_score_mean", float("nan"))
    clip_std  = metrics.get("clip_score_std",  float("nan"))
    if np.isfinite(clip_mean):
        bar = ax.bar(["CLIP score"], [clip_mean],
                     yerr=[clip_std] if np.isfinite(clip_std) else None,
                     color="mediumpurple", alpha=0.8, capsize=6)
        ax.bar_label(bar, fmt="%.3f", fontsize=10)
    else:
        skipped = metrics.get("clip_score_skipped", False) or metrics.get("skipped", False)
        msg = "No prompts\nfile found" if skipped else "N/A"
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0, 1)
    ax.set_title("CLIP Score\nMean frame-prompt cosine similarity.\n"
                 "Higher = better text alignment (text-to-video only).", fontsize=8)
    ax.grid(axis="y", **_GRID_KW)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config()
    device = get_device()

    out_json = Path(cfg.gen_out) / "distribution_metrics.json"
    out_plot = Path(cfg.gen_out) / "plots" / "08_distribution_metrics.png"

    print("=" * 64)
    print("  Step 07 — Distribution metrics")
    print(f"  Real      : {cfg.data_real}")
    print(f"  Generated : {cfg.gen_dir_resolved}")
    print(f"  Output    : {out_json}")
    print("=" * 64)

    from videonoise.metrics.distribution import (
        compute_fvd,
        compute_lpips_between_sets,
        compute_clip_score,
    )

    metrics: dict = {}

    # FVD
    print("\n  [1/3] FVD...")
    try:
        metrics["fvd"] = compute_fvd(
            cfg.data_real, cfg.gen_dir_resolved,
            max_frames=cfg.max_frames, resize=cfg.resize, device=device,
        )
        print(f"        FVD = {metrics['fvd']:.2f}")
    except Exception as e:
        print(f"        [skip] FVD failed: {e}")
        metrics["fvd"] = float("nan")

    # LPIPS
    print("\n  [2/3] Cross-set LPIPS...")
    try:
        lpips_res = compute_lpips_between_sets(
            cfg.data_real, cfg.gen_dir_resolved,
            max_frames=cfg.max_frames, resize=cfg.resize, device=device,
        )
        metrics.update(lpips_res)
        for k, v in lpips_res.items():
            print(f"        {k} = {v:.4f}")
    except Exception as e:
        print(f"        [skip] LPIPS failed: {e}")

    # CLIP score
    print("\n  [3/3] CLIP score...")
    try:
        clip_res = compute_clip_score(
            cfg.gen_dir_resolved,
            max_frames=cfg.max_frames, resize=cfg.resize, device=device,
        )
        if clip_res.get("skipped"):
            print("        [skip] No prompts file found")
            metrics["clip_score_mean"] = float("nan")
            metrics["clip_score_skipped"] = True
        else:
            metrics["clip_score_mean"] = clip_res["clip_score_mean"]
            metrics["clip_score_std"]  = clip_res["clip_score_std"]
            print(f"        CLIP score = {clip_res['clip_score_mean']:.4f} ± {clip_res['clip_score_std']:.4f}")
    except Exception as e:
        print(f"        [skip] CLIP score failed: {e}")
        metrics["clip_score_mean"] = float("nan")

    # Save JSON
    Path(cfg.gen_out).mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved → {out_json}")

    # Save figure
    _figure_08_distribution_metrics(metrics, cfg.gen_key, out_plot)

    # Print summary table
    print("\n" + "=" * 50)
    print(f"  {'Metric':<30} {'Value':>12}")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:>12.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
