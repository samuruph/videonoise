"""
Step 09 — Paired video analysis (matched real vs. generated).

Requires matched pairs: real and generated metrics.json must share video filenames.
Computes per-pair metric deltas and runs Wilcoxon signed-rank tests.

Reads   : results/<real_key>/metrics.json
          results/<gen_key>/metrics.json
Writes  : results/<gen_key>/paired_analysis/01_metric_deltas.png
          results/<gen_key>/paired_stats.json

All settings come from scripts/config.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from videonoise.config_loader import load_config
from videonoise.utils import load_json


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

_METRICS = [
    # (display_name, json_key_1, json_key_2, higher_is_better)
    ("TCS",           "temporal_ssim",    "temporal_ssim_mean",          True),
    ("ACF lag 1",     "temporal_acf",     "lag_1",                       True),
    ("Spectral slope","spatial_power_spectrum", "spectral_slope",         False),
    ("Flow std",      "optical_flow",     "flow_mag_std",                False),
    ("LPIPS",         "lpips_temporal",   "lpips_mean",                  False),
    ("Flow entropy",  "flow_direction_entropy", "flow_direction_entropy_mean", False),
]


def _extract(video_data: dict, k1: str, k2: str) -> float:
    return float(video_data.get(k1, {}).get(k2, float("nan")))


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def _figure_paired_deltas(
    matched: list[str],
    deltas: dict[str, list],
    pvals: dict[str, float],
    label: str,
    out: Path,
) -> None:
    """
    2×3 violin + strip chart per metric showing gen − real delta with Wilcoxon p-value.
    """
    n_metrics = len(_METRICS)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle(
        f"Paired Analysis: Generated − Real  |  {label}\n"
        f"Matched pairs: {len(matched)}. Wilcoxon signed-rank test (two-sided).",
        fontsize=10, fontweight="bold",
    )
    _GRID_KW = dict(color="lightgray", linestyle="--", linewidth=0.6, alpha=0.7)

    for i, (name, k1, k2, higher_better) in enumerate(_METRICS):
        ax = axes[i]
        d = deltas.get(name, [])
        finite_d = [v for v in d if np.isfinite(v)]
        if not finite_d:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name, fontsize=9)
            continue

        # Violin
        vp = ax.violinplot([finite_d], positions=[0], showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor("cornflowerblue")
            body.set_alpha(0.6)
        vp["cmedians"].set_color("navy")

        # Individual points with jitter
        jitter = np.random.uniform(-0.08, 0.08, len(finite_d))
        ax.scatter(jitter, finite_d, color="steelblue", s=15, zorder=5, alpha=0.7)

        # Zero reference line
        ax.axhline(0, color="k", linestyle="--", linewidth=1.2, zorder=10)

        # p-value annotation
        pval = pvals.get(name, float("nan"))
        pval_str = f"p = {pval:.3f}" if np.isfinite(pval) else "p = N/A"
        if np.isfinite(pval) and pval < 0.05:
            pval_str += " *"
        ax.set_title(f"{name}\n{pval_str}", fontsize=8)

        median = float(np.median(finite_d))
        direction = "↑" if higher_better else "↓"
        ax.set_xlabel(
            f"Δ = gen − real  |  med={median:+.3f}  ({direction} better for gen)",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.grid(axis="y", **_GRID_KW)

    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()

    real_json = Path(cfg.real_out) / "metrics.json"
    gen_json  = Path(cfg.gen_out)  / "metrics.json"

    print("=" * 64)
    print("  Step 09 — Paired analysis")
    print(f"  Real JSON : {real_json}")
    print(f"  Gen  JSON : {gen_json}")
    print("=" * 64)

    if not real_json.exists():
        print(f"\n  [error] {real_json} not found. Run step 02 first.")
        return
    if not gen_json.exists():
        print(f"\n  [error] {gen_json} not found. Run step 02 first.")
        return

    real_data = load_json(str(real_json)).get("per_video", {})
    gen_data  = load_json(str(gen_json)).get("per_video", {})

    # Match by stem
    def _stem(name: str) -> str:
        return Path(name).stem

    real_stems = {_stem(k): k for k in real_data}
    gen_stems  = {_stem(k): k for k in gen_data}
    common     = sorted(set(real_stems) & set(gen_stems))

    if not common:
        print(f"\n  [warn] No matching video names found between real ({len(real_data)}) "
              f"and gen ({len(gen_data)}) sets.")
        print("         Make sure real and generated video filenames share the same stem.")
        return

    print(f"\n  Matched pairs: {len(common)} / {len(real_data)} real / {len(gen_data)} gen")

    # Compute per-pair deltas
    deltas: dict[str, list] = {name: [] for name, *_ in _METRICS}
    for stem in common:
        rk = real_stems[stem]
        gk = gen_stems[stem]
        rd = real_data[rk]
        gd = gen_data[gk]
        for name, k1, k2, _ in _METRICS:
            rv = _extract(rd, k1, k2)
            gv = _extract(gd, k1, k2)
            deltas[name].append(gv - rv)

    # Wilcoxon signed-rank test
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    pvals: dict[str, float] = {}
    stats_out: dict = {}
    for name, _, _, higher_better in _METRICS:
        d = [v for v in deltas[name] if np.isfinite(v)]
        n = len(d)
        median_delta = float(np.median(d)) if d else float("nan")
        pval = float("nan")

        if wilcoxon is not None and n >= 5:
            try:
                stat, pval = wilcoxon(d, alternative="two-sided")
                pval = float(pval)
            except Exception:
                pass
        elif n < 5:
            print(f"  [warn] {name}: only {n} paired samples — Wilcoxon requires ≥ 5. Skipping test.")

        pvals[name] = pval
        stats_out[name] = {
            "n_pairs":      n,
            "median_delta": median_delta,
            "mean_delta":   float(np.mean(d)) if d else float("nan"),
            "std_delta":    float(np.std(d))  if d else float("nan"),
            "p_value":      pval,
            "significant":  bool(np.isfinite(pval) and pval < 0.05),
            "higher_is_better": higher_better,
        }

    # Save JSON
    out_json = Path(cfg.gen_out) / "paired_stats.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"matched_pairs": common, "metrics": stats_out}, f, indent=2)
    print(f"\n  Saved → {out_json}")

    # Save figure
    out_fig = Path(cfg.gen_out) / "paired_analysis" / "01_metric_deltas.png"
    _figure_paired_deltas(common, deltas, pvals, cfg.gen_key, out_fig)

    # Print summary table
    print("\n" + "=" * 72)
    print(f"  {'Metric':<20} {'N':>4} {'Median Δ':>10} {'Mean Δ':>10} {'p-value':>10} {'Sig?':>6}")
    print("=" * 72)
    for name, _, _, _ in _METRICS:
        s = stats_out.get(name, {})
        sig = "*" if s.get("significant") else ""
        pv = s.get("p_value", float("nan"))
        pv_str = f"{pv:.4f}" if np.isfinite(pv) else "  N/A"
        print(f"  {name:<20} {s.get('n_pairs', 0):>4} "
              f"{s.get('median_delta', float('nan')):>+10.4f} "
              f"{s.get('mean_delta',   float('nan')):>+10.4f} "
              f"{pv_str:>10} {sig:>6}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
