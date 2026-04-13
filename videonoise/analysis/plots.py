"""
Comparison plots and summary tables for real vs. generated video metrics.

Single-dataset visualisation (vn-viz)
--------------------------------------
Saves 5 numbered PNGs in results/plots/<label>/:

  01_temporal_coherence.png  — TCS, Pearson, Spearman, PSNR
  02_motion_dynamics.png     — optical-flow stats + temporal ACF
  03_spectral.png            — power spectra, slope, R², 3D energy ratio
  04_frame_content.png       — frame-level pixel statistics
  05_noise_stats.png         — inverted-noise stats (step 03, optional)
"""
import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import norm

from videonoise.utils import extract_scalar, load_json, save_json

# ─────────────────────────────────────────────────────────────────────────────
# Internal style helpers
# ─────────────────────────────────────────────────────────────────────────────
_COLOR = "steelblue"
_COLOR2 = "tomato"
_GRID_KW = dict(color="lightgray", linestyle="--", linewidth=0.6, alpha=0.7)


def _bar_per_video(
    ax,
    names: list,
    values: list,
    ylabel: str,
    title: str,
    color: str = _COLOR,
    ref_line: float = None,
    ref_label: str = None,
    ylim: tuple = None,
) -> None:
    """
    Horizontal bar chart: one bar per video.
    Shows individual values with a mean ± std title annotation.
    """
    vals = np.asarray(values, dtype=float)
    finite = vals[np.isfinite(vals)]
    x = np.arange(len(names))

    ax.barh(x, vals, color=color, alpha=0.75)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", **_GRID_KW)
    ax.invert_yaxis()

    if ref_line is not None:
        ax.axvline(ref_line, color="k", linestyle="--", linewidth=1.2,
                   label=ref_label or f"ref={ref_line:.2f}")
        ax.legend(fontsize=7)
    if ylim:
        ax.set_xlim(ylim)

    if finite.size > 0:
        ax.set_title(
            f"{title}\n"
            + r"$\bar{x}$" + f"={np.mean(finite):.3f}  "
            + r"$\sigma$" + f"={np.std(finite):.3f}",
            fontsize=8,
        )


def _line_acf(ax, per_video: dict, names: list, color: str = _COLOR) -> None:
    """Overlay temporal ACF curves (lags 1–10) for each video."""
    lags = list(range(1, 11))
    cmap = plt.get_cmap("tab10")
    for i, n in enumerate(names[:10]):
        acf = per_video[n].get("temporal_acf", {})
        vals = [acf.get(f"lag_{k}", float("nan")) for k in lags]
        ax.plot(lags, vals, "-o", alpha=0.7, markersize=3,
                color=cmap(i), label=n)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("ACF")
    ax.set_title(
        "Temporal ACF\n"
        "How quickly temporal correlation decays. "
        "Real videos → slow decay; generated → fast.",
        fontsize=8,
    )
    ax.grid(**_GRID_KW)
    if len(names) <= 8:
        ax.legend(fontsize=6, ncol=2)


def _line_spectra(ax, per_video: dict, names: list, color: str = _COLOR) -> None:
    """Overlay log-log radial power spectra."""
    cmap = plt.get_cmap("tab10")
    for i, n in enumerate(names[:10]):
        profile = per_video[n].get("spatial_power_spectrum", {}).get("radial_profile", [])
        if profile:
            ax.semilogy(profile, alpha=0.7, color=cmap(i), label=n)
    ax.set_xlabel("Spatial frequency bin")
    ax.set_ylabel("Power (log)")
    ax.set_title(
        "Radial Power Spectrum\n"
        "Natural images → 1/f² slope (−2.0 on log-log). "
        "Generated often steeper or flatter.",
        fontsize=8,
    )
    ax.grid(**_GRID_KW)
    if len(names) <= 8:
        ax.legend(fontsize=6, ncol=2)


# ─────────────────────────────────────────────────────────────────────────────
# Individual comparison-plot helpers  (used by comparison_figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_bars(real_vals, gen_vals, metric_name, ax) -> None:
    """Bar chart with individual-point scatter overlay for real vs. generated."""
    means  = [np.mean(real_vals) if real_vals else 0, np.mean(gen_vals) if gen_vals else 0]
    stds   = [np.std(real_vals)  if real_vals else 0, np.std(gen_vals)  if gen_vals else 0]
    colors = [_COLOR, _COLOR2]
    ax.bar(["Real", "Generated"], means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    for i, (vals, c) in enumerate(zip([real_vals, gen_vals], colors)):
        jitter = np.random.uniform(-0.1, 0.1, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color="k", s=15, zorder=5, alpha=0.6)
    ax.set_title(metric_name)
    ax.set_ylabel("value")


def plot_acf_comparison(real_results, gen_results, ax) -> None:
    lags = list(range(1, 11))
    for vid_data in list(real_results.get("per_video", {}).values())[:5]:
        acf  = vid_data.get("temporal_acf", {})
        vals = [acf.get(f"lag_{k}", float("nan")) for k in lags]
        ax.plot(lags, vals, "b-o", alpha=0.4, markersize=3)
    for vid_data in list(gen_results.get("per_video", {}).values())[:5]:
        acf  = vid_data.get("temporal_acf", {})
        vals = [acf.get(f"lag_{k}", float("nan")) for k in lags]
        ax.plot(lags, vals, "r-s", alpha=0.4, markersize=3)
    ax.legend(handles=[Line2D([0], [0], color="b", label="Real"),
                        Line2D([0], [0], color="r", label="Generated")])
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("ACF")
    ax.set_title("Temporal ACF")


def plot_power_spectra(real_results, gen_results, ax) -> None:
    for vid_data in list(real_results.get("per_video", {}).values())[:5]:
        profile = vid_data.get("spatial_power_spectrum", {}).get("radial_profile", [])
        if profile:
            ax.semilogy(profile, "b-", alpha=0.4)
    for vid_data in list(gen_results.get("per_video", {}).values())[:5]:
        profile = vid_data.get("spatial_power_spectrum", {}).get("radial_profile", [])
        if profile:
            ax.semilogy(profile, "r-", alpha=0.4)
    ax.legend(handles=[Line2D([0], [0], color="b", label="Real"),
                        Line2D([0], [0], color="r", label="Generated")])
    ax.set_xlabel("Spatial frequency (radial)")
    ax.set_ylabel("Power")
    ax.set_title("2D Power Spectrum")


def plot_noise_histograms(noise_real, noise_gen, ax) -> None:
    for vid_data in list(noise_real.get("per_video", {}).values())[:3]:
        s  = vid_data.get("statistics", {})
        mu, sigma = s.get("mean", 0), s.get("std", 1)
        x  = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(x, norm.pdf(x, mu, sigma), "b-", alpha=0.5)
    for vid_data in list(noise_gen.get("per_video", {}).values())[:3]:
        s  = vid_data.get("statistics", {})
        mu, sigma = s.get("mean", 0), s.get("std", 1)
        x  = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(x, norm.pdf(x, mu, sigma), "r-", alpha=0.5)
    x = np.linspace(-4, 4, 200)
    ax.plot(x, norm.pdf(x), "k--", label="N(0,1)", linewidth=2)
    ax.legend(handles=[
        Line2D([0], [0], color="b", label="Real (inverted)"),
        Line2D([0], [0], color="r", label="Generated (inverted)"),
        Line2D([0], [0], color="k", linestyle="--", label="N(0,1)"),
    ])
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.set_title("Inverted Noise Distribution")


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(real_results, gen_results, noise_real, noise_gen) -> None:
    rows = [
        ("Frame Pearson corr",        real_results, gen_results,  "frame_correlation",     "pearson_mean"),
        ("Temporal SSIM (TCS)",       real_results, gen_results,  "temporal_ssim",         "temporal_ssim_mean"),
        ("PSNR consecutive frames",   real_results, gen_results,  "psnr",                  "psnr_mean"),
        ("Optical flow mag (mean)",   real_results, gen_results,  "optical_flow",          "flow_mag_mean"),
        ("Spectral slope",            real_results, gen_results,  "spatial_power_spectrum","spectral_slope"),
        ("Low-freq energy ratio",     real_results, gen_results,  "spatiotemporal_3d",     "low_freq_energy_ratio"),
        ("Noise KL from N(0,1)",      noise_real,   noise_gen,    "statistics",            "kl_from_gaussian"),
        ("Noise cross-frame corr",    noise_real,   noise_gen,    "cross_frame_correlation","cross_frame_corr_mean"),
    ]
    print("\n" + "=" * 70)
    print(f"  {'Metric':<38} {'Real':>12} {'Generated':>12}")
    print("=" * 70)
    for name, rr, gr, k1, k2 in rows:
        rv = extract_scalar(rr, k1, k2)
        gv = extract_scalar(gr, k1, k2)
        if rv and gv:
            print(f"  {name:<38} {np.mean(rv):>12.4f} {np.mean(gv):>12.4f}")
        else:
            print(f"  {name:<38} {'N/A':>12} {'N/A':>12}")
    print("=" * 70 + "\n")


def print_single_table(results: dict, label: str) -> None:
    """Print a per-video metrics table for a single dataset."""
    per_video = results.get("per_video", {})
    print(f"\n{'=' * 92}")
    print(f"  {label}")
    print(f"  {'Video':<30} {'TCS':>7} {'Pearson':>8} {'Spearman':>9} {'PSNR':>7} "
          f"{'Flow':>7} {'Slope':>7} {'R²':>6}")
    print(f"{'=' * 92}")
    for name, data in per_video.items():
        tcs   = data.get("temporal_ssim",         {}).get("temporal_ssim_mean",  float("nan"))
        pear  = data.get("frame_correlation",      {}).get("pearson_mean",        float("nan"))
        spear = data.get("frame_correlation",      {}).get("spearman_mean",       float("nan"))
        psnr  = data.get("psnr",                   {}).get("psnr_mean",           float("nan"))
        flow  = data.get("optical_flow",           {}).get("flow_mag_mean",       float("nan"))
        slope = data.get("spatial_power_spectrum", {}).get("spectral_slope",      float("nan"))
        r2    = data.get("spatial_power_spectrum", {}).get("spectral_r2",         float("nan"))
        print(f"  {name:<30} {tcs:>7.4f} {pear:>8.4f} {spear:>9.4f} {psnr:>7.2f} "
              f"{flow:>7.4f} {slope:>7.3f} {r2:>6.3f}")

    def _mean(k1, k2):
        vals = extract_scalar(results, k1, k2) or []
        return np.nanmean(vals) if vals else float("nan")

    print(f"{'─' * 92}")
    print(f"  {'MEAN':<30} "
          f"{_mean('temporal_ssim', 'temporal_ssim_mean'):>7.4f} "
          f"{_mean('frame_correlation', 'pearson_mean'):>8.4f} "
          f"{_mean('frame_correlation', 'spearman_mean'):>9.4f} "
          f"{_mean('psnr', 'psnr_mean'):>7.2f} "
          f"{_mean('optical_flow', 'flow_mag_mean'):>7.4f} "
          f"{_mean('spatial_power_spectrum', 'spectral_slope'):>7.3f} "
          f"{_mean('spatial_power_spectrum', 'spectral_r2'):>6.3f}")
    print(f"{'=' * 92}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Full comparison figure  (vn-compare, requires both JSONs)
# ─────────────────────────────────────────────────────────────────────────────

def comparison_figure(real_results, gen_results, noise_real, noise_gen,
                      output_path: str) -> None:
    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    scalar_metrics = [
        ("Frame Pearson Corr",    "frame_correlation",      "pearson_mean"),
        ("Temporal SSIM",         "temporal_ssim",          "temporal_ssim_mean"),
        ("PSNR (consec. frames)", "psnr",                   "psnr_mean"),
        ("Optical flow mean",     "optical_flow",           "flow_mag_mean"),
    ]
    for col, (label, k1, k2) in enumerate(scalar_metrics):
        ax = fig.add_subplot(gs[0, col])
        plot_metric_bars(extract_scalar(real_results, k1, k2),
                         extract_scalar(gen_results,  k1, k2), label, ax)

    plot_acf_comparison(real_results, gen_results, fig.add_subplot(gs[1, 0:2]))
    plot_power_spectra(real_results,  gen_results, fig.add_subplot(gs[1, 2:4]))

    plot_noise_histograms(noise_real, noise_gen, fig.add_subplot(gs[2, 0:2]))
    plot_metric_bars(extract_scalar(noise_real, "statistics", "kl_from_gaussian"),
                     extract_scalar(noise_gen,  "statistics", "kl_from_gaussian"),
                     "Noise KL from N(0,1)", fig.add_subplot(gs[2, 2]))
    plot_metric_bars(extract_scalar(noise_real, "cross_frame_correlation", "cross_frame_corr_mean"),
                     extract_scalar(noise_gen,  "cross_frame_correlation", "cross_frame_corr_mean"),
                     "Noise cross-frame corr", fig.add_subplot(gs[2, 3]))

    plt.suptitle("Real vs. Generated Video — Metric Comparison", fontsize=16, fontweight="bold")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive single-dataset plots  (vn-viz, step 02 / step 03)
# ─────────────────────────────────────────────────────────────────────────────

def _figure_01_temporal_coherence(per_video: dict, names: list,
                                   label: str, out: Path) -> None:
    """
    01 — Temporal Coherence
    ━━━━━━━━━━━━━━━━━━━━━━
    4 panels showing how coherent consecutive frames are:

    • TCS (Temporal SSIM)   — structural similarity t→t+1; higher = smoother
    • Pearson r             — linear pixel correlation t→t+1; higher = more coherent
    • Spearman ρ            — rank correlation t→t+1; robust to nonlinearities
    • PSNR                  — peak signal-to-noise ratio t→t+1; higher = less noise
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"01 — Temporal Coherence  |  {label}\n"
        "Measures how similar consecutive frames are. "
        "Higher values → smoother, more temporally consistent video.",
        fontsize=10, fontweight="bold",
    )

    metrics = [
        (axes[0, 0], "temporal_ssim",    "temporal_ssim_mean", "TCS",
         "Temporal SSIM (TCS)\nSSIM between t and t+1. [0–1], higher = more coherent.", None),
        (axes[0, 1], "frame_correlation", "pearson_mean",       "Pearson r",
         "Frame-to-frame Pearson r\nLinear pixel correlation. [−1…1], ~0.95+ typical for real.", None),
        (axes[1, 0], "frame_correlation", "spearman_mean",      "Spearman ρ",
         "Frame-to-frame Spearman ρ\nRank-based correlation. Robust to outliers.", None),
        (axes[1, 1], "psnr",             "psnr_mean",          "PSNR (dB)",
         "Inter-frame PSNR\nHigher = less pixel-level noise between frames.\n"
         "30+ dB → good; <20 dB → strong flicker.", None),
    ]
    for ax, k1, k2, ylabel, title, ref in metrics:
        vals = [per_video[n].get(k1, {}).get(k2, float("nan")) for n in names]
        _bar_per_video(ax, names, vals, ylabel, title, ref_line=ref)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def _figure_02_motion_dynamics(per_video: dict, names: list,
                                label: str, out: Path) -> None:
    """
    02 — Motion & Dynamics
    ━━━━━━━━━━━━━━━━━━━━━
    4 panels describing motion characteristics:

    • Flow mean             — average motion magnitude (pixels/frame)
    • Flow std              — variability of motion (low = smooth pan; high = chaotic)
    • Flow p95              — 95th-percentile motion magnitude (peaks / fast edges)
    • Temporal ACF          — autocorrelation at lags 1–10 (persistence of signal)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"02 — Motion & Dynamics  |  {label}\n"
        "Describes how much and how smoothly objects move between frames.",
        fontsize=10, fontweight="bold",
    )

    # flow mean
    vals = [per_video[n].get("optical_flow", {}).get("flow_mag_mean", float("nan")) for n in names]
    _bar_per_video(axes[0, 0], names, vals, "px / frame",
                   "Optical Flow Mean\nAverage motion magnitude. "
                   "Real: 1–10 px typical; near 0 → static / frozen.")

    # flow std
    vals = [per_video[n].get("optical_flow", {}).get("flow_mag_std", float("nan")) for n in names]
    _bar_per_video(axes[0, 1], names, vals, "px / frame",
                   "Optical Flow Std Dev\nVariability in motion. "
                   "High → erratic / flickering; Low → uniform movement.")

    # flow p95
    vals = [per_video[n].get("optical_flow", {}).get("flow_mag_p95", float("nan")) for n in names]
    _bar_per_video(axes[1, 0], names, vals, "px / frame",
                   "Optical Flow p95\n95th-percentile magnitude. "
                   "Highlights fast-moving edges or motion peaks.")

    # temporal ACF
    _line_acf(axes[1, 1], per_video, names)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def _figure_03_spectral(per_video: dict, names: list,
                         label: str, out: Path) -> None:
    """
    03 — Spectral Properties
    ━━━━━━━━━━━━━━━━━━━━━━━
    4 panels describing spatial and spatio-temporal frequency content:

    • Radial power spectra  — log-log plot of 2D FFT power vs spatial frequency
    • Spectral slope        — best-fit exponent of the 1/f^β power law (ref −2.0)
    • Spectral R²           — goodness of fit; how well the 1/f² model holds
    • 3D low-freq ratio     — fraction of 3D energy in low spatio-temporal frequencies
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"03 — Spectral Properties  |  {label}\n"
        "How spatial frequency content is distributed. "
        "Natural images follow a 1/f² (slope ≈ −2) power law.",
        fontsize=10, fontweight="bold",
    )

    # radial power spectrum curves
    _line_spectra(axes[0, 0], per_video, names)

    # spectral slope
    vals = [per_video[n].get("spatial_power_spectrum", {}).get("spectral_slope", float("nan"))
            for n in names]
    _bar_per_video(axes[0, 1], names, vals, "slope (log-log)",
                   "Spectral Slope β\n1/f^β power law exponent. "
                   "Natural images: β ≈ −2.0 (dashed line).\n"
                   "Flatter (> −2): too much detail / noise; "
                   "Steeper (< −2): over-smoothed.",
                   ref_line=-2.0, ref_label="natural image (−2.0)")

    # spectral R²
    vals = [per_video[n].get("spatial_power_spectrum", {}).get("spectral_r2", float("nan"))
            for n in names]
    _bar_per_video(axes[1, 0], names, vals, "R²",
                   "Power-Law Fit R²\nHow well the 1/f² model fits. "
                   "[0–1], higher = cleaner power-law structure.\n"
                   "< 0.85 may indicate mixed/irregular textures.",
                   ylim=(0, 1.05))

    # 3D low-freq energy ratio
    vals = [per_video[n].get("spatiotemporal_3d", {}).get("low_freq_energy_ratio", float("nan"))
            for n in names]
    _bar_per_video(axes[1, 1], names, vals, "ratio [0–1]",
                   "3D Low-Frequency Energy Ratio\nFraction of spatio-temporal energy\n"
                   "in low frequencies. Higher → smoother motion and content;\n"
                   "Lower → high-frequency / flickering video.",
                   ylim=(0, 1.05))

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def _figure_04_frame_content(per_video: dict, names: list,
                              label: str, out: Path) -> None:
    """
    04 — Frame Content Statistics
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    4 panels describing pixel-level distribution of frames:

    • Mean brightness       — average pixel intensity (0–1 range)
    • Std deviation         — contrast / dynamic range
    • Skewness              — asymmetry of pixel histogram; 0 = symmetric
    • Kurtosis              — tail heaviness; 3.0 = Gaussian, higher = more peaked
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"04 — Frame Content Statistics  |  {label}\n"
        "Pixel-level distribution of frame intensities. "
        "Helps detect over/under-exposure and unusual brightness patterns.",
        fontsize=10, fontweight="bold",
    )

    metrics = [
        (axes[0, 0], "mean",     "Mean brightness\nAverage pixel value [0–1]. "
                                  "Too high (> 0.8) → over-exposed; "
                                  "too low (< 0.2) → under-exposed.", None),
        (axes[0, 1], "std",      "Std Dev (contrast)\nHigher → richer contrast / more detail.\n"
                                  "Very low (< 0.05) → flat / grey-ish frames.", None),
        (axes[1, 0], "skewness", "Skewness\nAsymmetry of brightness histogram.\n"
                                  "0 = symmetric; + = more dark; − = more bright.", 0.0),
        (axes[1, 1], "kurtosis", "Kurtosis\nTail heaviness of brightness distribution.\n"
                                  "~3 = Gaussian; > 3 = sharp peak (many near-equal pixels).", 3.0),
    ]
    for ax, key, title, ref in metrics:
        vals = [per_video[n].get("frame_statistics", {}).get(key, float("nan")) for n in names]
        ref_label = {"skewness": "symmetric (0)", "kurtosis": "Gaussian (3.0)"}.get(key)
        _bar_per_video(ax, names, vals, key, title, ref_line=ref, ref_label=ref_label)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def _figure_05_noise_stats(noise_results: dict, names: list,
                            label: str, out: Path) -> None:
    """
    05 — Inverted Noise Statistics  (step 03)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    4 panels describing the statistical properties of recovered noise latents:

    • KL from N(0,1)        — divergence from standard Gaussian; 0 = perfectly Gaussian
    • KS p-value            — Kolmogorov-Smirnov normality test; > 0.05 = "looks Gaussian"
    • Cross-frame corr      — temporal correlation in noise across frames; 0 = i.i.d.
    • Noise spectral slope  — power-law exponent of 1D noise spectrum
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"05 — Inverted Noise Statistics  |  {label}\n"
        "Properties of the noise latent recovered from the video via DDPM inversion.\n"
        "Ideal i.i.d. Gaussian noise: KL=0, KS p>0.05, cross-frame corr≈0.",
        fontsize=10, fontweight="bold",
    )
    noise_pv = noise_results.get("per_video", {})

    # KL from Gaussian
    vals = [noise_pv.get(n, {}).get("statistics", {}).get("kl_from_gaussian", float("nan"))
            for n in names]
    _bar_per_video(axes[0, 0], names, vals, "KL divergence",
                   "KL from N(0,1)\n0 = perfectly Gaussian. "
                   "Higher → inverted noise deviates from i.i.d.",
                   color="mediumseagreen", ref_line=0.0, ref_label="ideal (0)")

    # KS p-value
    vals = [noise_pv.get(n, {}).get("statistics", {}).get("ks_p_value", float("nan"))
            for n in names]
    _bar_per_video(axes[0, 1], names, vals, "p-value",
                   "KS Normality p-value\np > 0.05 → cannot reject Gaussian.\n"
                   "Low p → noise is non-Gaussian (structured).",
                   color="mediumseagreen", ref_line=0.05, ref_label="α=0.05",
                   ylim=(0, 1.05))

    # Cross-frame correlation
    vals = [noise_pv.get(n, {}).get("cross_frame_correlation", {}).get("cross_frame_corr_mean",
                                                                        float("nan"))
            for n in names]
    _bar_per_video(axes[1, 0], names, vals, "correlation",
                   "Cross-frame Noise Correlation\nCorrelation between ε_t and ε_{t+1}.\n"
                   "0 = temporally independent; > 0 → correlated across frames.",
                   color="mediumseagreen", ref_line=0.0, ref_label="i.i.d. (0)")

    # Noise spectral slope
    vals = [noise_pv.get(n, {}).get("power_spectrum", {}).get("spectral_slope_1d", float("nan"))
            for n in names]
    _bar_per_video(axes[1, 1], names, vals, "slope",
                   "Noise 1D Spectral Slope\nPower-law slope of noise spectrum.\n"
                   "White noise → 0; red noise → negative; blue → positive.",
                   color="mediumseagreen", ref_line=0.0, ref_label="white noise (0)")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def _figure_06_motion_structure(per_video: dict, names: list,
                                 label: str, out: Path) -> None:
    """
    06 — Motion Structure & Perceptual Consistency
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    4 panels:

    • Flow direction entropy    — how organised motion direction is (low=coherent)
    • Multiscale SSIM           — SSIM coherence by spatial scale
    • Temporal MI               — non-linear frame dependence at lags 1–5
    • LPIPS temporal            — perceptual consistency between consecutive frames
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"06 — Motion Structure & Perceptual Consistency  |  {label}\n"
        "Flow entropy (lower=coherent), multiscale SSIM, temporal MI, and perceptual consistency.",
        fontsize=10, fontweight="bold",
    )

    # flow direction entropy
    vals = [per_video[n].get("flow_direction_entropy", {}).get(
                "flow_direction_entropy_mean", float("nan")) for n in names]
    _bar_per_video(
        axes[0, 0], names, vals, "entropy (bits)",
        "Flow Direction Entropy\nShannon entropy of optical-flow angle distribution.\n"
        "Low → coherent directed motion; ~5.17 bits → fully random.",
        ref_line=float(np.log2(36)), ref_label="max (random, 5.17 bits)",
    )

    # multiscale SSIM — line per video, x=scale index
    ax_ms = axes[0, 1]
    cmap = plt.get_cmap("tab10")
    has_data = False
    for i, n in enumerate(names[:10]):
        ms = per_video[n].get("multiscale_ssim", {})
        scales = sorted(k for k in ms if k.startswith("scale_"))
        if not scales:
            continue
        xs = [int(k.split("_")[1]) for k in scales]
        ys = [ms[k].get("ssim_mean", float("nan")) for k in scales]
        ax_ms.plot(xs, ys, "-o", markersize=4, alpha=0.7, color=cmap(i), label=n)
        has_data = True
    ax_ms.set_xlabel("Scale index (0=full, 1=½, 2=¼, …)")
    ax_ms.set_ylabel("SSIM mean")
    ax_ms.set_title(
        "Multiscale Temporal SSIM\nSSIM between consecutive frames at each spatial scale.\n"
        "Rapid drop → fine-scale coherence lost; flat → robust temporal structure.",
        fontsize=8,
    )
    ax_ms.grid(color="lightgray", linestyle="--", linewidth=0.6, alpha=0.7)
    if has_data and len(names) <= 8:
        ax_ms.legend(fontsize=6, ncol=2)

    # temporal MI — line per video, x=lag
    ax_mi = axes[1, 0]
    lags = list(range(1, 6))
    has_data = False
    for i, n in enumerate(names[:10]):
        mi = per_video[n].get("temporal_mi", {})
        vals_mi = [mi.get(f"lag_{k}", float("nan")) for k in lags]
        if any(np.isfinite(v) for v in vals_mi):
            ax_mi.plot(lags, vals_mi, "-o", markersize=4, alpha=0.7, color=cmap(i), label=n)
            has_data = True
    ax_mi.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_mi.set_xlabel("Lag (frames)")
    ax_mi.set_ylabel("MI (bits)")
    ax_mi.set_title(
        "Temporal Mutual Information\nNon-linear frame dependence at lags 1–5.\n"
        "Higher MI at lag 1 → strong temporal structure; fast decay → weak coherence.",
        fontsize=8,
    )
    ax_mi.grid(color="lightgray", linestyle="--", linewidth=0.6, alpha=0.7)
    if has_data and len(names) <= 8:
        ax_mi.legend(fontsize=6, ncol=2)

    # LPIPS temporal
    vals = [per_video[n].get("lpips_temporal", {}).get("lpips_mean", float("nan")) for n in names]
    _bar_per_video(
        axes[1, 1], names, vals, "LPIPS",
        "LPIPS Temporal Consistency\nPerceptual distance between consecutive frames.\n"
        "Lower → more perceptually smooth; 0 = identical frames.",
        ref_line=0.0, ref_label="identical (0)",
    )

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def _figure_07_noise_covariance(noise_results: dict, names: list,
                                  label: str, out: Path) -> None:
    """
    07 — Noise Covariance Structure  (step 03, optional)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    4 panels analysing temporal covariance of inverted noise:

    • Off-diagonal energy ratio — fraction of covariance energy off the diagonal
    • Top eigenvalue ratio      — dominance of the largest eigenvalue
    • Cumulative explained var  — how fast eigenvalues accumulate energy
    • Normalised eigenspectra   — log-scale eigenvalue distributions per video
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"07 — Noise Covariance Structure  |  {label}\n"
        "Temporal covariance matrix of the inverted noise latent. "
        "i.i.d. Gaussian: off-diag≈0, uniform eigenspectrum.",
        fontsize=10, fontweight="bold",
    )
    noise_pv = noise_results.get("per_video", {})

    # off-diagonal energy ratio
    vals = [noise_pv.get(n, {}).get("covariance_structure", {}).get(
                "off_diagonal_energy_ratio", float("nan")) for n in names]
    _bar_per_video(
        axes[0, 0], names, vals, "ratio [0–1]",
        "Off-Diagonal Energy Ratio\nFraction of Frobenius norm from off-diagonal C entries.\n"
        "0 = perfectly i.i.d. frames; higher → temporal structure in noise.",
        color="mediumseagreen", ref_line=0.0, ref_label="i.i.d. (0)",
    )

    # top eigenvalue ratio
    vals = [noise_pv.get(n, {}).get("covariance_structure", {}).get(
                "top_eigenvalue_ratio", float("nan")) for n in names]
    # reference: 1/T (uniform spectrum)
    T_ref = None
    for n in names:
        shape = noise_pv.get(n, {}).get("shape")
        if shape:
            T_ref = shape[0]
            break
    ref_line = 1.0 / T_ref if T_ref else None
    _bar_per_video(
        axes[0, 1], names, vals, "ratio [0–1]",
        "Top Eigenvalue Ratio\nFraction of total variance in the dominant eigenvector.\n"
        "1/T = uniform spectrum (i.i.d.); higher → low-rank temporal structure.",
        color="mediumseagreen",
        ref_line=ref_line,
        ref_label=f"uniform (1/T={ref_line:.3f})" if ref_line else None,
    )

    # cumulative explained variance — lines per video
    ax_cum = axes[1, 0]
    cmap = plt.get_cmap("tab10")
    has_data = False
    for i, n in enumerate(names[:10]):
        cev = noise_pv.get(n, {}).get("covariance_structure", {}).get(
            "eigenvalue_explained_variance", [])
        if cev:
            ax_cum.plot(range(1, len(cev) + 1), cev, "-", alpha=0.7, color=cmap(i), label=n)
            has_data = True
    ax_cum.set_xlabel("Eigenvalue index")
    ax_cum.set_ylabel("Cumulative explained variance")
    ax_cum.set_title(
        "Cumulative Eigenvalue Explained Variance\n"
        "How fast the eigenspectrum concentrates energy.\n"
        "i.i.d. noise → linear diagonal line; structured → fast initial rise.",
        fontsize=8,
    )
    ax_cum.grid(color="lightgray", linestyle="--", linewidth=0.6, alpha=0.7)
    if has_data and len(names) <= 8:
        ax_cum.legend(fontsize=6, ncol=2)

    # normalised eigenspectra (log scale)
    ax_eig = axes[1, 1]
    has_data = False
    for i, n in enumerate(names[:10]):
        cev = noise_pv.get(n, {}).get("covariance_structure", {}).get(
            "eigenvalue_explained_variance", [])
        if len(cev) > 1:
            marginal = np.diff([0.0] + list(cev))
            ax_eig.semilogy(range(1, len(marginal) + 1), marginal + 1e-9,
                            "-", alpha=0.7, color=cmap(i), label=n)
            has_data = True
    ax_eig.set_xlabel("Eigenvalue index")
    ax_eig.set_ylabel("Marginal variance (log)")
    ax_eig.set_title(
        "Normalised Eigenvalue Spectrum (log)\n"
        "Flat → white noise; rapid decay → dominant temporal modes.",
        fontsize=8,
    )
    ax_eig.grid(color="lightgray", linestyle="--", linewidth=0.6, alpha=0.7)
    if has_data and len(names) <= 8:
        ax_eig.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_all_single(results: dict, label: str, output_dir: str,
                    noise_results: dict = None) -> None:
    """
    Render all per-dataset plots and save them directly into output_dir.

    The caller is responsible for supplying the correct directory
    (e.g. results/real/plots/ or results/modelscope_gaussian/plots/).
    The label is used only for figure titles — it does NOT create a sub-folder.

    Files produced:
      01_temporal_coherence.png
      02_motion_dynamics.png
      03_spectral.png
      04_frame_content.png
      05_noise_stats.png          ← only when noise_results is provided
      06_motion_structure.png     ← when new metric keys are present
      07_noise_covariance.png     ← only when noise_results has covariance_structure
    """
    per_video = results.get("per_video", {})
    names     = list(per_video.keys())

    if not names:
        print(f"  [warn] No per-video data found in results — skipping {label}")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Dataset : {label}  ({len(names)} videos)")
    print(f"  Output  : {out_dir}/")

    _figure_01_temporal_coherence(per_video, names, label, out_dir / "01_temporal_coherence.png")
    _figure_02_motion_dynamics(per_video, names, label, out_dir / "02_motion_dynamics.png")
    _figure_03_spectral(per_video, names, label, out_dir / "03_spectral.png")
    _figure_04_frame_content(per_video, names, label, out_dir / "04_frame_content.png")

    if noise_results is not None:
        _figure_05_noise_stats(noise_results, names, label, out_dir / "05_noise_stats.png")
    else:
        print("  (skipping 05_noise_stats — no noise JSON provided; "
              "run step 03 and pass --noise_stats)")

    # Figure 06: new motion-structure metrics (only if keys are present)
    first = next(iter(per_video.values()), {})
    has_new_metrics = any(
        k in first for k in ("flow_direction_entropy", "temporal_mi", "lpips_temporal", "multiscale_ssim")
    )
    if has_new_metrics:
        _figure_06_motion_structure(per_video, names, label, out_dir / "06_motion_structure.png")
    else:
        print("  (skipping 06_motion_structure — rerun step 02 to compute new metrics)")

    # Figure 07: noise covariance (only if noise_results has covariance_structure)
    if noise_results is not None:
        noise_pv = noise_results.get("per_video", {})
        has_cov = any(
            "covariance_structure" in noise_pv.get(n, {})
            for n in names
        )
        if has_cov:
            _figure_07_noise_covariance(noise_results, names, label, out_dir / "07_noise_covariance.png")
        else:
            print("  (skipping 07_noise_covariance — rerun step 03 to compute covariance structure)")

    print(f"\n  All plots saved to {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Console-script entry points
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    """vn-compare: compare real vs. generated (requires both JSONs)."""
    parser = argparse.ArgumentParser(description="Compare real vs. generated video metrics")
    parser.add_argument("--real",       default="results/metrics/real.json")
    parser.add_argument("--generated",  default="results/metrics/generated.json")
    parser.add_argument("--noise_real", default="results/noise_stats/real.json")
    parser.add_argument("--noise_gen",  default="results/noise_stats/generated.json")
    parser.add_argument("--output",     default="results/plots/comparison/")
    args = parser.parse_args()

    _load = lambda p: load_json(p) if Path(p).exists() else {"per_video": {}}
    real_results = _load(args.real)
    gen_results  = _load(args.generated)
    noise_real   = _load(args.noise_real)
    noise_gen    = _load(args.noise_gen)

    print_summary_table(real_results, gen_results, noise_real, noise_gen)
    comparison_figure(
        real_results, gen_results, noise_real, noise_gen,
        str(Path(args.output) / "comparison_overview.png"),
    )


def _cli_viz() -> None:
    """
    vn-viz: visualize a single metrics JSON (real or generated).

    Usage examples
    ──────────────
    vn-viz results/metrics/real.json
    vn-viz results/metrics/svd_gaussian.json --label "SVD gaussian"
    vn-viz results/metrics/real.json --noise_stats results/noise_stats/real.json
    vn-viz results/metrics/real.json --output results/plots/
    """
    parser = argparse.ArgumentParser(
        description=(
            "Visualize metrics from a single dataset.\n"
            "Produces 4–5 numbered PNG figures in results/plots/<label>/\n\n"
            "  01_temporal_coherence.png\n"
            "  02_motion_dynamics.png\n"
            "  03_spectral.png\n"
            "  04_frame_content.png\n"
            "  05_noise_stats.png  (requires --noise_stats)"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to metrics JSON  (step 02 output)\n"
             "  e.g. results/metrics/real.json",
    )
    parser.add_argument(
        "--label", default=None,
        help="Title label / subfolder name (defaults to JSON filename stem)",
    )
    parser.add_argument(
        "--noise_stats", default=None,
        help="Path to noise-stats JSON  (step 03 output, optional)\n"
             "  e.g. results/noise_stats/real.json",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Directory to save plots in (default: <input_dir>/plots/)\n"
            "e.g. results/real/plots/  or  results/modelscope_gaussian/plots/"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    label      = args.label or input_path.stem

    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}")
        raise SystemExit(1)

    # Default: sibling plots/ directory next to the input JSON
    output_dir = args.output or str(input_path.parent / "plots")

    results       = load_json(str(input_path))
    noise_results = load_json(str(args.noise_stats)) if args.noise_stats and Path(args.noise_stats).exists() else None

    if args.noise_stats and noise_results is None:
        print(f"  [warn] Noise stats file not found: {args.noise_stats} — skipping panel 05")

    print_single_table(results, label)
    plot_all_single(results, label, output_dir, noise_results=noise_results)
