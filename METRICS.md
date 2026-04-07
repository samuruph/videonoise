# Metrics Reference — videonoise

Detailed interpretation of every metric produced by the pipeline.  
For a quick lookup table (JSON key → plot file) see the [README](README.md#metrics-reference).

**Central research question:** does noise initialization structure affect temporal coherence and realism in video diffusion models?

**How to interpret numbers:** for most metrics, there is no universal "good" threshold — the right reference is always the same dataset of real videos. Use the real-vs-generated gap, not an absolute value.

---

## Step 02 — `compute_metrics.py`

Output: `results/<run_key>/metrics.json`  
Plots:  `results/<run_key>/plots/01_temporal_coherence.png` … `04_frame_content.png`

---

### Temporal coherence

#### Temporal SSIM (TCS) — `temporal_ssim.temporal_ssim_mean`  ★ primary metric

SSIM between consecutive frame pairs, averaged over the whole video. Jointly measures luminance, contrast, and local structural similarity.

Higher = more temporally coherent. **The meaningful signal is the gap between real and generated videos on the same dataset.** A smaller gap means the noise initialization produces more realistic temporal dynamics.

**Plot:** `plots/01_temporal_coherence.png`, panel 1  
**Research signal:** In the noise ablation (step 05), look for the noise type that minimises the TCS gap relative to real videos while still producing visible motion (i.e. not a trivially static output).

---

#### Frame Pearson r — `frame_correlation.pearson_mean`

Pearson correlation between consecutive grayscale frames (flattened to 1D vectors). Measures linear frame-to-frame similarity; higher = more similar successive frames = slower or smoother motion.

**Plot:** `plots/01_temporal_coherence.png`, panel 2  
**Research signal:** Use as a relative signal — generated should be close to real for the same content type.

---

#### Frame Spearman ρ — `frame_correlation.spearman_mean`

Rank-based version of Pearson — robust to outlier pixels and non-linear brightness shifts. A large gap between Pearson and Spearman signals non-linear artifacts: clipping, posterisation, or colour banding.

**Plot:** `plots/01_temporal_coherence.png`, panel 3

---

#### PSNR (frame-to-frame) — `psnr.psnr_mean`

Peak Signal-to-Noise Ratio between consecutive frames (dB). Sensitive to global pixel brightness differences.

**Note:** Very high PSNR can indicate the model collapsed to a near-constant output (too little motion). PSNR and TCS are complementary — a video can have high PSNR but low TCS when flickering preserves overall brightness but destroys local structure.

**Plot:** `plots/01_temporal_coherence.png`, panel 4

---

#### Temporal ACF — `temporal_acf.lag_1` … `lag_10`

How well mean pixel intensity at frame *t* predicts intensity at frame *t + k*, for lags 1–10.

- **Slow exponential decay** → long temporal memory, typical of real footage with continuous motion.
- **Fast decay** → near-Markov; each frame nearly independent beyond a short window. More common with i.i.d. Gaussian noise init.
- **Oscillation** → periodic motion (camera pan, repetitive action).

**Plot:** `plots/02_motion_dynamics.png`, ACF curves panel  
**Research signal:** Compare the shape of the generated ACF curve against the real video ACF curve directly. A generated video whose ACF decays faster than real footage suggests the noise init lacks temporal memory. `alpha ≈ ACF(lag=1)` of the real video is a reasonable AR(1) starting point.

---

### Motion

#### Optical flow — `optical_flow.flow_mag_{mean,std,p95}`

Dense Farneback optical flow (pixel displacement vectors) between consecutive frames.

- `flow_mag_mean`: average motion magnitude in pixels. Depends strongly on resolution and content type — use comparatively.
- `flow_mag_p95`: 95th percentile displacement. In real videos this comes from fast-moving objects. In generated videos with Gaussian noise, inflated tail values often come from spatially incoherent flickering patches.
- `flow_mag_std`: spatial variance of motion. Lower = more spatially uniform flow = more coherent motion field.

**Plot:** `plots/02_motion_dynamics.png`, bar panels  
**Research signal:** If generated `flow_mag_std` is higher than real, the motion field is spatially incoherent — flickering rather than structured motion. AR(1) or spatially low-pass noise tend to reduce this std.

---

### Spectral

#### Spatial spectral slope — `spatial_power_spectrum.spectral_slope`  ★

Log-log slope of the radially-averaged 2D power spectrum, averaged across frames. Measures how energy is distributed across spatial frequencies.

**Natural images broadly follow a 1/f² power law** (Field 1987; Ruderman & Bialek 1994), so the slope of real natural footage tends to be negative and steep. Use the slope of your real video dataset as the target reference — content type, colour space, and resolution all shift the exact value.

**Plot:** `plots/03_spectral.png`, power spectrum curves + slope bar panel  
**Research signal:** Perlin and spatially low-pass noise push slopes steeper (more low-freq energy); blue noise pushes them flatter. The noise type whose slope matches the real video slope most closely is the most spectrally natural. The dashed reference line in the plot marks the real-dataset mean.

---

#### Spectral R² — `spatial_power_spectrum.spectral_r2`

Coefficient of determination from the log-log linear regression used to estimate the slope. Measures how well the spectrum fits a single power law.

Lower R² = more deviation from a clean 1/f² shape — could indicate periodic artifacts (e.g. grid patterns from attention layers) or multi-scale structure.

**Plot:** `plots/03_spectral.png`, R² bar panel

---

#### 3D low-frequency energy ratio — `spatiotemporal_3d.low_freq_energy_ratio`

Fraction of 3D FFT energy (T × H × W video cube) in the low-frequency octant. Higher = more energy in large-scale, slow spatiotemporal patterns.

Compare generated against real on the same content type. A gap (real > generated) indicates the generated video has more high-frequency spatiotemporal variation than expected — typically flickering or temporal noise.

**Plot:** `plots/03_spectral.png`, bottom-right panel

---

### Frame statistics — `frame_statistics.*`

Per-video pixel intensity statistics, averaged over frames. Primarily a sanity-check for exposure and distribution issues in the dataset.

- `mean`: average pixel brightness (normalised 0–1).
- `std`: contrast / dynamic range.
- `skewness`: symmetry of the brightness histogram. Values near 0 indicate a balanced distribution; large magnitude indicates a bias toward dark or bright pixels.
- `kurtosis` (excess): tail heaviness relative to Gaussian. Strongly positive values indicate clipping or banding.

Compare distributions directly between real and generated rather than checking against fixed ranges.

**Plot:** `plots/04_frame_content.png`

---

## Step 03 — `noise_inversion.py`

Output: `results/<run_key>/noise_stats.json`  
Plot:   `results/<run_key>/plots/05_noise_stats.png` (auto-added by `viz.sh` when file exists)

Characterises the noise latent recovered by DDPM inversion.  
- **Generated videos (known seed):** verifies inversion quality.  
- **Real videos:** reveals what noise structure the model needs to "explain" real footage — the key signal for motivating structured initialization.

---

#### KL from N(0,1) — `statistics.kl_from_gaussian`

Divergence (nats) of the recovered noise distribution from a standard Gaussian. Values close to 0 indicate the inverted noise is nearly Gaussian.

**Research signal:** A meaningfully larger KL when inverting **real videos** vs. generated videos means real footage cannot be explained by i.i.d. Gaussian noise — directly motivates structured initialization.

---

#### KS normality p-value — `statistics.ks_p_value`

Kolmogorov-Smirnov test for Gaussianity.

- p > 0.05 → cannot reject Gaussianity at 5% significance.
- p < 0.05 → significantly non-Gaussian.
- p < 0.001 → strongly non-Gaussian.

---

#### Noise moments — `statistics.{mean,std,skewness,kurtosis}`

For ideal i.i.d. Gaussian latent noise: mean = 0, std = 1, skewness = 0, kurtosis = 0.

Deviations indicate: systematic inversion bias (`mean` ≠ 0), heavier-than-Gaussian tails (`kurtosis` > 0), or asymmetry (`skewness` ≠ 0).

---

#### Cross-frame noise correlation — `cross_frame_correlation.cross_frame_corr_mean`  ★

Pearson correlation between consecutive time-slices of the recovered noise latent. **The most actionable inversion metric.**

- Near 0 for generated videos → consecutive latent frames are independent, consistent with i.i.d. Gaussian init.
- Positive for real videos → the model would need temporally correlated noise to reproduce the real video's latent statistics. **Use `alpha ≈ cross_frame_corr_mean` as a starting point for AR(1) init.**

---

#### Noise spectral slope — `power_spectrum.spectral_slope_1d`

Spectral slope of the flattened noise tensor (spatial + temporal).

- Slope ≈ 0 → white noise (i.i.d. Gaussian, as expected for ideal inversion).
- Negative slope → low frequencies are stronger than expected; inverted noise has smoothness/correlations.
- Positive slope → high frequencies are stronger.

**Research signal:** If the inverted noise of real videos has a markedly negative slope, spatially low-pass or Perlin initialization may better match the latent statistics than Gaussian.

---

## Step 04 — `spatiotemporal_analysis.py`

Output: `results/<run_key>/spatiotemporal/`

---

#### PCA spatial modes — `pca_spatial_mode_{0,1,2}.png`

Video tensor treated as T observations of an (H × W) spatial field. PCA finds spatial patterns that vary most over time.

- Mode 0 (highest variance): dominant region changing across frames — typically the main moving object or camera motion direction.
- Modes 1–3: progressively subtler patterns.
- **Sharp, localised modes** → structured motion (foreground object).
- **Diffuse, noisy modes** → unstructured temporal variation (flickering or noise-driven).

Compare real vs. generated qualitatively: if generated modes look diffuse and noisy while real modes show clear spatial structure, the model's temporal variation is spatially incoherent.

---

#### PCA temporal signal — `pca_temporal_0.png`

Amplitude of mode 0 over frames. A smooth curve indicates coherent motion; a noisy curve indicates frame-independent variation.

---

#### 3D power spectrum — `3d_spectrum_{temporal,vertical,horizontal}.png`

Log-power of the 3D FFT marginalised along each axis.

- **Steep temporal spectrum** (power concentrated at low temporal frequencies) → slow, smooth motion.
- **Flat temporal spectrum** → rapid frame-to-frame changes — more noise-like.

**Research signal:** If the real video temporal spectrum is steeper than the generated one, the model's temporal dynamics run too fast. Increasing AR(1) α typically steepens this spectrum.

---

## Step 06 — `comparison_overview.png`

`results/<run_key>/comparison/comparison_overview.png` — 12-panel figure comparing real (blue) vs. generated (red).

```
Row 1 — scalar bar charts (mean ± std across all videos):
  [Pearson r]  [Temporal SSIM]  [PSNR]  [Optical flow mean]

Row 2 — curve overlays:
  [Temporal ACF lags 1–10]  [Radial power spectrum]

Row 3 — noise inversion panels:
  [Noise distributions + Gaussian PDF fit]  [KL from N(0,1)]  [Cross-frame noise corr]
```

**How to read it:**
- **Bar height gap** = realism gap for that metric. Smaller = better.
- **Error bars** = std across videos. Large error bars mean results may not be reliable with few videos.
- **ACF overlay** (row 2, left): generated curves (red) should overlap real curves (blue) for a well-calibrated noise type.
- **Spectrum overlay** (row 2, right): slopes should be parallel. A vertical shift = brightness/contrast offset (acceptable); a slope difference = spatial frequency imbalance (model issue).

---

## Step 05 — Noise ablation interpretation

After running `05_noise_init_ablation.sh`, results are in `results/<model>_<noise_type>/metrics.json` for each noise type.

**Hypothesis to test (expected ordering, most → least temporally coherent):**  
`ar1 (α ≈ 0.8) > spatial_lowpass > perlin > gaussian > blue`

To find the optimal α for AR(1) noise:

```bash
for alpha in 0.0 0.3 0.5 0.7 0.9; do
    python -m videonoise.scripts.generate_videos \
        --noise_type ar1 --alpha $alpha --n 10 \
        --output data/generated/ar1_$alpha/
    python -m videonoise.scripts.compute_metrics \
        --input  data/generated/ar1_$alpha/ \
        --output results/ar1_$alpha.json
done
```

Plot TCS and spectral slope vs. α. The optimal α minimises the gap to real videos on both metrics while keeping motion realistic (non-static output).
