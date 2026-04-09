# Metrics & Analysis Reference — videonoise

**Central research question:** do real videos have a non-Gaussian, non-i.i.d. latent noise structure, and can we learn a prior over it to improve generation fidelity?

**How to read numbers:** almost every metric is only meaningful *relative to the same real-video reference*. The gap between real and generated — not an absolute threshold — is the signal.

---

## Table of Contents

1. [Implemented metrics](#1-implemented-metrics)
   - [Step 02 — Temporal coherence](#step-02--temporal-coherence)
   - [Step 02 — Motion](#step-02--motion)
   - [Step 02 — Spectral](#step-02--spectral)
   - [Step 02 — Frame statistics](#step-02--frame-statistics)
   - [Step 03 — Noise inversion](#step-03--noise-inversion)
   - [Step 04 — Spatio-temporal PCA / 3D spectrum](#step-04--spatio-temporal-pca--3d-spectrum)
2. [Comparison figure guide](#2-comparison-figure-guide)
3. [Proposed additional analyses](#3-proposed-additional-analyses)
4. [Noise ablation interpretation](#4-noise-ablation-interpretation)
5. [Connecting evidence to a learnable prior](#5-connecting-evidence-to-a-learnable-prior)

---

## 1. Implemented metrics

---

### Step 02 — Temporal coherence

**Output:** `results/<run_key>/metrics.json`  
**Plots:** `results/<run_key>/plots/01_temporal_coherence.png`, `02_motion_dynamics.png`

---

#### Temporal SSIM (TCS) `temporal_ssim.temporal_ssim_mean` ★ primary

SSIM between consecutive frame pairs, averaged over the video. Jointly captures luminance, contrast, and local structural similarity (3 components, all important).

A temporally coherent video has high TCS because adjacent frames look similar at the structural level. Flickering or frame-level noise drives TCS down even when global brightness is stable.

**Why it matters:** TCS is the best single-number proxy for whether a video "looks stable". The real–generated TCS gap directly measures how much temporal incoherence the noise initialization introduces. In the noise ablation, the noise type that minimises this gap (while keeping flow > 0) is the best static prior candidate.

**Plot:** Panel 1 of `01_temporal_coherence.png` — one bar per video, mean ± std in title.

---

#### Frame Pearson r `frame_correlation.pearson_mean`

Linear correlation between consecutive grayscale frames (flattened vectors). High = slower motion / more similar frames.

Useful as a relative indicator but sensitive to global brightness shifts. Compare real vs. generated on matched-content datasets only.

**Plot:** Panel 2, `01_temporal_coherence.png`.

---

#### Frame Spearman ρ `frame_correlation.spearman_mean`

Rank-based version of Pearson — robust to monotone pixel transformations (gamma, tone mapping). A large *Pearson − Spearman* gap signals non-linear artifacts: colour banding, posterisation, or clipping introduced by the model.

**Plot:** Panel 3, `01_temporal_coherence.png`.

---

#### PSNR (frame-to-frame) `psnr.psnr_mean`

Peak Signal-to-Noise Ratio between consecutive frames (dB). Sensitive to mean-squared pixel differences.

**Pitfall:** very high PSNR → the video is nearly static (model collapsed). Very low PSNR → high-frequency flicker. Neither extreme is good. Use alongside TCS and flow magnitude.

**Plot:** Panel 4, `01_temporal_coherence.png`.

---

#### Temporal ACF `temporal_acf.lag_1` … `lag_10`

Autocorrelation of mean pixel intensity across time, for lags 1–10 frames.

| Shape | Interpretation |
|-------|----------------|
| Slow exponential decay | Long temporal memory — characteristic of real footage with continuous motion |
| Fast decay (near zero by lag 2–3) | Near-Markov; each frame ~independent. Common with i.i.d. Gaussian noise init |
| Oscillation | Periodic structure (camera pan, repetitive action) |

**Key use:** read ACF(lag=1) from real videos → use that value as the starting α for AR(1) noise initialization. If generated ACF decays faster than real, the noise init lacks temporal memory.

**Plot:** ACF curve overlay in `02_motion_dynamics.png` — multiple lines, one per video.

---

### Step 02 — Motion

#### Optical flow `optical_flow.flow_mag_{mean,std,p95,p99}`

Dense Farneback optical flow between consecutive frames. All values in pixels/frame.

| Key | Meaning |
|-----|---------|
| `flow_mag_mean` | Average motion magnitude. Scales with content speed and resolution |
| `flow_mag_std` | Spatial variance of the motion field. **Low = coherent, structured motion**; high = spatially incoherent flickering |
| `flow_mag_p95` | 95th-percentile displacement. In real videos: fast objects. In generated videos with Gaussian init: high-frequency noise patches |

**Why it matters:** `flow_mag_std` is arguably more diagnostic than the mean. If generated `flow_mag_std` >> real, the model's motion field is spatially fragmented — a strong sign that the noise init lacks spatial coherence. Spatially low-pass or AR(1) noise typically reduce this gap.

**Plot:** Bar panels in `02_motion_dynamics.png`.

---

### Step 02 — Spectral

#### Spatial spectral slope `spatial_power_spectrum.spectral_slope` ★

Log-log slope of the radially-averaged 2D power spectrum, averaged over frames. Measures the spatial frequency distribution of image content.

Natural images follow an approximate 1/f² power law (Field 1987; Ruderman & Bialek 1994), giving a slope around −2 for luminance. This is content- and resolution-dependent — the meaningful reference is **the slope of your own real-video dataset**, not a universal number.

| Noise type | Expected effect on slope |
|------------|--------------------------|
| Gaussian (i.i.d.) | Flat spectrum, slope pushed toward 0 |
| Spatial low-pass | More low-freq energy → steeper (more negative) slope |
| Blue noise | More high-freq energy → shallower slope |
| AR(1) temporal | Minimal spatial effect; mainly shifts ACF |
| Perlin | Multiple spectral peaks from octaves |

**Why it matters:** if generated slope ≠ real slope, the spatial frequency distribution of the video is wrong, regardless of content. A learnable prior over the spatial noise covariance would correct this automatically.

**Plot:** Power spectrum curves + slope bar in `03_spectral.png`. Dashed reference line = real-dataset mean slope.

---

#### Spectral R² `spatial_power_spectrum.spectral_r2`

Goodness-of-fit of the log-log linear regression used to estimate the slope. Low R² means the spectrum deviates from a clean power law.

**What it reveals:** model artifacts at specific frequencies (e.g. grid patterns from attention layers, aliasing from upsampling) show up as bumps that reduce R². Real footage typically has smooth spectra with R² > 0.9.

**Plot:** R² bar panel in `03_spectral.png`.

---

#### 3D low-frequency energy ratio `spatiotemporal_3d.low_freq_energy_ratio`

Fraction of 3D FFT energy (T×H×W cube) in the lowest-frequency octant (first quarter along each axis). Higher = more energy in large-scale, slow spatiotemporal patterns.

A gap (real > generated) indicates the generated video has excess high-frequency spatiotemporal variation — typically flickering or frame-level noise not anchored to motion.

**Plot:** Bottom-right panel of `03_spectral.png`.

---

### Step 02 — Frame statistics

#### `frame_statistics.{mean, std, skewness, kurtosis}`

Per-video pixel-intensity moments averaged over all frames. Primarily a sanity check.

| Stat | What to check |
|------|---------------|
| `mean` | Overall exposure. Large real–generated gap → tonal shift |
| `std` | Contrast / dynamic range |
| `skewness` | Near 0 = balanced histogram. Large magnitude = over-bright or over-dark |
| `kurtosis` | Large positive = clipping or posterisation (tails heavier than Gaussian) |

**Plot:** `04_frame_content.png`.

---

### Step 03 — Noise inversion

**Output:** `results/<run_key>/noise_stats.json`  
**Plot:** `results/<run_key>/plots/05_noise_stats.png` (auto-added when file exists)

Characterises the noise latent recovered by pixel-domain DDPM inversion.

- **Generated videos (known seed):** verifies inversion quality and baseline.
- **Real videos:** the key signal — reveals the noise structure the model would need to reproduce real footage from i.i.d. Gaussian init. Deviations directly motivate structured initialization.

---

#### KL from N(0,1) `statistics.kl_from_gaussian`

KL divergence (nats) of the recovered noise marginal from a standard Gaussian.

**Research signal:** if inverting real videos gives significantly higher KL than inverting generated ones, it means real footage cannot be explained by i.i.d. Gaussian noise latents — the most direct evidence for a structured prior.

---

#### KS normality p-value `statistics.ks_p_value`

Kolmogorov-Smirnov test for whether the recovered noise is Gaussian.

| p-value | Interpretation |
|---------|----------------|
| > 0.05 | Cannot reject Gaussianity at 5% |
| < 0.05 | Significantly non-Gaussian |
| < 0.001 | Strongly non-Gaussian |

Ideal i.i.d. generated latents: p > 0.05. Real-video inverted noise: expected to show p < 0.05 for structured content.

---

#### Noise moments `statistics.{mean, std, skewness, kurtosis}`

Ideal i.i.d. Gaussian latent: mean = 0, std = 1, skewness = 0, kurtosis = 0 (excess).

Deviations:
- `mean` ≠ 0 → systematic inversion bias or DC offset in real footage
- `kurtosis` > 0 → heavier tails than Gaussian → real noise has outlier structure (motivates heavy-tailed prior)
- `skewness` ≠ 0 → asymmetric noise distribution

---

#### Cross-frame noise correlation `cross_frame_correlation.cross_frame_corr_mean` ★

Pearson correlation between consecutive temporal slices of the recovered noise latent. **The most actionable metric from step 03.**

| Value | Interpretation |
|-------|----------------|
| ≈ 0 (generated) | Consecutive latent frames are independent — consistent with i.i.d. Gaussian init |
| > 0 (real) | The recovered noise is temporally correlated — the model needs temporal memory to reproduce real video latents |

**Direct use:** set AR(1) `alpha ≈ cross_frame_corr_mean` from real videos as your first-guess initialization parameter.

---

#### Noise spectral slope `power_spectrum.spectral_slope_1d`

Spectral slope of the recovered noise tensor.

| Slope | Interpretation |
|-------|----------------|
| ≈ 0 | White noise (ideal i.i.d. Gaussian) |
| Negative | Smooth/correlated noise; recovered latent has low-frequency structure |
| Positive | Spiky noise; high-frequency dominated |

If real videos' inverted noise has a negative slope, spatially correlated (low-pass or Perlin) initialization better matches the latent structure.

---

### Step 04 — Spatio-temporal PCA / 3D spectrum

**Output:** `results/<run_key>/spatiotemporal/`

---

#### PCA spatial modes `pca_spatial_mode_{0,1,2}.png`

Video tensor treated as T observations of an (H×W) spatial field. PCA finds patterns that vary most over time.

| Mode appearance | Interpretation |
|-----------------|----------------|
| Sharp, localised | Structured motion (moving object, foreground action) |
| Diffuse, noisy | Unstructured temporal variation (flickering, noise-driven) |

Compare real vs. generated qualitatively: if generated modes are diffuse and noisy while real modes show spatial structure, the model's temporal variation is spatially incoherent — a strong argument for spatially structured noise init.

---

#### PCA temporal signal `pca_temporal_0.png`

Amplitude of mode 0 over frames. Smooth = coherent motion; noisy = frame-independent variation. The smoothness of this curve directly visualises temporal memory in the video.

---

#### 3D power spectrum `3d_spectrum_{temporal,vertical,horizontal}.png`

Log-power of the 3D FFT marginalised along each axis separately.

| Panel | What to look for |
|-------|------------------|
| Temporal | Steep decay → slow motion; flat → rapid frame changes / noise-driven |
| Spatial (V/H) | Compare slopes to real; mismatches indicate spatial frequency imbalance |

If the generated temporal spectrum is flatter than real, the noise init runs too fast temporally — increase AR(1) α.

---

## 2. Comparison figure guide

`results/<run_key>/comparison/comparison_overview.png` — 12-panel real (blue) vs. generated (red).

```
Row 1 — scalar bar charts (mean ± std across all videos):
  [Pearson r]  [Temporal SSIM]  [PSNR]  [Optical flow mean]

Row 2 — curve overlays:
  [Temporal ACF lags 1–10]     [Radial power spectrum]

Row 3 — noise inversion:
  [Noise distributions + Gaussian PDF fit]  [KL from N(0,1)]  [Cross-frame noise corr]
```

**Reading the panels:**
- **Bar height gap:** realism gap for that metric. Smaller = better calibrated noise init.
- **Error bars:** std across videos. Large error bars → more videos needed for reliable conclusions.
- **ACF overlay:** red should overlap blue. Red decays faster → noise init lacks temporal memory.
- **Spectrum overlay:** slopes should be parallel (vertical offset = brightness, acceptable; slope difference = spatial frequency imbalance, problematic).
- **Noise distributions:** red curve deviating from the Gaussian PDF → non-Gaussian latent structure in that dataset.

---

## 3. Proposed additional analyses

These are not yet implemented but are strongly motivated by the research goal.

---

### 3.1 Latent noise covariance structure

**Goal:** show that the noise latent covariance matrix is not the identity — it has structure.

Run DDPM inversion on a set of videos, collect the recovered noise tensors, then compute:
- Full spatial covariance C = E[ε_spatial · ε_spatial^T] — visualise as a heatmap
- Off-diagonal energy / trace ratio (should be 0 for i.i.d. noise)
- Eigenvalue spectrum of C — if there are a few dominant eigenvalues, the noise lives in a low-dimensional subspace (directly motivates a learned latent prior)

**Expected finding:** real videos will show structured covariance (banded, spatially smooth); generated videos with Gaussian init will show near-identity covariance.

---

### 3.2 Temporal mutual information between latent frames

**Goal:** measure statistical *dependence* between frames beyond linear correlation.

Compute the mutual information I(ε_t ; ε_{t+k}) for k = 1 … 10 lags using a k-NN estimator (e.g. scikit-learn `mutual_info_regression`). This captures non-linear temporal dependencies that Pearson r misses.

**Expected finding:** real-video inverted noise shows positive MI at short lags even after accounting for linear correlation — evidence for non-linear temporal structure that a simple AR(1) prior cannot capture, motivating a *learned* (non-parametric) prior.

---

### 3.3 Flow direction entropy

**Goal:** measure whether the motion field is spatially organised (low entropy) vs. chaotic (high entropy).

Compute the histogram of optical flow *directions* (0–360°) for each frame pair. Calculate Shannon entropy of this histogram. Real videos with camera motion or object motion have low directional entropy (all vectors point roughly the same way). Flickering from poor noise init produces high-entropy, nearly uniform direction histograms.

```python
angles = np.arctan2(flow[..., 1], flow[..., 0])   # per-pixel flow direction
hist, _ = np.histogram(angles, bins=36)
entropy = -np.sum(p * np.log(p + 1e-8) for p in hist / hist.sum())
```

**Expected finding:** generated (Gaussian init) > real direction entropy. Low-pass or AR(1) noise init reduces entropy by making motion more spatially coherent.

---

### 3.4 Paired video analysis (DeepAction matched pairs)

**Goal:** since DeepAction provides exact (real, generated) video pairs for the same action class, compute *per-pair* metric differences — much more statistically powerful than comparing group means.

For each matched pair:
- Δ TCS = TCS_real − TCS_generated
- Δ spectral_slope = slope_real − slope_generated
- Δ ACF_lag1 = ACF_real − ACF_generated

Plot distribution of Δ across pairs. Run paired t-tests / Wilcoxon signed-rank tests.

**Expected finding:** consistent positive Δ TCS and Δ ACF across most pairs (real is always more coherent) — provides controlled evidence free of content-type confounds.

---

### 3.5 Fréchet Video Distance (FVD)

**Goal:** distribution-level comparison using deep video features (I3D / S3D).

FVD measures the Fréchet distance between feature distributions of real and generated video sets. Lower = more similar. Widely used as the video equivalent of FID.

```bash
pip install frechet-video-distance
```

```python
from frechet_video_distance import frechet_video_distance
fvd = frechet_video_distance(real_videos, generated_videos)
```

**Expected finding:** FVD decreases as noise init becomes more structured (AR1 < Gaussian). Use FVD as the primary optimization target in the noise ablation (step 05), with TCS and spectral slope as diagnostic signals.

---

### 3.6 Frequency-band temporal coherence

**Goal:** measure SSIM at different spatial frequency bands separately, to identify *which scales* are temporally incoherent.

Decompose each frame with a Gaussian pyramid (4 scales) and compute temporal SSIM independently at each scale:
- Scale 0 (original): full detail
- Scale 1–2: mid-frequency structure
- Scale 3 (coarsest): global layout

```python
from skimage.transform import pyramid_gaussian
pyramid = list(pyramid_gaussian(frame, max_layer=3))
```

**Expected finding:** generated videos with Gaussian init are most incoherent at high frequencies (fine details flicker) — Gaussian noise has no spatial structure. Low-pass or AR(1) init reduces high-frequency temporal incoherence first. A learnable prior could target the specific scale where the gap is largest.

---

### 3.7 DDIM inversion trajectory divergence

**Goal:** quantify how much the denoising trajectory of real vs. generated videos diverges at each timestep.

During DDIM inversion, record the intermediate latent x_t at each step t. For real and generated videos with similar content:
- Compute L2 distance ‖x_t(real) − x_t(gen)‖ at each t
- Plot trajectory divergence vs. t

**Expected finding:** trajectories diverge early (high-t, high-noise end) even for matched content — evidence that the *starting point* (noise init) matters, and that real footage requires a different starting distribution than i.i.d. Gaussian.

---

### 3.8 Noise prior capacity analysis

**Goal:** determine how much of the real–generated gap can be explained by parametric priors vs. how much requires a learned prior.

Fit a sequence of increasingly expressive models to the inverted noise statistics of real videos:
1. i.i.d. Gaussian N(0, I) — baseline
2. AR(1) temporal: N(0, Σ_temporal) — captures lag-1 correlation
3. Spatially correlated: N(0, Σ_spatial) — captures 2D spatial structure
4. Full spatio-temporal: N(0, Σ_st) — AR(1) × spatial covariance
5. Learned flow-based prior (normalizing flow on noise latents) — non-parametric upper bound

For each, generate videos and measure FVD / TCS gap against real. Plot the gap as a function of prior expressiveness.

**Expected finding:** the gap decreases monotonically and the residual gap between (4) and (5) justifies training a learned prior, providing both the ablation story and the motivation for the neural model.

---

## 4. Noise ablation interpretation

After running `05_noise_init_ablation.sh`, each noise type produces `results/<model>_<noise>__<settings>/metrics.json`.

**Primary ranking metric:** TCS gap = |TCS_real − TCS_generated| → minimise.  
**Secondary:** FVD (if computed), spectral slope gap, ACF lag-1 gap.

| Noise type | Expected TCS rank | Mechanism |
|------------|-------------------|-----------|
| `ar1` (α ≈ 0.8) | 1st (best) | Temporal memory matches real ACF |
| `spatial_lowpass` | 2nd | Spatial coherence reduces flow_std |
| `perlin` | 3rd | Multi-scale structure helps both |
| `gaussian` | 4th | Baseline; no structure |
| `blue` | 5th (worst) | Anti-correlated noise amplifies flicker |

**Finding the optimal α for AR(1):**
1. Read `cross_frame_corr_mean` from real-video `noise_stats.json` → use as starting α
2. Run a grid search: α ∈ {0.0, 0.3, 0.5, 0.7, 0.9}
3. Plot TCS gap and ACF lag-1 gap vs. α — look for the knee

```bash
for alpha in 0.0 0.3 0.5 0.7 0.9; do
    python -m videonoise.scripts.generate_videos \
        --noise_type ar1 --alpha $alpha --n_videos 10
    python -m videonoise.scripts.compute_metrics \
        --data_real data/matched_pairs/deepaction_all/real \
        --gen_dir   data/generated/hf_ar1/
done
```

---

## 5. Connecting evidence to a learnable prior

The pipeline is designed to build a chain of evidence. Each piece contributes a specific argument:

| Evidence | Metric | Argument |
|----------|--------|----------|
| TCS gap (real > gen) | `temporal_ssim_mean` | Generated videos are temporally incoherent relative to real |
| ACF mismatch | `temporal_acf.lag_1` | Real footage has temporal memory that i.i.d. noise lacks |
| Cross-frame noise correlation > 0 on real | `cross_frame_corr_mean` | The latent space of real footage is temporally structured |
| Noise KL > 0 on real | `kl_from_gaussian` | Real latents cannot be explained by a Gaussian noise model |
| Noise spectral slope ≠ 0 on real | `spectral_slope_1d` | Real latent noise has spatial correlations |
| Covariance structure (proposed 3.1) | off-diagonal energy | The full noise covariance is not the identity |
| MI > 0 beyond linear (proposed 3.2) | mutual information | Temporal structure is non-linear → simple AR(1) insufficient |
| Parametric prior ablation (proposed 3.8) | FVD vs. prior complexity | Residual gap after best parametric prior justifies learned model |
| FVD improves with structured init (proposed 3.5) | FVD | Better init → better perceptual quality at distribution level |

**The argument chain:**
> Real videos have structured, non-Gaussian latent noise (3.1 + 3.2 + KL + ACF).
> Simple parametric priors (AR1, spatial-LP) partially close the real–generated gap (ablation).
> A residual gap remains that parametric priors cannot explain (3.8).
> Therefore a learned prior over the latent noise distribution is warranted and expected to improve generation quality (FVD, TCS).
