# videonoise

Research toolkit for analysing noise initialization in video diffusion models.

The goal is to understand the correlation structure of noise in both real and generated videos, run noise inversion to characterise what noise the model "uses", and find initialization strategies that improve generation realism and temporal consistency.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
3. [Datasets](#datasets)
4. [Models](#models)
5. [Running the Pipeline](#running-the-pipeline)
6. [Scripts Reference](#scripts-reference)
7. [Metrics Explained](#metrics-explained)
8. [Noise Initializations](#noise-initializations)
9. [Using External Repos and Models](#using-external-repos-and-models)
10. [Notebook](#notebook)
11. [Results Layout](#results-layout)

---

## Project Structure

```
videonoise/
├── videonoise/               # Python package (pip install -e .)
│   ├── io/video.py           # Video loading / saving
│   ├── metrics/
│   │   ├── correlation.py    # Frame correlation, ACF, 3D spatio-temporal
│   │   ├── quality.py        # SSIM, PSNR, optical flow
│   │   └── spectral.py       # 2D/3D power spectrum
│   ├── noise/
│   │   ├── generators.py     # Noise init strategies (Gaussian, AR1, blue, Perlin …)
│   │   └── inversion.py      # DDIM inversion + noise statistics
│   ├── diffusion/
│   │   └── pipelines.py      # SVD pipeline wrapper + custom-noise injection
│   ├── analysis/
│   │   ├── spatiotemporal.py # PCA, UMAP
│   │   └── plots.py          # Comparison figures, summary tables
│   ├── scripts/              # CLI entry points (python -m videonoise.scripts.X)
│   └── utils.py              # Device selection, JSON I/O
├── scripts/
│   └── run_experiments.sh    # End-to-end bash pipeline
├── notebooks/
│   └── 01_analysis.ipynb     # Interactive exploration
├── data/
│   ├── real/                 # Real video datasets
│   └── generated/            # Generated videos
├── results/
│   ├── real/                 # metrics, plots, spatiotemporal for real videos
│   ├── <model>_<noise>/      # same structure per generated run (mirrors data/generated/)
│   └── noise_init/           # global noise shape visualizations (step 05)
├── models/                   # Pretrained weights (gitignored)
├── environment.yml
└── pyproject.toml
```

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate videonoise
pip install -e .
```

> **MPS (Apple Silicon):** PyTorch MPS is included automatically. Verify with:
> ```python
> import torch; print(torch.backends.mps.is_available())
> ```

> **Linux / NVIDIA GPU:** replace the torch lines in `environment.yml` with:
> ```
> - torch==2.3.1+cu118
> - torchvision==0.18.1+cu118
> - torchaudio==2.3.1+cu118
> ```
> and add `--extra-index-url https://download.pytorch.org/whl/cu118` in a pip section.

This makes `videonoise` importable everywhere in the environment and registers the `vn-*` console scripts.

### 6. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name videonoise --display-name "Python 3 (videonoise)"
```

---

## Datasets

### Option A — Synthetic test videos (no download required)

Generates smooth gradient videos to verify the pipeline works end-to-end:

```bash
python -m videonoise.scripts.download_data --dataset synthetic \
    --output data/real/ --n_synthetic 10
```

### Option B — DAVIS 2017

High-quality, densely annotated video object segmentation dataset (~2 GB).  
Homepage: https://davischallenge.org/

```bash
python -m videonoise.scripts.download_data --dataset davis --output data/real/
```

This downloads the 480p trainval split, extracts it, and converts each image sequence to an MP4 using `ffmpeg`.

### Option C — UCF-101

Action recognition dataset with 13,320 clips across 101 categories.  
Homepage: https://www.crcv.ucf.edu/data/UCF101.php

```bash
# Manual download
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar data/real/
```

### Option D — Kinetics-400 (subset)

Use the `kinetics-dataset` downloader for a manageable subset:

```bash
pip install kinetics-downloader
# Download 100 clips from 10 categories
kd download --classes "basketball,swimming,dancing" \
    --num-clips 30 --output-dir data/real/
```

### Option E — Custom videos

Place any `.mp4` files directly into `data/real/` or `data/generated/`. The pipeline discovers all `.mp4` files in a given folder automatically.

---

## Models

All models are downloaded on first use from HuggingFace and cached in `~/.cache/huggingface/`.  
Store any manually downloaded weights in `models/` (gitignored).

### Stable Video Diffusion (SVD)

Image-conditioned video generation from Stability AI.  
Paper: [arXiv 2311.15127](https://arxiv.org/abs/2311.15127)  
HuggingFace: `stabilityai/stable-video-diffusion-img2vid-xt`

```bash
python -m videonoise.scripts.generate_videos \
    --model_id stabilityai/stable-video-diffusion-img2vid-xt \
    --noise_type gaussian --n 10 --output data/generated/svd_gaussian/
```

Requires ~14 GB VRAM (fp16) or ~28 GB RAM (fp32). On CPU/MPS it is slow but functional.

### ModelScope Text-to-Video

Text-conditioned model, good for generating diverse content.  
HuggingFace: `damo-vilab/text-to-video-ms-1.7b`

```python
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b",
                                          torch_dtype=torch.float16)
pipe = pipe.to("cuda")
frames = pipe("a dog running on a beach", num_frames=16).frames[0]
```

### VideoCrafter2

High-quality text-to-video model with strong motion fidelity.  
Repo: https://github.com/AILab-CVC/VideoCrafter  
Weights: https://huggingface.co/VideoCrafter/VideoCrafter2

```bash
git clone https://github.com/AILab-CVC/VideoCrafter
cd VideoCrafter
pip install -r requirements.txt
# Download weights to models/videocrafter2/
wget -P models/videocrafter2/ \
    https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt
```

### CogVideoX

State-of-the-art open-source video generation model.  
HuggingFace: `THUDM/CogVideoX-5b`

```python
from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b",
                                          torch_dtype=torch.bfloat16)
```

### Wan2.1 (Wanvideo)

Latest high-resolution video diffusion model.  
HuggingFace: `Wan-AI/Wan2.1-T2V-14B`

---

## Running the Pipeline

### Configuration

All settings live in one file: **`scripts/config.yaml`**. Edit it before running any step.

```yaml
model: svd          # svd | svd_t2v | modelscope | cogvideox | wan
noise_type: gaussian # gaussian | ar1 | spatial_lowpass | blue | perlin
n_videos:   4
num_frames: 8       # keep ≤ 8 on MPS; up to 25 on CUDA
num_steps:  20
data_real:  data/real/DAVIS/JPEGImages/480p/
active_prompt: "A dog running on a beach"
```

### Run the full pipeline

```bash
conda activate videonoise
pip install -e .            # first time only
bash scripts/run_all.sh
```

### Run individual steps

Each step reads config from `scripts/config.yaml` and can be run independently:

```bash
bash scripts/steps/00_download_data.sh        # download / generate real reference videos
bash scripts/steps/01_generate_videos.sh      # generate with chosen model + noise  [GPU]
# — OR, if you have no GPU / want pre-existing AI outputs: —
bash scripts/steps/01b_download_generated.sh  # download pre-generated AI videos    [no GPU]
bash scripts/steps/01c_download_matched.sh    # download matched (gen, real) pairs   [no GPU]
bash scripts/steps/02_compute_metrics.sh      # correlation, spectral, quality metrics
bash scripts/steps/03_noise_inversion.sh      # DDPM inversion + noise stats
bash scripts/steps/04_spatiotemporal.sh       # PCA + 3D power spectrum
bash scripts/steps/05_noise_init_ablation.sh  # sweep all 5 noise types
bash scripts/steps/06_compare_results.sh      # summary table + comparison figure
```

### Step 01b — Download pre-generated videos (no GPU required)

`01b_download_generated.sh` is an alternative to `01_generate_videos.sh` for machines without a GPU.
It downloads existing AI-generated video clips and places them in `data/generated/<run_key>/` so all downstream steps work identically.

**Backends:**

| Backend | Description | API key |
|---------|-------------|---------|
| `archive` | Public-domain / CC clips from archive.org | None |
| `hf_generated` | AI-generated evaluation sets from HuggingFace Hub | None |
| `pexels` | Short clips from pexels.com | Free key required |

```bash
# Internet Archive — nature clips (RECOMMENDED, easiest)
bash scripts/steps/01b_download_generated.sh

# Internet Archive — custom query
bash scripts/steps/01b_download_generated.sh archive "city street"

# HuggingFace — VBench 2.0 CogVideo outputs (30k videos, default for hf_generated)
bash scripts/steps/01b_download_generated.sh hf_generated vbench2

# HuggingFace — CogVideoX-2b / 5b outputs (78 videos)
bash scripts/steps/01b_download_generated.sh hf_generated cogvideox

# HuggingFace — Wan T2V baseline outputs (121 videos)
bash scripts/steps/01b_download_generated.sh hf_generated wan

# Pexels — free API key required (https://www.pexels.com/api/)
PEXELS_KEY=your_key bash scripts/steps/01b_download_generated.sh pexels "nature"
```

### Step 01c — Download matched (generated, real) pairs (no GPU required)

`01c_download_matched.sh` downloads pairs of videos that share the same semantic prompt — one AI-generated (VBench 2.0 from HuggingFace) and one real (Internet Archive). This is the most direct setup for comparing generated vs. real content.

```bash
# 50 matched pairs, balanced across all VBench2 categories (default)
bash scripts/steps/01c_download_matched.sh

# 100 pairs
bash scripts/steps/01c_download_matched.sh --n 100

# Only Camera_Motion category, CogVideo model
bash scripts/steps/01c_download_matched.sh \
    --category_filter Camera_Motion --model_filter CogVideo
```

Output is written to `data/matched_pairs/<model>_<category>/` with a `pairs.json` cross-reference.

Visualise intermediate results at any point (no second dataset required):

```bash
bash scripts/steps/viz.sh --real   # after step 02, real only
bash scripts/steps/viz.sh --gen    # after step 02, generated only
vn-viz results/metrics/real.json   # same, directly with label
```

---

## Scripts Reference

| Script | Console cmd | Purpose |
|--------|-------------|---------|
| `download_data.py` | — | Download DAVIS or create synthetic videos |
| `generate_videos.py` | `vn-generate` | Generate videos with any model + custom noise |
| `compute_metrics.py` | `vn-metrics` | Correlation, spectral, quality metrics → JSON |
| `noise_inversion.py` | `vn-inversion` | Invert noise + statistical characterisation → JSON |
| `noise_init.py` | `vn-noise-init` | Generate & compare all noise types |
| `spatiotemporal_analysis.py` | `vn-stanalysis` | PCA, UMAP, 3D FFT → plots |
| `compare_results.py` | `vn-compare` | Summary table + comparison figure (needs both datasets) |
| `viz_metrics.py` | `vn-viz` | Visualise a single metrics JSON (one dataset) |

### Common flags

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | `svd`, `svd_t2v`, `modelscope`, `cogvideox`, `wan` | `svd` |
| `--noise_type` | `gaussian`, `ar1`, `spatial_lowpass`, `blue`, `perlin` | `gaussian` |
| `--input` | Folder of videos or JPEG-sequence subfolders | — |
| `--output` | Output path (JSON or folder) | — |
| `--max_frames` | Max frames to read per video | 64 |
| `--resize W H` | Resize frames before processing | original size |

---

## Metrics Explained

This section explains every metric computed by the pipeline, what it measures, how to read the numbers, and — most importantly — what to look for when comparing real vs. generated videos.

The central research question is: **does the noise initialization structure affect how realistic and temporally coherent the generated video is?** Each metric probes a different aspect of that question.

---

### Step 02 — Temporal & correlation metrics

---

#### Frame-to-frame Pearson correlation  
`frame_correlation.pearson_mean`

Pearson *r* between consecutive grayscale frames (each frame is flattened to a 1D vector), averaged over all frame pairs in the video.

| Value | What it means |
|-------|---------------|
| **0.95 – 0.99** | Smooth, natural motion (real footage, slow camera) |
| **0.80 – 0.95** | Moderate motion; normal for action scenes |
| **< 0.80** | Fast motion, scene cuts, or flickering artifacts |

**Research signal:** If generated videos score lower than real at the same content type, the model produces frame-to-frame discontinuities. Try increasing `alpha` in AR(1) noise to smooth the temporal latent.

---

#### Spearman correlation  
`frame_correlation.spearman_mean`

Rank-based version of Pearson — less sensitive to outlier pixel values and non-linear brightness shifts. Interpret the same way. A large gap between Pearson and Spearman signals non-linear artifacts (clipping, posterisation, colour shifts).

---

#### Temporal Autocorrelation Function (ACF)  
`temporal_acf.lag_1` … `temporal_acf.lag_10`

How well the mean pixel intensity at frame *t* predicts intensity at frame *t + k*, for lags 1–10.

**How to read the ACF curve:**

- **Slow exponential decay** (lag-1 ≈ 0.90, lag-10 ≈ 0.50) → long temporal memory, typical of real camera footage with continuous motion.
- **Fast decay** (lag-1 ≈ 0.90, lag-5 ≈ 0.10) → near-Markov; each frame is nearly independent beyond a short window. Common in generated videos with i.i.d. Gaussian noise.
- **Oscillation** → periodic motion (camera pan, repetitive action).

The lag at which the real video ACF falls below 0.1 is a practical guide for setting the AR(1) `alpha` parameter: `alpha ≈ ACF(lag=1)` is a good starting point.

**Research signal:** Generated videos with Gaussian noise typically decay faster than real videos at lags > 3. This gap is the primary motivation for using temporally correlated noise.

---

#### 3D low-frequency energy ratio  
`spatiotemporal_3d.low_freq_energy_ratio`

Fraction of the 3D FFT energy (T × H × W video cube) in the low-frequency octant.

| Value | What it means |
|-------|---------------|
| **> 0.70** | Mostly smooth, coherent content — typical of real natural scenes |
| **0.40 – 0.70** | Mixed; moderate motion or texture |
| **< 0.40** | High-frequency dominated — noisy, flickering, or very fast motion |

**Research signal:** Real landscape/nature videos typically score > 0.65. Generated videos with Gaussian noise often score 0.40–0.55. AR(1) or spatially low-pass noise initialization shifts this toward 0.60+.

---

### Step 02 — Spectral metrics

---

#### Spatial power spectrum slope  
`spatial_power_spectrum.spectral_slope`

The log-log slope of the radially-averaged 2D power spectrum, averaged across frames. This measures how spatial frequency energy is distributed across a frame.

**Natural images obey a 1/f² power law**, so the expected value is **≈ −2.0**.

| `spectral_slope` | Interpretation |
|------------------|---------------|
| **−1.8 to −2.2** | Consistent with natural image statistics |
| **> −1.5** (flatter) | Over-sharpened or noisy at high frequencies |
| **< −2.5** (steeper) | Over-smoothed or blurry frames |

**Research signal:** Perlin and spatially low-pass noise initializations tend to push generated frames toward steeper slopes (more low-frequency energy). Blue noise pushes toward flatter slopes. The noise type that produces slopes closest to −2.0 for your content type is the most natural.

---

#### Spectral R²  
`spatial_power_spectrum.spectral_r2`

How well the power spectrum fits a power law (coefficient of determination from the log-log linear regression).

- **R² > 0.95** → well-described by a power law; natural.
- **R² < 0.80** → spectrum has bumps or breaks at specific frequencies, suggesting model artifacts.

---

### Step 02 — Quality metrics

---

#### Temporal Consistency Score (TCS) / Temporal SSIM  
`temporal_ssim.temporal_ssim_mean`

SSIM (Structural Similarity Index) between each pair of consecutive frames, averaged over the whole video. SSIM jointly measures luminance similarity, contrast similarity, and local structural similarity.

| TCS value | Interpretation |
|-----------|---------------|
| **0.90 – 1.00** | Excellent temporal coherence; very smooth video |
| **0.75 – 0.90** | Good; some motion but structurally consistent |
| **0.50 – 0.75** | Moderate; visible temporal changes, possibly flickering |
| **< 0.50** | Poor coherence; strong flickering or large scene changes |

> **TCS is the single most important metric for this project** — it directly quantifies whether consecutive frames look structurally similar, which is what "temporal coherence" means perceptually.

**Research signal:** The noise ablation (step 05) should show TCS increasing as `alpha` increases from 0 to ~0.8, then decreasing at very high alpha (over-correlated noise makes videos look static). The optimal alpha is where TCS peaks while optical flow magnitude remains realistic.

---

#### PSNR between consecutive frames  
`psnr.psnr_mean` (dB)

Peak Signal-to-Noise Ratio between frame pairs. Sensitive to overall pixel brightness differences.

| PSNR (dB) | Interpretation |
|-----------|---------------|
| **> 35 dB** | High similarity; near-static content |
| **25 – 35 dB** | Normal motion |
| **< 25 dB** | Large frame differences; fast motion or flickering |

> Unlike image restoration benchmarks, *excessively high PSNR* here could indicate the model collapsed to a near-constant output (too little motion).

PSNR and TCS complement each other: PSNR is sensitive to global brightness changes; TCS is sensitive to local structural changes. A video can have high PSNR but low TCS if it has flickering that preserves overall brightness.

---

#### Optical flow magnitude  
`optical_flow.flow_mag_mean`, `flow_mag_p95`, `flow_mag_std`

Dense Farneback optical flow (pixel displacement vectors) between consecutive frames.

| `flow_mag_mean` (px) | Typical scene |
|----------------------|---------------|
| **0 – 1** | Near-static (talking head, locked-off camera) |
| **1 – 5** | Moderate motion (walking, slow pan) |
| **5 – 20** | Fast motion (sports, fast camera) |

`flow_mag_p95` and `flow_mag_p99` are the 95th/99th percentile displacements. In real videos, these tail values come from fast-moving objects. In generated videos with Gaussian noise, high tail values often correspond to small flickering patches rather than real motion — a tell-tale generation artifact.

**Research signal:**
- Real videos: spatially smooth, coherent flow fields (displacement is organised).
- Generated videos (Gaussian): irregular flow with higher `flow_mag_std` (displacement variance).
- AR(1) or spatially low-pass noise: smoother, more organised flow (lower std, lower p99).

---

#### Frame statistics  
`frame_statistics.mean / std / skewness / kurtosis`

Per-video pixel intensity statistics, averaged over frames.

| Statistic | Natural range | Flag if… |
|-----------|---------------|----------|
| `mean` | 0.35 – 0.65 | < 0.15 or > 0.85 (very dark or overexposed) |
| `std` | 0.10 – 0.30 | < 0.05 (low contrast / flat) or > 0.40 (oversaturated) |
| `skewness` | −0.5 – 0.5 | \|skew\| > 1.5 (strong brightness bias) |
| `kurtosis` | 0 – 5 | > 10 (very spiky histogram — banding / clipping) |

---

### Step 03 — Noise inversion metrics

These metrics characterise the noise latent recovered by running the DDPM forward process in reverse.  
For **generated videos** (known seed), this verifies inversion quality.  
For **real videos**, it reveals what noise structure the model would need to "explain" them — the key signal for motivating structured initialization.

---

#### KL divergence from N(0,1)  
`statistics.kl_from_gaussian`

How many nats the recovered noise distribution is away from a standard Gaussian.

| KL value | Interpretation |
|----------|---------------|
| **< 0.05** | Nearly Gaussian — inversion is working; noise is well-behaved |
| **0.05 – 0.20** | Mild non-Gaussianity — slight bias or scale mismatch |
| **> 0.20** | Meaningful structure in the noise |

**Research signal:** If inverting **real videos** gives KL > 0.10, their temporal statistics cannot be explained by i.i.d. Gaussian noise. This directly motivates structured initialization.

---

#### Kolmogorov-Smirnov p-value  
`statistics.ks_p_value`

Hypothesis test for Gaussianity.

- **p > 0.05** → cannot reject Gaussianity at 5% significance (noise looks Gaussian).
- **p < 0.05** → significantly non-Gaussian.
- **p < 0.001** → strongly non-Gaussian.

---

#### Noise mean, std, skewness, kurtosis

For ideal i.i.d. Gaussian latent noise:

| Statistic | Ideal |
|-----------|-------|
| `mean` | ≈ 0.0 |
| `std` | ≈ 1.0 |
| `skewness` | ≈ 0.0 |
| `kurtosis` (excess) | ≈ 0.0 |

Deviations indicate the inversion has introduced a systematic bias (`mean` ≠ 0), that the noise has heavier tails than Gaussian (`kurtosis` > 0), or that it is asymmetric (`skewness` ≠ 0).

---

#### Cross-frame correlation of inverted noise  
`cross_frame_correlation.cross_frame_corr_mean`

Pearson correlation between consecutive time-slices of the recovered noise latent.

| Value | Interpretation |
|-------|---------------|
| **≈ 0.0** | Independent slices — consistent with i.i.d. initialization |
| **0.05 – 0.15** | Weak temporal structure |
| **> 0.15** | Strong temporal structure — consecutive latent frames are correlated |

> **This is the most actionable inversion metric.**  
> - Inverted noise of **generated videos** with cross-frame corr ≈ 0: Gaussian initialization was fine.  
> - Inverted noise of **real videos** with cross-frame corr > 0.15: the model needs temporally correlated noise to reproduce those statistics. Set AR(1) `alpha ≈ cross_frame_corr_mean` as a starting point.

---

#### Noise power spectrum slope  
`power_spectrum.spectral_slope_1d`

Spectral slope of the flattened noise tensor (spatial + temporal combined).

- **Slope ≈ 0** → white noise (i.i.d. Gaussian, as expected).
- **Slope < −0.3** → low frequencies are stronger; noise has spatial/temporal smoothness.
- **Slope > +0.3** → high frequencies are stronger (blue-noise-like).

**Research signal:** Inverted real video noise with slope < −0.5 suggests spatially low-pass or Perlin noise would better match the latent statistics than Gaussian.

---

### Step 04 — Spatio-temporal analysis

---

#### PCA spatial modes  
`results/plots/*_st/pca_spatial_mode_*.png`

The video tensor is treated as T observations of an (H × W) spatial field. PCA finds the spatial patterns that vary most over time.

- **Mode 0** (first PC, highest variance): the dominant spatial region that changes most across frames — usually the main moving object or camera motion direction.
- **Modes 1–3**: progressively subtler patterns.
- **Sharp, localised modes** → structured motion (e.g. a moving foreground object).
- **Diffuse, noisy modes** → unstructured temporal variation (flickering or noise-driven changes).

**Research signal:** Compare PCA modes of real vs. generated videos qualitatively. If generated video modes look random and noisy while real video modes show clear spatial structure, the model's temporal variation is spatially incoherent — a noise problem.

---

#### 3D power spectrum plots  
`results/plots/*_st/3d_spectrum_*.png`

Log-power of the 3D FFT marginalised along temporal, vertical, and horizontal frequency axes.

- **Steep temporal spectrum** (power concentrated at low temporal frequencies) → slow, smooth motion over time.
- **Flat temporal spectrum** → rapid frame-to-frame changes — more noise-like.

**Research signal:** If the real video temporal spectrum is steeper than the generated one, the model's temporal dynamics run too fast. Increasing AR(1) `alpha` slows temporal dynamics and typically steepens this spectrum.

---

### Step 05 — Noise initialization ablation

After running the full ablation, compare `results/metrics/<model>_<noise>.json` across all five noise types.  
Key metrics ranked by importance for this research:

| Metric | What the ablation reveals |
|--------|--------------------------|
| `temporal_ssim_mean` | Which noise type produces the most temporally coherent video |
| `pearson_mean` | Which type produces smoothest frame-to-frame transitions |
| `low_freq_energy_ratio` | Which type produces most spatially/temporally coherent content |
| `spectral_slope` | Which type produces frames closest to the natural 1/f² spectrum |
| `flow_mag_std` | Which type produces the most consistent (low-variance) motion |

**Expected ordering from most to least temporally coherent:**  
`ar1 (high alpha) > spatial_lowpass > perlin > gaussian > blue`

This is the hypothesis — the ablation confirms or refutes it.

---

### Step 06 — Comparison figure

`results/plots/comparison/comparison_overview.png` shows 12 panels.

**How to read it:**

- **Bar height gap (real vs. generated)** = the "realism gap" for that metric. Smaller gap = better.
- **Error bars** = standard deviation across videos. Large error bars mean high variance — results may not be reliable with few videos.
- **ACF overlay** (bottom-left): red curves (generated) should overlap blue curves (real) for a well-calibrated noise type.
- **Power spectrum overlay** (bottom-right): slopes should be parallel. A vertical shift = brightness/contrast offset (acceptable); a slope difference = spatial frequency imbalance (model issue).

---

## Noise Initializations

Five strategies are implemented in `videonoise/noise/generators.py`:

| Type | Description | Key parameter |
|------|-------------|---------------|
| `gaussian` | i.i.d. N(0,1) — diffusion model default | — |
| `ar1` | Temporally correlated: `ε_t = α·ε_{t-1} + √(1−α²)·z_t` | `--alpha` ∈ [0, 1) |
| `spatial_lp` | Spatially low-pass filtered Gaussian (smooth blobs) | `--sigma` (pixels) |
| `blue` | High-pass noise (white minus low-pass) — fine-grained texture | — |
| `perlin` | Fractal / Perlin-like via multi-scale spectral synthesis | `--octaves` |

**Ablation experiment** — vary `--alpha` to find the temporal correlation that maximises TCS or minimises FVD:

```bash
for alpha in 0.0 0.3 0.5 0.7 0.9; do
    python -m videonoise.scripts.generate_videos \
        --noise_type ar1 --alpha $alpha --n 10 \
        --output data/generated/ar1_$alpha/
    python -m videonoise.scripts.compute_metrics \
        --input data/generated/ar1_$alpha/ \
        --output results/metrics/ar1_$alpha.json
done
```

---

## Using External Repos and Models

### diffusers (Hugging Face)

The primary interface for SVD, ModelScope, CogVideoX, etc.

```bash
pip install diffusers transformers accelerate
```

Load any supported pipeline:

```python
from diffusers import StableVideoDiffusionPipeline, DiffusionPipeline
import torch

# SVD
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
).to("cuda")

# Inject custom noise from this toolkit
from videonoise.noise import ar1_noise
T, C, h, w = 14, 4, 72, 128   # SVD latent shape
custom_noise = ar1_noise((T, C, h, w), alpha=0.8).half().to("cuda")
frames = pipe(image=cond_img, latents=custom_noise.unsqueeze(0)).frames[0]
```

### VideoCrafter2

```bash
git clone https://github.com/AILab-CVC/VideoCrafter external/VideoCrafter
cd external/VideoCrafter
pip install -r requirements.txt

# Download weights
mkdir -p ../../models/videocrafter2
wget -P ../../models/videocrafter2/ \
    https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt
```

Inference:
```bash
python scripts/evaluation/inference.py \
    --seed 123 \
    --ckpt_path ../../models/videocrafter2/model.ckpt \
    --config configs/inference_t2v_512_v2.0.yaml \
    --savedir results/videocrafter2/ \
    --n_samples 1 --bs 1 --height 320 --width 512 \
    --unconditional_guidance_scale 12.0 \
    --ddim_steps 50 --ddim_eta 1.0 \
    --prompt "a dog running on a beach"
```

### Open-Sora

Fully open-source text-to-video model.

```bash
git clone https://github.com/hpcaitech/Open-Sora external/Open-Sora
cd external/Open-Sora
pip install -v .
```

### RAFT (optical flow)

For high-quality optical flow (better than Farneback):

```bash
git clone https://github.com/princeton-vl/RAFT external/RAFT
cd external/RAFT
pip install -r requirements.txt
./download_models.sh
```

```python
# Use RAFT instead of Farneback in your analysis
sys.path.append('external/RAFT/core')
from raft import RAFT
```

### Fréchet Video Distance (FVD)

```bash
pip install frechet-video-distance
```

```python
from frechet_video_distance import frechet_video_distance
fvd = frechet_video_distance(real_videos, generated_videos)
```

Alternatively use `clean-fid` for frame-level FID:

```python
from cleanfid import fid
score = fid.compute_fid("data/real/frames/", "data/generated/frames/")
```

### LPIPS

```python
import lpips
loss_fn = lpips.LPIPS(net='alex')
d = loss_fn(frame_a, frame_b)   # both (N, C, H, W) in [-1, 1]
```

---

## Notebook

Open `notebooks/01_analysis.ipynb` for interactive exploration:

```bash
conda activate videonoise
jupyter notebook notebooks/01_analysis.ipynb
```

The notebook covers:
1. Loading real and generated videos
2. Frame visualisation strips
3. Temporal ACF comparison (real vs generated)
4. Spatial power spectrum overlays
5. Noise inversion and Gaussianity analysis
6. AR(1) α sweep visualisation
7. All noise types: spatial maps + histograms
8. Temporal consistency score comparison
9. PCA spatial modes

---

## Results Layout

### Naming convention — `<run_key>`

Everything is organized around a **run key**: `<model>_<noise_type>`.

```
modelscope_gaussian   svd_ar1   wan_perlin   cogvideox_blue   …
```

Both generated data and all results use the same key, so you can always match them:

```
data/generated/modelscope_gaussian/   ←→   results/modelscope_gaussian/
```

Real reference videos always use the key `real`.

---

### Full directory tree after running all steps

```
data/
├── real/
│   └── DAVIS/JPEGImages/480p/
│       ├── bear/               # JPEG frame sequences (DAVIS layout)
│       ├── blackswan/
│       └── ...
└── generated/
    ├── modelscope_gaussian/    # step 01 — one folder per run
    │   ├── video_000_seed42.mp4
    │   ├── video_001_seed43.mp4
    │   └── metadata.json       # model, noise, prompt, all generation params
    ├── modelscope_ar1/
    │   └── ...
    └── svd_gaussian/
        └── ...

results/
├── real/                       # all results for real reference videos
│   ├── metrics.json            # step 02 — correlation, spectral, quality metrics
│   ├── noise_stats.json        # step 03 — inverted noise statistics (optional)
│   ├── plots/                  # viz step — 5 themed PNG panels
│   │   ├── 01_temporal_coherence.png
│   │   ├── 02_motion_dynamics.png
│   │   ├── 03_spectral.png
│   │   ├── 04_frame_content.png
│   │   └── 05_noise_stats.png  #   only after step 03
│   └── spatiotemporal/         # step 04 — PCA modes + 3D spectra
│       ├── pca_spatial_mode_0.png
│       ├── pca_temporal_0.png
│       └── 3d_spectrum_*.png
│
├── modelscope_gaussian/        # all results for this run — mirrors data/generated/
│   ├── metrics.json            # step 02
│   ├── noise_stats.json        # step 03 (optional)
│   ├── plots/                  # viz step
│   │   ├── 01_temporal_coherence.png
│   │   ├── 02_motion_dynamics.png
│   │   ├── 03_spectral.png
│   │   ├── 04_frame_content.png
│   │   └── 05_noise_stats.png
│   ├── spatiotemporal/         # step 04
│   │   └── ...
│   └── comparison/             # step 06 — real vs this run
│       └── comparison_overview.png
│
├── modelscope_ar1/             # same structure for every other run
│   └── ...
├── svd_gaussian/
│   └── ...
│
└── noise_init/                 # step 05 — global noise shape comparison (no model)
    ├── noise_init_stats.json
    └── noise_comparison.png
```

**The rule:** for any run key `K`, find everything in one place — `data/generated/K/` for the videos, `results/K/` for all analysis outputs.

---

### What each output file contains

#### `results/<run_key>/metrics.json`  (step 02)

One JSON per dataset. Structure:

```json
{
  "per_video": {
    "blackswan": {
      "frame_correlation":      { "pearson_mean": 0.97, "spearman_mean": 0.96 },
      "temporal_acf":           { "lag_1": 0.94, "lag_2": 0.89, "lag_5": 0.61 },
      "spatial_power_spectrum": { "spectral_slope": -2.1, "spectral_r2": 0.97,
                                  "radial_profile": [1.2, 0.9, ...] },
      "temporal_ssim":          { "temporal_ssim_mean": 0.91 },
      "psnr":                   { "psnr_mean": 32.4 },
      "optical_flow":           { "flow_mag_mean": 1.8, "flow_mag_std": 0.9,
                                  "flow_mag_p95": 5.2, "flow_mag_p99": 9.1 },
      "frame_statistics":       { "mean": 0.48, "std": 0.18,
                                  "skewness": 0.12, "kurtosis": 1.4 },
      "spatiotemporal_3d":      { "low_freq_energy_ratio": 0.68 }
    },
    "bear": { ... }
  },
  "aggregate": {
    "frame_correlation": { "pearson_mean": { "mean": 0.96, "std": 0.02 } },
    ...
  }
}
```

The `aggregate` block contains mean ± std across all videos for every scalar metric — use this for quick comparison between conditions.

---

#### `results/<run_key>/noise_stats.json`  (step 03)

```json
{
  "per_video": {
    "blackswan": {
      "statistics": {
        "mean": 0.01, "std": 0.98, "skewness": 0.04, "kurtosis": 0.11,
        "kl_from_gaussian": 0.03, "ks_p_value": 0.42
      },
      "cross_frame_correlation": { "cross_frame_corr_mean": 0.07 },
      "power_spectrum":          { "spectral_slope_1d": -0.08,
                                   "low_freq_energy_ratio": 0.52 }
    }
  }
}
```

---

#### `results/<run_key>/plots/`  (viz step — `vn-viz` / `viz.sh`)

Running `bash scripts/steps/viz.sh --real` (or `vn-viz results/real/metrics.json`) produces **5 separate PNG files** inside `results/real/plots/`.  Each file focuses on one theme and has exactly 4 panels with annotated titles explaining how to interpret each value.

| File | What you see | Key reference values |
|------|--------------|----------------------|
| `01_temporal_coherence.png` | TCS, Pearson r, Spearman ρ, PSNR — one bar per video | TCS > 0.9, Pearson > 0.95 → smooth |
| `02_motion_dynamics.png` | Optical-flow mean, std, p95 (bar), + temporal ACF curves | Flow mean 1–10 px typical for action |
| `03_spectral.png` | Radial power-spectrum curves, spectral slope (dashed −2 ref), R², 3D low-freq ratio | Slope ≈ −2.0 for natural images |
| `04_frame_content.png` | Frame mean, std, skewness, kurtosis — one bar per video | Mean 0.2–0.8, skew ≈ 0 |
| `05_noise_stats.png` | KL from N(0,1), KS p-value, cross-frame correlation, noise spectral slope | KL ≈ 0, KS p > 0.05 → Gaussian |

> **`05_noise_stats.png` only appears** when `results/<run_key>/noise_stats.json` exists (i.e. step 03 has been run). `viz.sh` picks it up automatically.

Each panel includes a mean ± std annotation in the title so you can assess the dataset at a glance.

**Direct use:**
```bash
# Auto-saves to results/real/plots/
vn-viz results/real/metrics.json

# With noise stats
vn-viz results/real/metrics.json --noise_stats results/real/noise_stats.json
```

---

#### `results/<run_key>/comparison/comparison_overview.png`  (step 06)

12-panel figure comparing real (blue) vs. generated (red):

```
Row 1 — scalar metrics as bar charts (mean ± std across videos):
  [Pearson corr]  [Temporal SSIM]  [PSNR]  [Optical flow mean]

Row 2 — distribution / curve comparisons:
  [Temporal ACF overlay — lags 1-10]  [Spatial power spectrum overlay]

Row 3 — noise inversion panels:
  [Noise distributions (Gaussian PDF fits)]  [Noise KL from N(0,1)]  [Noise cross-frame corr]
```

---

#### `results/<run_key>/spatiotemporal/`  (step 04)

| File | What it shows |
|------|---------------|
| `pca_spatial_mode_0.png` | Heatmap of the dominant spatial pattern that varies over time |
| `pca_spatial_mode_1.png` | Second mode (next most variable pattern) |
| `pca_temporal_0.png` | How the amplitude of mode 0 evolves frame by frame |
| `3d_spectrum_temporal.png` | Log-power vs temporal frequency |
| `3d_spectrum_vertical.png` | Log-power vs vertical (row) frequency |
| `3d_spectrum_horizontal.png` | Log-power vs horizontal (column) frequency |

---

### Quick lookup — "where is X?"

#### Spectral metrics

| I want to see… | Where to find it | Produced by |
|----------------|------------------|-------------|
| Spectral slope number (per video) | `results/real/metrics.json` → `per_video.<name>.spatial_power_spectrum.spectral_slope` | step 02 |
| Spectral R² (how well it fits 1/f²) | same JSON → `spectral_r2` | step 02 |
| Radial power spectrum **curves** + slope bars | `results/real/plots/03_spectral.png` | `viz.sh --real` |
| 3D low-frequency energy ratio | `results/real/plots/03_spectral.png` → bottom-right panel | `viz.sh --real` |
| Radial spectra real **vs** generated overlaid | `results/modelscope_gaussian/comparison/comparison_overview.png` → row 2, right | step 06 |
| 3D spectrum (temporal / H / W axes) for real | `results/real/spatiotemporal/3d_spectrum_*.png` | step 04 |
| 3D spectrum for a generated run | `results/modelscope_gaussian/spatiotemporal/3d_spectrum_*.png` | step 04 |

> **Step 04 must be run** to get the per-video 3D spectrum PNG files.  
> The 2D radial spectrum (slope + curve) is available after step 02 via `vn-viz`.

#### Correlation & quality metrics

| I want to see… | Where to find it | Produced by |
|----------------|------------------|-------------|
| TCS / temporal SSIM per video | `results/real/metrics.json` → `per_video.<name>.temporal_ssim.temporal_ssim_mean` | step 02 |
| TCS, Pearson, Spearman, PSNR — bar charts | `results/real/plots/01_temporal_coherence.png` | `viz.sh --real` |
| Temporal ACF curves for all videos | `results/real/plots/02_motion_dynamics.png` → bottom-right panel | `viz.sh --real` |
| Optical flow mean / std / p95 — bar charts | `results/real/plots/02_motion_dynamics.png` | `viz.sh --real` |
| Frame pixel statistics (mean, std, skew, kurtosis) | `results/real/plots/04_frame_content.png` | `viz.sh --real` |
| Same for a generated run | `results/modelscope_gaussian/plots/01–04_*.png` | `viz.sh --gen` |
| ACF real **vs** generated overlaid | `results/modelscope_gaussian/comparison/comparison_overview.png` → row 2, left | step 06 |

#### Noise inversion

| I want to see… | Where to find it | Produced by |
|----------------|------------------|-------------|
| Whether inverted noise is Gaussian (KL, KS) | `results/real/noise_stats.json` → `statistics.kl_from_gaussian` | step 03 |
| Cross-frame noise correlation number | `results/real/noise_stats.json` → `cross_frame_correlation.cross_frame_corr_mean` | step 03 |
| All noise stats as bar charts | `results/real/plots/05_noise_stats.png` | `viz.sh --real` (after step 03) |
| Noise distributions real **vs** generated | `results/modelscope_gaussian/comparison/comparison_overview.png` → row 3, left | step 06 |

#### Other

| I want to see… | Where to find it | Produced by |
|----------------|------------------|-------------|
| PCA spatial modes for real videos | `results/real/spatiotemporal/pca_spatial_mode_*.png` | step 04 |
| All ablation metrics in one place | `results/modelscope_*/metrics.json` | step 05 |
| Exact generation parameters for a run | `data/generated/modelscope_gaussian/metadata.json` | step 01 |
| All noise shapes visualised side-by-side | `results/noise_init/noise_comparison.png` | step 05 |
