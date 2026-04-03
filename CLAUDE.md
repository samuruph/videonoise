# Video Generation Noise Analysis — Project Plan

## 1. Project Overview

Investigate how noise initialization affects realism and quality in video diffusion models. We analyze real vs. generated videos to understand optimal noise correlation structures, run noise inversion to study what noise was "used," and extract spatio-temporal feature maps for deeper insight. The goal is actionable findings: better noise initializations that improve generation fidelity.

---

## 2. Environment Setup

```bash
conda create -n videonoise python=3.10
conda activate videonoise
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install opencv ffmpeg scikit-image matplotlib seaborn jupyter
pip install einops tqdm imageio diffusers transformers accelerate
pip install decord av lpips clean-fid
```

---

## 3. Folder Structure

```
videonoise/
├── data/
│   ├── real/               # Real video datasets (UCF101, DAVIS, etc.)
│   └── generated/          # Videos from diffusion models
├── notebooks/              # Jupyter notebooks for experiments and visualization
├── scripts/
│   ├── download_data.py
│   ├── generate_videos.py
│   ├── compute_metrics.py
│   ├── noise_inversion.py
│   ├── spatiotemporal_analysis.py
│   └── utils.py
├── results/
│   ├── metrics/            # JSON/CSV files with computed metrics
│   ├── plots/              # Figures and visualizations
│   └── noise_stats/        # Inverted noise statistics
├── models/                 # Pretrained or custom model weights
└── CLAUDE.md
```

---

## 4. Datasets

- **Real Videos:** UCF101, DAVIS 2017, Kinetics-400 (sample subset), custom clips
- **Generated Videos:** Run ModelScope, Stable Video Diffusion (SVD), or VideoCrafter; optionally generate with multiple noise seeds to study variance

---

## 5. Models for Video Generation

- [Stable Video Diffusion (SVD)](https://github.com/Stability-AI/generative-models)
- [ModelScope Text-to-Video](https://github.com/modelscope/modelscope)
- [VideoCrafter](https://github.com/VideoCrafter/VideoCrafter)

Use HuggingFace `diffusers` for easy access to pipelines where available.

---

## 6. Metrics and Analysis Plan

### 6.1. Noise Correlation Metrics (on real video frames and generated video frames)

**Frame-level:**
- Pearson and Spearman correlation between consecutive frames (grayscale and per-channel)
- Autocorrelation function (ACF) along temporal axis — identify correlation decay timescale

**Spatial:**
- 2D spatial autocorrelation of each frame (compute normalized 2D FFT power spectrum)
- Compare power-law exponent of spatial frequency spectrum (real vs. generated)

**Spatio-temporal:**
- 3D correlation tensor across (x, y, t) windows
- Temporal power spectrum — is the noise 1/f, white, or something else?

**Target insight:** What is the optimal correlation structure of noise that produces videos most similar to real ones?

---

### 6.2. Noise Inversion Metrics

Run DDIM inversion (or equivalent) to recover the noise latent from a generated or real video.

**Reconstruction quality:**
- $\|\epsilon_{\text{inv}} - \epsilon_{\text{orig}}\|_2$ (for generated videos where $\epsilon_{\text{orig}}$ is known)
- PSNR and SSIM between reconstructed and original video

**Statistical properties of inverted noise:**
- Mean, variance, skewness, kurtosis — compare to $\mathcal{N}(0,1)$
- KL divergence from standard Gaussian
- Shapiro-Wilk or KS normality test

**Spectral analysis:**
- 1D, 2D, 3D power spectrum of inverted noise maps
- Compare spectral density of noise from real vs. generated videos

**Cross-frame correlation of inverted noise:**
- How correlated is $\epsilon_t$ with $\epsilon_{t+1}$ after inversion?
- Does the model "encode" temporal structure into the noise latent?

---

### 6.3. Spatio-Temporal Map Analysis

Extract intermediate feature maps from the diffusion U-Net (e.g., attention maps, residual activations) at multiple timesteps and spatial scales.

**Visualization:**
- Plot attention maps per head, per layer, per diffusion timestep
- Animate feature maps across diffusion steps to see what structure emerges

**Statistical analysis:**
- PCA of spatiotemporal feature tensors to find dominant modes
- t-SNE / UMAP of feature vectors to cluster video regions by content type
- Measure feature map correlation with motion (optical flow) and semantic regions

**Temporal consistency of feature maps:**
- Compute SSIM / cosine similarity between feature maps at consecutive frames
- Identify which layers maintain temporal coherence vs. which are frame-independent

---

### 6.4. Video Quality Metrics (Real vs. Generated)

| Metric | What it measures |
|--------|-----------------|
| FID / FVD | Distribution distance between real and generated frame/video sets |
| LPIPS | Perceptual frame similarity |
| SSIM / PSNR | Pixel-level frame quality |
| CLIP score | Semantic alignment (text-to-video case) |
| Optical flow smoothness | Mean and std of flow magnitude between frames |
| Temporal SSIM | SSIM computed across the time axis |
| No-reference VQA | BRISQUE or NIQE on individual frames |

---

### 6.5. Motion and Temporal Coherence

- Compute dense optical flow (RAFT or Farneback) between consecutive frames
- Compare flow statistics (magnitude, direction entropy) between real and generated
- Identify failure modes: flickering (high-frequency temporal noise), drift, jitter
- Measure temporal consistency score: $\text{TCS} = \frac{1}{T-1}\sum_t \text{SSIM}(f_t, f_{t+1})$

---

## 7. Experiments

### Exp 1 — Baseline Correlation Analysis
For both real and generated videos:
- Extract all frames as tensors
- Compute all correlation metrics in §6.1
- Plot and compare distributions

### Exp 2 — Noise Inversion Study
For generated videos (with known seeds):
- Run DDIM inversion at multiple guidance scales
- Record inverted noise statistics
- Compare to original sampled noise

For real videos:
- Run DDIM inversion treating the video as "generated"
- Analyze the recovered noise — does it look Gaussian? Correlated?

### Exp 3 — Noise Initialization Ablation
Generate videos with different noise initializations:
- Pure Gaussian (baseline)
- Temporally correlated noise: $\epsilon_t = \alpha \epsilon_{t-1} + \sqrt{1-\alpha^2} z_t$ for varying $\alpha$
- Spatially colored noise (low-pass filtered Gaussian)
- Blue noise (high spatial frequency emphasis)
- Perlin / fractal noise

Measure downstream video quality with metrics from §6.4 for each condition.

### Exp 4 — Optimal Correlation Search
Parameterize noise correlation (e.g., AR(1) temporal coefficient $\alpha$, spatial bandwidth $\sigma$) and run a grid search or Bayesian optimization over these parameters. Objective: maximize FVD or LPIPS against a reference real video set.

### Exp 5 — Spatio-Temporal Feature Analysis
Extract U-Net activations at multiple layers and timesteps for a set of real-inverted and generated videos. Run PCA/UMAP and cluster. Compare cluster composition (real-inverted vs. generated) to find which regions of feature space generated videos fail to cover.

---

## 8. Implementation Notes

- Use `torch.fft.rfftn` for efficient N-D spectral analysis
- Use `diffusers` `DDIMInverseScheduler` for noise inversion
- Use `torchvision.utils` and `matplotlib` for frame-level visualization
- Cache extracted features to disk (`.pt` files) to avoid re-running expensive forward passes
- Log all experiments with a simple JSON config + results file per run

---

## 9. Next Steps (Ordered)

1. [ ] Set up conda environment and verify GPU access
2. [ ] Download DAVIS or a subset of UCF101 as real video reference
3. [ ] Generate a matched set of videos with SVD or ModelScope
4. [ ] Implement `compute_metrics.py` covering §6.1 and §6.4
5. [ ] Implement `noise_inversion.py` with DDIM inversion pipeline
6. [ ] Run Exp 1 and Exp 2 — baseline analysis
7. [ ] Implement noise initialization variants (Exp 3)
8. [ ] Run ablation and identify best-performing noise type
9. [ ] Implement spatio-temporal feature extraction (Exp 5)
10. [ ] Write summary notebook with figures and findings
