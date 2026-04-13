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
│   │   ├── correlation.py    # Frame correlation, ACF, temporal MI, 3D spatio-temporal
│   │   ├── quality.py        # SSIM, PSNR, optical flow, LPIPS, flow entropy, multiscale SSIM
│   │   ├── spectral.py       # 2D/3D power spectrum
│   │   └── distribution.py   # FVD, cross-set LPIPS, CLIP score (dataset-level)
│   ├── noise/
│   │   ├── generators.py     # Noise init strategies (Gaussian, AR1, blue, Perlin …)
│   │   └── inversion.py      # DDIM inversion + noise statistics + covariance structure
│   ├── diffusion/
│   │   └── pipelines.py      # SVD pipeline wrapper + custom-noise injection
│   ├── analysis/
│   │   ├── spatiotemporal.py # PCA, UMAP
│   │   ├── attention.py      # Attention hook registration, aggregation, plotting
│   │   └── plots.py          # All figures 01–08, comparison tables
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

All settings live in one file: **`scripts/config.yaml`**. Edit it for persistent defaults.

```yaml
model:      hf          # svd | modelscope | cogvideox | wan
noise_type: gaussian    # gaussian | ar1 | spatial_lowpass | blue | perlin
n_videos:   10
num_frames: 32          # keep ≤ 8 on MPS; up to 49 on CUDA
num_steps:  50
data_real:  data/matched_pairs/deepaction_all/real
gen_dir:    data/matched_pairs/deepaction_all/generated_CogVideoX5B/
```

### Run the full pipeline

```bash
conda activate videonoise
pip install -e .            # first time only
bash scripts/run_all.sh
```

### Run individual steps

Each step is a Python script that reads from `config.yaml`. Every step also
accepts CLI overrides for any config param — edit the values inside the `.sh`
wrapper or run the Python script directly:

```bash
bash scripts/steps/00_download_data.sh        # download real reference videos
bash scripts/steps/01_generate_videos.sh      # generate with chosen model + noise  [GPU]
# — OR, if you have no GPU / want pre-existing AI outputs: —
bash scripts/steps/01b_download_generated.sh  # download pre-generated AI videos    [no GPU]
bash scripts/steps/01c_download_matched.sh    # download matched (gen, real) pairs   [no GPU]
bash scripts/steps/02_compute_metrics.sh      # correlation, spectral, quality metrics
bash scripts/steps/03_noise_inversion.sh      # DDIM inversion + noise stats
bash scripts/steps/04_spatiotemporal.sh       # PCA + 3D power spectrum
bash scripts/steps/05_noise_init_ablation.sh  # sweep all 5 noise types              [GPU]
bash scripts/steps/06_compare_results.sh      # summary table + comparison figure
bash scripts/steps/07_distribution_metrics.sh # FVD, cross-set LPIPS, CLIP score
bash scripts/steps/08_attention_analysis.sh   # attention map extraction + analysis  [GPU]
bash scripts/steps/09_paired_analysis.sh      # paired Δ analysis + Wilcoxon tests
```

### Overriding settings per run

Every step accepts CLI overrides for any config param. You can either edit the
values directly inside the `.sh` file, or call the Python script with flags:

```bash
# Run step 02 with lower resolution for a quick check
python scripts/steps/compute_metrics.py --max_frames 8 --resize 128 128

# Generate with a different model without touching config.yaml
python scripts/steps/generate_videos.py --model modelscope --noise_type ar1 --n_videos 5

# Visualise only the real dataset
python scripts/steps/viz.py --mode real
```

Any param not passed on the command line falls back to `scripts/config.yaml`.
Adding a new config param only requires changing `config.yaml` and
[`scripts/config_loader.py`](scripts/config_loader.py) — no other file needs to change.

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

`01c_download_matched.sh` downloads pairs of videos with genuine 1-1 semantic correspondence. The default source is **DeepAction v1** (`faridlab/deepaction_v1`, CC BY 4.0) — a HuggingFace dataset that contains both real videos and AI-generated videos organized by the same action class and filename, so each pair shows the same action.

```bash
# 50 matched pairs with CogVideoX5B outputs, balanced across action classes (default)
bash scripts/steps/01c_download_matched.sh

# 100 pairs
bash scripts/steps/01c_download_matched.sh --n 100

# Different AI model
bash scripts/steps/01c_download_matched.sh --gen_model RunwayML
# Available models: BDAnimateDiffLightning  CogVideoX5B  RunwayML
#                   StableDiffusion         Veo          VideoPoet

# VBench2 + Internet Archive fallback (weaker semantic matching)
bash scripts/steps/01c_download_matched.sh --source vbench
bash scripts/steps/01c_download_matched.sh --source vbench \
    --category_filter Camera_Motion --model_filter CogVideo
```

Output is written to `data/matched_pairs/deepaction_<model>/` (or `matched_pairs/<model>_<cat>/` for `--source vbench`) with a `pairs.json` cross-reference that includes `action_class`, HF paths, and match type.

Visualise intermediate results at any point (no second dataset required):

```bash
bash scripts/steps/viz.sh --real   # after step 02, real only
bash scripts/steps/viz.sh --gen    # after step 02, generated only
vn-viz results/metrics/real.json   # same, directly with label
```

---

## Scripts Reference

### Step scripts — `scripts/steps/`

All step scripts read defaults from `config.yaml` and accept CLI overrides for
any config param. Use `--help` to see all available flags.

| Step script | Bash wrapper | Purpose |
|-------------|--------------|---------|
| `download_data.py` | `00_download_data.sh` | Download real reference videos |
| `generate_videos.py` | `01_generate_videos.sh` | Generate videos (model + custom noise) [GPU] |
| `compute_metrics.py` | `02_compute_metrics.sh` | Correlation, spectral, quality, perceptual metrics → JSON |
| `noise_inversion.py` | `03_noise_inversion.sh` | Noise inversion + statistical characterisation + covariance → JSON |
| `spatiotemporal.py` | `04_spatiotemporal.sh` | PCA, UMAP, 3D FFT → plots |
| `noise_ablation.py` | `05_noise_init_ablation.sh` | Sweep all 5 noise types |
| `compare_results.py` | `06_compare_results.sh` | Summary table + comparison figure |
| `distribution_metrics.py` | `07_distribution_metrics.sh` | FVD, cross-set LPIPS, CLIP score → JSON + figure |
| `attention_analysis.py` | `08_attention_analysis.sh` | Attention map extraction + comparison [GPU] |
| `paired_analysis.py` | `09_paired_analysis.sh` | Paired Δ analysis + Wilcoxon tests → figure |
| — | `viz.sh` | Visualise a single run's metrics (`--mode real\|gen\|both`) |

### Low-level CLI entry points — `vn-*`

These are called by the step scripts internally; you can also use them directly.

| Console command | Module | Purpose |
|-----------------|--------|---------|
| `vn-generate` | `videonoise.scripts.generate_videos` | Generate videos |
| `vn-metrics` | `videonoise.scripts.compute_metrics` | Compute metrics |
| `vn-inversion` | `videonoise.scripts.noise_inversion` | Noise inversion |
| `vn-stanalysis` | `videonoise.scripts.spatiotemporal_analysis` | ST analysis |
| `vn-compare` | `videonoise.scripts.compare_results` | Comparison figure |
| `vn-viz` | `videonoise.scripts.viz_metrics` | Single-run visualisation |
| `vn-noise-init` | `videonoise.noise.generators` | Noise shape comparison |
| `vn-dist-metrics` | `videonoise.scripts.distribution_metrics` | FVD + LPIPS + CLIP |
| `vn-attention` | `videonoise.scripts.attention_analysis` | Attention analysis [GPU] |
| `vn-paired` | `videonoise.scripts.paired_analysis` | Paired video analysis |

### Common CLI flags (all step scripts)

| Flag | Description |
|------|-------------|
| `--config` | Path to config YAML (default: `scripts/config.yaml`) |
| `--max_frames N` | Max frames to load per video |
| `--resize W H` | Resize frames before processing |
| `--model` | `svd` \| `modelscope` \| `cogvideox` \| `wan` |
| `--noise_type` | `gaussian` \| `ar1` \| `spatial_lowpass` \| `blue` \| `perlin` |
| `--n_videos N` | Number of videos |
| `--num_frames N` | Frames per video |
| `--data_real` | Real videos directory |
| `--gen_dir` | Override: exact generated videos directory |

---

## Metrics Reference

> Full details — interpretation, research signals, proposed additional analyses, and connection to a learnable prior — are in [METRICS.md](METRICS.md).

★ = primary metric  ·  all metrics compare real vs. generated; the gap is the signal, not an absolute value.

### Step 02 outputs — `results/<run_key>/metrics.json`

| Metric ★ | JSON key | Plot |
|----------|----------|------|
| **Temporal SSIM** ★ | `temporal_ssim.temporal_ssim_mean` | `01_temporal_coherence.png` |
| Frame Pearson r | `frame_correlation.pearson_mean` | `01_temporal_coherence.png` |
| Frame Spearman ρ | `frame_correlation.spearman_mean` | `01_temporal_coherence.png` |
| PSNR (frame-to-frame) | `psnr.psnr_mean` | `01_temporal_coherence.png` |
| Temporal ACF (lags 1–10) | `temporal_acf.lag_k` | `02_motion_dynamics.png` |
| Flow magnitude (mean, std, p95) | `optical_flow.flow_mag_*` | `02_motion_dynamics.png` |
| **Flow direction entropy** ★ | `flow_direction_entropy.flow_direction_entropy_mean` | `06_motion_structure.png` |
| **LPIPS temporal** ★ | `lpips_temporal.lpips_mean` | `06_motion_structure.png` |
| Multiscale temporal SSIM | `multiscale_ssim.scale_{0..3}.ssim_mean` | `06_motion_structure.png` |
| **Temporal MI** ★ | `temporal_mi.lag_{1..5}` | `06_motion_structure.png` |
| **Spatial spectral slope** ★ | `spatial_power_spectrum.spectral_slope` | `03_spectral.png` |
| Spectral R² | `spatial_power_spectrum.spectral_r2` | `03_spectral.png` |
| 3D low-freq energy ratio | `spatiotemporal_3d.low_freq_energy_ratio` | `03_spectral.png` |
| Frame pixel statistics | `frame_statistics.*` | `04_frame_content.png` |

### Step 03 outputs — `results/<run_key>/noise_stats.json`

| Metric ★ | JSON key | Plot |
|----------|----------|------|
| KL from N(0,1) | `statistics.kl_from_gaussian` | `05_noise_stats.png` |
| KS normality p-value | `statistics.ks_p_value` | `05_noise_stats.png` |
| Noise moments (mean/std/skew/kurt) | `statistics.*` | `05_noise_stats.png` |
| **Cross-frame noise correlation** ★ | `cross_frame_correlation.cross_frame_corr_mean` | `05_noise_stats.png` |
| Noise spectral slope | `power_spectrum.spectral_slope_1d` | `05_noise_stats.png` |
| **Off-diagonal energy ratio** ★ | `covariance_structure.off_diagonal_energy_ratio` | `07_noise_covariance.png` |
| Top eigenvalue ratio | `covariance_structure.top_eigenvalue_ratio` | `07_noise_covariance.png` |
| Eigenvalue explained variance | `covariance_structure.eigenvalue_explained_variance` | `07_noise_covariance.png` |

### Step 04 outputs — `results/<run_key>/spatiotemporal/`

| File | What it shows |
|------|---------------|
| `pca_spatial_mode_{0,1,2}.png` | Dominant spatial patterns varying over time |
| `pca_temporal_0.png` | Amplitude of mode 0 across frames |
| `*_3d_spectrum.png` | Log-power vs temporal / spatial frequency axes |

### Step 07 outputs — `results/<gen_key>/distribution_metrics.json`

| Metric | JSON key | Plot |
|--------|----------|------|
| FVD | `fvd` | `08_distribution_metrics.png` |
| LPIPS temporal (real) | `lpips_temporal_real` | `08_distribution_metrics.png` |
| LPIPS temporal (gen) | `lpips_temporal_gen` | `08_distribution_metrics.png` |
| LPIPS real vs gen | `lpips_real_vs_gen_mean` | `08_distribution_metrics.png` |
| CLIP score | `clip_score_mean` | `08_distribution_metrics.png` |

### Step 08 outputs — `results/<run_key>/attention/`

| File | What it shows |
|------|---------------|
| `<video>/attn_self_overview.png` | Self-attention activation grid per layer |
| `<video>/attn_cross_overview.png` | Cross-attention activations (text-conditioned models) |
| `attention_stats.json` | Entropy, mean, max/mean ratio per layer |
| `inversion_comparison.png` | Real vs generated inversion attention entropy (gen folder) |

### Step 09 outputs — `results/<gen_key>/`

| File | What it shows |
|------|---------------|
| `paired_stats.json` | Wilcoxon p-values + median Δ per metric |
| `paired_analysis/01_metric_deltas.png` | Violin plots of gen − real Δ with p-value annotations |

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

### Naming convention

Results folders are named automatically from the **data path** and **processing settings**
so runs with different datasets or resolutions never collide:

```
results/<dataset_path>__<frames>f_<W>x<H>/      ← real videos
results/<gen_folder>__<frames>f_<W>x<H>/        ← generated videos
```

Examples with the current config (`max_frames: 32`, `resize: [512, 768]`):

```
results/deepaction_all__real__32f_512x768/
results/generated_CogVideoX5B__32f_512x768/
```

Changing any of `data_real`, `max_frames`, or `resize` automatically creates a
new folder — stale results from a previous run are never silently reused.

For the noise ablation (step 05), each noise type gets its own folder:

```
results/hf_gaussian__32f_512x768/
results/hf_ar1__32f_512x768/
results/hf_perlin__32f_512x768/
…
```

---

### Full directory tree after running all steps

Example with `data/matched_pairs/deepaction_all/`, `max_frames: 32`, `resize: [512, 768]`:

```
data/
├── matched_pairs/
│   └── deepaction_all/
│       ├── real/                    # real Pexels videos
│       └── generated_CogVideoX5B/   # matched AI-generated videos
└── generated/                       # step 01 outputs (when generating locally)
    ├── hf_gaussian/
    │   ├── video_000_seed42.mp4
    │   └── metadata.json            # model, noise, prompt, all generation params
    ├── hf_ar1/
    └── ...

results/
├── deepaction_all__real__32f_512x768/          # real video analysis
│   ├── metrics.json                            # step 02
│   ├── noise_stats.json                        # step 03 (optional)
│   ├── plots/                                  # viz step
│   │   ├── 01_temporal_coherence.png
│   │   ├── 02_motion_dynamics.png
│   │   ├── 03_spectral.png
│   │   ├── 04_frame_content.png
│   │   └── 05_noise_stats.png                  # only after step 03
│   └── spatiotemporal/                         # step 04
│       ├── pca_spatial_mode_0.png
│       ├── pca_temporal_0.png
│       └── 3d_spectrum_*.png
│
├── generated_CogVideoX5B__32f_512x768/         # generated video analysis
│   ├── metrics.json
│   ├── noise_stats.json
│   ├── plots/
│   ├── spatiotemporal/
│   └── comparison/                             # step 06
│       └── comparison_overview.png
│
└── noise_init/                                 # step 05 — noise shape comparison
    └── noise_comparison.png
```

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

