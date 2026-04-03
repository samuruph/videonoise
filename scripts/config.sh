#!/usr/bin/env bash
# ============================================================
# Shared configuration for all experiment steps.
# Edit these variables, then run individual steps or run_all.sh
# ============================================================

# --- Paths ---
DATA_REAL="data/real/"
DATA_GEN="data/generated/"
RESULTS="results/"

# --- Video loading ---
MAX_FRAMES=32
RESIZE="128 128"          # W H passed to --resize

# --- Generation model ---
# Choices: svd | modelscope | cogvideox | wan
MODEL="svd"

# svd         ~14 GB — needs image conditioning (auto-generated placeholder)
# modelscope  ~7 GB  — text-to-video
# cogvideox   ~14 GB — text-to-video, 49 frames at 8 fps
# wan         ~6 GB  — text-to-video, requires diffusers>=0.32

# Override HuggingFace model ID (leave empty to use the per-model default)
MODEL_ID=""

# Text prompt (used by modelscope / cogvideox / wan)
PROMPT="A beautiful natural landscape with flowing water"

# --- Noise initialization ---
# Choices: gaussian | ar1 | spatial_lowpass | blue | perlin
NOISE_TYPE="gaussian"
ALPHA=0.8     # AR(1) temporal coefficient  [ar1 noise]
SIGMA=2.0     # Spatial sigma               [spatial_lowpass noise]

# --- Run settings ---
N_VIDEOS=4    # Videos to generate
NUM_FRAMES=16
NUM_STEPS=25
SEED=42
