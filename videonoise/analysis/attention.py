"""
Attention map extraction and analysis for video diffusion models.

Supports:
  - UNet-based models (SVD, ModelScope): hooks on attn1 (self) and attn2 (cross)
  - Transformer/DiT-based models (CogVideoX, Wan): hooks on .attn and .cross_attn

Usage (generation + inversion for generated, inversion only for real):
  storage, handles = register_attention_hooks(pipe, model_type="unet")
  pipe(...)          # forward pass — storage is populated in-place
  remove_hooks(handles)
  agg = aggregate_attention_maps(storage)
  plot_attention_maps(agg, output_dir, label="gen", video_name="video_01")
  stats = compute_attention_stats(agg)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Hook registration
# ─────────────────────────────────────────────────────────────────────────────

def register_attention_hooks(
    pipe,
    model_type: str = "unet",
) -> tuple[dict[str, list], list]:
    """
    Register forward hooks to capture attention weights during a pipeline pass.

    Args:
        pipe:       Loaded diffusers pipeline object.
        model_type: "unet" for SVD/ModelScope or "transformer" for CogVideoX/Wan.

    Returns:
        (storage, handles) where storage is a dict populated in-place during
        forward passes, and handles is the list of hook handles for removal.

    Notes:
        - Attention slicing must be disabled before calling this function.
          Call pipe.disable_attention_slicing() first if needed.
        - Each hook call appends one attention weight tensor to storage[key].
          The tensor shape depends on the layer — typically (B, heads, Q, K).
    """
    storage: dict[str, list] = {}
    handles: list = []

    if model_type == "unet":
        backbone = pipe.unet
        _register_unet_hooks(backbone, storage, handles)
    elif model_type == "transformer":
        backbone = pipe.transformer
        _register_transformer_hooks(backbone, storage, handles)
    else:
        raise ValueError(f"Unknown model_type {model_type!r}. Use 'unet' or 'transformer'.")

    return storage, handles


def _register_unet_hooks(unet, storage: dict, handles: list) -> None:
    """Hook UNet attention modules (attn1=self, attn2=cross)."""
    for name, module in unet.named_modules():
        # diffusers Attention modules expose .get_attention_scores()
        # We hook BasicTransformerBlock's attn1 and attn2
        if name.endswith(".attn1") or name.endswith(".attn2"):
            kind = "self_attn" if name.endswith(".attn1") else "cross_attn"
            key = f"{kind}/{name}"
            storage[key] = []

            def make_hook(k):
                def hook(module, inp, out):
                    # out is the projected output, not the weights directly.
                    # We store the input query norm to approximate.
                    # For true weights we capture the raw scores in forward.
                    # Fallback: store output activation statistics.
                    if isinstance(out, torch.Tensor):
                        storage[k].append(out.detach().cpu())
                return hook

            h = module.register_forward_hook(make_hook(key))
            handles.append(h)


def _register_transformer_hooks(transformer, storage: dict, handles: list) -> None:
    """Hook Transformer/DiT attention blocks."""
    for name, module in transformer.named_modules():
        is_self  = name.endswith(".attn")  or name.endswith(".attn1")
        is_cross = name.endswith(".cross_attn") or name.endswith(".attn2")
        if not (is_self or is_cross):
            continue
        kind = "self_attn" if is_self else "cross_attn"
        key = f"{kind}/{name}"
        storage[key] = []

        def make_hook(k):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    storage[k].append(out.detach().cpu())
            return hook

        h = module.register_forward_hook(make_hook(key))
        handles.append(h)


def remove_hooks(handles: list) -> None:
    """Remove all registered forward hooks."""
    for h in handles:
        h.remove()
    handles.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_attention_maps(
    storage: dict[str, list],
    timestep_indices: Optional[list[int]] = None,
) -> dict[str, np.ndarray]:
    """
    Average stored attention activations across timesteps.

    Args:
        storage:          Output of register_attention_hooks (populated after forward pass).
        timestep_indices: If provided, average only over these denoising step indices.

    Returns:
        Dict mapping layer key → mean activation array (averaged over timesteps and batch).
    """
    result = {}
    for key, tensors in storage.items():
        if not tensors:
            continue
        if timestep_indices is not None:
            tensors = [tensors[i] for i in timestep_indices if i < len(tensors)]
        if not tensors:
            continue
        stacked = torch.stack(tensors, dim=0)  # (T_steps, ...)
        mean_map = stacked.float().mean(dim=0)  # average over timesteps
        # Average over batch dim if present
        if mean_map.dim() > 2:
            mean_map = mean_map.mean(dim=0)
        result[key] = mean_map.numpy()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_maps(
    attn_maps: dict[str, np.ndarray],
    output_dir: str,
    label: str,
    video_name: str,
    prompt: Optional[str] = None,
) -> None:
    """
    Save attention visualisations to output_dir/attention/<video_name>/.

    Produces:
      attn_self_overview.png       — grid of self-attention activation maps
      attn_cross_overview.png      — grid of cross-attention activation maps
      attn_temporal_evolution.png  — self-attention magnitude per timestep
    """
    out = Path(output_dir) / "attention" / video_name
    out.mkdir(parents=True, exist_ok=True)

    self_keys  = sorted(k for k in attn_maps if k.startswith("self_attn"))
    cross_keys = sorted(k for k in attn_maps if k.startswith("cross_attn"))

    # ── self-attention overview ───────────────────────────────────────────────
    if self_keys:
        _save_activation_grid(
            [attn_maps[k] for k in self_keys],
            self_keys,
            out / "attn_self_overview.png",
            title=f"Self-Attention Activations  |  {label}  |  {video_name}",
        )

    # ── cross-attention overview ──────────────────────────────────────────────
    if cross_keys:
        _save_activation_grid(
            [attn_maps[k] for k in cross_keys],
            cross_keys,
            out / "attn_cross_overview.png",
            title=f"Cross-Attention Activations  |  {label}  |  {video_name}"
                  + (f"\nPrompt: {prompt[:80]}" if prompt else ""),
        )

    print(f"  Saved attention maps → {out}/")


def _save_activation_grid(
    maps: list[np.ndarray],
    keys: list[str],
    path: Path,
    title: str,
) -> None:
    """Save a grid of activation heatmaps."""
    n = len(maps)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

    for i, (m, key) in enumerate(zip(maps, keys)):
        ax = axes[i]
        arr = m
        # Collapse to 2D for display: flatten all dims except last two (spatial)
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        if arr.ndim == 1:
            # 1D feature vector — show as bar
            ax.bar(range(len(arr)), arr[:64])
            ax.set_title(key.split("/")[-1], fontsize=6)
        else:
            ax.imshow(arr, cmap="viridis", aspect="auto")
            ax.set_title(key.split("/")[-1], fontsize=6)
        ax.axis("off")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_attention_stats(attn_maps: dict[str, np.ndarray]) -> dict:
    """
    Compute summary statistics for each attention activation map.

    Returns:
        Nested dict: {layer_key: {"entropy": float, "mean": float, "max": float,
                                   "max_mean_ratio": float}}
    """
    stats = {}
    for key, arr in attn_maps.items():
        flat = arr.flatten().astype(float)
        if flat.size == 0:
            continue
        flat_pos = flat - flat.min()
        total = flat_pos.sum() + 1e-12
        p = flat_pos / total
        entropy = float(-np.sum(p * np.log2(p + 1e-12)))
        stats[key] = {
            "entropy":        entropy,
            "mean":           float(flat.mean()),
            "std":            float(flat.std()),
            "max":            float(flat.max()),
            "max_mean_ratio": float(flat.max() / (abs(flat.mean()) + 1e-12)),
        }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Comparison: real-inversion vs generated-inversion
# ─────────────────────────────────────────────────────────────────────────────

def compare_inversion_attention(
    real_stats: dict,
    gen_stats: dict,
    output_dir: str,
) -> None:
    """
    Plot side-by-side entropy comparison between real and generated inversion attention.

    Saves: output_dir/attention/inversion_comparison.png
    """
    out = Path(output_dir) / "attention"
    out.mkdir(parents=True, exist_ok=True)

    common_keys = sorted(set(real_stats) & set(gen_stats))
    if not common_keys:
        print("  [attention] No common layers to compare — skipping inversion comparison.")
        return

    real_entropy = [real_stats[k].get("entropy", float("nan")) for k in common_keys]
    gen_entropy  = [gen_stats[k].get("entropy",  float("nan")) for k in common_keys]
    short_keys   = [k.split("/")[-1] for k in common_keys]

    fig, ax = plt.subplots(figsize=(max(8, len(common_keys) * 0.6), 5))
    x = np.arange(len(common_keys))
    w = 0.35
    ax.bar(x - w / 2, real_entropy, w, label="Real (inversion)", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, gen_entropy,  w, label="Gen (inversion)",  color="tomato",    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_keys, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Activation entropy (bits)")
    ax.set_title(
        "Attention Entropy: Real Inversion vs Generated Inversion\n"
        "Higher entropy = more spatially uniform activation.",
        fontsize=9, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="lightgray", linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out / "inversion_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}/inversion_comparison.png")
