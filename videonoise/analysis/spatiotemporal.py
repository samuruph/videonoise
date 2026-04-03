"""
Spatio-temporal feature analysis: PCA, UMAP, 3D frequency maps.
"""
import argparse
from pathlib import Path
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from videonoise.io.video import to_grayscale
from videonoise.metrics.spectral import frequency_analysis, plot_3d_spectrum


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pixel_pca(video: torch.Tensor, n_components: int = 8) -> dict:
    """
    PCA on the spatio-temporal volume.

    Each spatial position (H×W) is treated as a T×C-dimensional feature vector.

    Args:
        video:        (T, C, H, W) float tensor.
        n_components: Number of PCA components to retain.

    Returns:
        Dict with coords (H*W × n_components), components, explained_variance_ratio,
        and the original H, W, T dimensions.
    """
    T, C, H, W = video.shape
    flat = video.permute(2, 3, 0, 1).reshape(H * W, T * C).numpy()
    flat -= flat.mean(axis=0, keepdims=True)

    pca    = PCA(n_components=n_components)
    coords = pca.fit_transform(flat)

    return {
        "coords":                   coords,
        "components":               pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "H": H, "W": W, "T": T, "C": C,
    }


def plot_pca_maps(pca_result: dict, output_dir: str, name: str) -> None:
    """Save top-N PCA component maps as spatial heatmaps."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    H, W  = pca_result["H"], pca_result["W"]
    n     = min(4, len(pca_result["explained_variance_ratio"]))
    coords = pca_result["coords"]

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(coords[:, i].reshape(H, W), cmap="RdBu_r")
        var = pca_result["explained_variance_ratio"][i] * 100
        ax.set_title(f"PC{i+1} ({var:.1f}%)")
        ax.axis("off")
    plt.suptitle(f"PCA spatial maps — {name}")
    plt.tight_layout()
    out = str(Path(output_dir) / f"{name}_pca_maps.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def plot_pca_temporal(pca_result: dict, output_dir: str, name: str) -> None:
    """Plot how the top PC loadings vary over time (temporal modes)."""
    T, C  = pca_result["T"], pca_result["C"]
    n     = min(4, len(pca_result["explained_variance_ratio"]))
    modes = pca_result["components"][:n].reshape(n, T, C)

    fig, axes = plt.subplots(n, 1, figsize=(10, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(modes[i].mean(axis=1), color=cm.tab10(i))
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_ylabel(f"PC{i+1}")
    axes[-1].set_xlabel("frame")
    plt.suptitle(f"Temporal PCA modes — {name}")
    plt.tight_layout()
    out = str(Path(output_dir) / f"{name}_pca_temporal.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def umap_embedding(
    videos: dict,
    output_dir: str,
    n_components: int = 8,
    max_points: int = 20_000,
) -> None:
    """
    UMAP 2D projection of PCA coordinates from multiple videos.

    Args:
        videos:       {name: pixel_pca result dict}.
        output_dir:   Where to save the plot.
        n_components: How many PCA dimensions to feed into UMAP.
        max_points:   Subsample to this many points if the dataset is larger.
    """
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed (pip install umap-learn). Skipping UMAP.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_coords, labels = [], []
    for name, res in videos.items():
        all_coords.append(res["coords"][:, :n_components])
        labels.extend([name] * len(res["coords"]))

    X = np.concatenate(all_coords, axis=0)
    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X, labels = X[idx], [labels[i] for i in idx]

    print(f"  Running UMAP on {len(X)} points...")
    embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

    unique_labels = list(dict.fromkeys(labels))
    color_map     = {l: cm.tab20(i / len(unique_labels)) for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[color_map[lbl]], label=lbl, s=2, alpha=0.5)
    ax.legend(markerscale=5, fontsize=8)
    ax.set_title("UMAP of spatio-temporal features")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    out = str(Path(output_dir) / "umap_features.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Console-script entry point  (vn-stanalysis)
# ---------------------------------------------------------------------------

def _cli() -> None:
    from videonoise.io import load_video_folder
    from videonoise.utils import save_json

    parser = argparse.ArgumentParser(description="Spatio-temporal analysis")
    parser.add_argument("--input",           required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--max_frames",      type=int, default=32)
    parser.add_argument("--resize",          type=int, nargs=2, default=[128, 128], metavar=("W", "H"))
    parser.add_argument("--n_pca_components", type=int, default=8)
    parser.add_argument("--pixel_pca",       action="store_true")
    parser.add_argument("--umap",            action="store_true")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    videos = load_video_folder(args.input, max_frames=args.max_frames, resize=tuple(args.resize))
    if not videos:
        print(f"No .mp4 files found in {args.input}")
        return

    all_pca, all_freq = {}, {}
    for name, video in videos:
        print(f"Processing {name} {tuple(video.shape)}")
        all_freq[name] = frequency_analysis(video)
        plot_3d_spectrum(video, args.output, name)
        if args.pixel_pca:
            res = pixel_pca(video, n_components=args.n_pca_components)
            plot_pca_maps(res, args.output, name)
            plot_pca_temporal(res, args.output, name)
            all_pca[name] = {k: res[k] for k in ("coords", "explained_variance_ratio")}

    if args.umap and all_pca:
        umap_embedding(all_pca, args.output, n_components=args.n_pca_components)

    save_json({"frequency_analysis": all_freq}, str(Path(args.output) / "frequency_stats.json"))
    print(f"Done → {args.output}")
