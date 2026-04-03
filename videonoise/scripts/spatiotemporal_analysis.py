"""
CLI: spatio-temporal analysis (PCA, UMAP, 3D FFT) on a folder of .mp4 files.

Usage:
    python -m videonoise.scripts.spatiotemporal_analysis \\
        --input data/real/ --output results/plots/real_st/ --pixel_pca
    python -m videonoise.scripts.spatiotemporal_analysis \\
        --input data/real/ --output results/plots/real_st/ --pixel_pca --umap
"""
from videonoise.analysis.spatiotemporal import _cli

if __name__ == "__main__":
    _cli()
