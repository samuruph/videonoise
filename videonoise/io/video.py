"""
Video I/O: loading from disk, saving to disk, tensor conversions.

Supported sources
-----------------
- MP4 / AVI / MOV files   → load_video_cv2()
- JPEG / PNG frame dirs   → load_frame_sequence()   (e.g. DAVIS JPEGImages)
- Mixed folder            → load_video_folder()      (auto-detects both)
"""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# Image extensions recognised as frame sequences
_FRAME_EXTS = {".jpg", ".jpeg", ".png"}


def load_video_cv2(
    path: str,
    max_frames: int = 64,
    resize: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Load a video file (MP4, AVI, …) → (T, C, H, W) float tensor in [0, 1].
    Tries OpenCV first, then imageio-ffmpeg as a fallback.
    """
    frames = _read_with_cv2(path, max_frames, resize)
    if not frames:
        frames = _read_with_imageio(path, max_frames, resize)
    if not frames:
        raise ValueError(f"Could not read any frames from {path!r}")
    return _to_tensor(frames)


def load_frame_sequence(
    folder: str,
    max_frames: int = 64,
    resize: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Load a directory of JPEG/PNG frames → (T, C, H, W) float tensor in [0, 1].
    Frames are sorted by filename (works for DAVIS, UCF-101 frame dumps, etc.).
    """
    folder = Path(folder)
    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in _FRAME_EXTS
    )[:max_frames]
    if not paths:
        raise ValueError(f"No image frames found in {folder!r}")
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize is not None:
            img = cv2.resize(img, resize)
        frames.append(img)
    return _to_tensor(frames)


def load_video_folder(
    folder: str,
    ext: str = ".mp4",
    **kwargs,
) -> List[Tuple[str, torch.Tensor]]:
    """
    Load all videos in *folder*. Returns a list of (name, tensor) pairs.

    Handles two layouts automatically:
    - MP4 files directly in *folder*  (ext=".mp4")
    - Subdirectories of JPEG/PNG frames  (e.g. DAVIS JPEGImages/480p/)
    """
    folder = Path(folder)
    results = []

    # 1. Video files
    for p in sorted(folder.glob(f"*{ext}")):
        results.append((p.stem, load_video_cv2(str(p), **kwargs)))

    # 2. Frame-sequence subdirectories (skip if we already found video files)
    if not results:
        for d in sorted(folder.iterdir()):
            if d.is_dir() and any(p.suffix.lower() in _FRAME_EXTS for p in d.iterdir()):
                results.append((d.name, load_frame_sequence(str(d), **kwargs)))

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_tensor(frames: list) -> torch.Tensor:
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T, C, H, W)


def _read_with_cv2(path, max_frames, resize):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
    cap.release()
    return frames


def _read_with_imageio(path, max_frames, resize):
    try:
        import imageio.v2 as iio
    except ImportError:
        try:
            import imageio as iio
        except ImportError:
            return []
    frames = []
    try:
        reader = iio.get_reader(path, format="ffmpeg")
        for i, frame in enumerate(reader):
            if i >= max_frames:
                break
            frame = np.array(frame)[..., :3]
            if resize is not None:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
        reader.close()
    except Exception:
        pass
    return frames


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def frames_to_numpy(video: torch.Tensor) -> np.ndarray:
    """(T, C, H, W) float [0, 1]  →  (T, H, W, C) uint8."""
    return (video.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


def save_video_cv2(frames: torch.Tensor, path: str, fps: int = 8) -> None:
    """
    Save a (T, C, H, W) float tensor [0, 1] to an MP4 file.

    Uses imageio-ffmpeg (H.264) as the primary writer — reliable on macOS/MPS.
    Falls back to OpenCV mp4v only when imageio is unavailable.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr = frames_to_numpy(frames)  # (T, H, W, C) uint8, RGB

    # --- imageio / ffmpeg path (preferred on macOS) ---
    try:
        import imageio
        writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264",
            output_params=["-pix_fmt", "yuv420p"],  # widest player compatibility
        )
        for frame in arr:
            writer.append_data(frame)
        writer.close()
        return
    except Exception:
        pass  # fall through to OpenCV

    # --- OpenCV fallback ---
    T, H, W, C = arr.shape
    # Try H.264 first (avc1 on macOS), then mp4v as last resort
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
        if writer.isOpened():
            for frame in arr:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            return
        writer.release()
    raise RuntimeError(
        f"Could not open a VideoWriter for {path!r}. "
        "Install imageio-ffmpeg: pip install imageio-ffmpeg"
    )


def to_grayscale(video: torch.Tensor) -> torch.Tensor:
    """(T, C, H, W) RGB  →  (T, 1, H, W) luma (ITU-R 601)."""
    r, g, b = video[:, 0], video[:, 1], video[:, 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(1)


def normalize_zero_mean(x: torch.Tensor) -> torch.Tensor:
    """Subtract mean and divide by std (per tensor)."""
    return (x - x.mean()) / (x.std() + 1e-8)
