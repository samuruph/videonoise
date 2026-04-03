"""
CLI: download or generate video datasets.

Real reference videos
---------------------
    python -m videonoise.scripts.download_data --dataset synthetic --n 8 --output data/real/
    python -m videonoise.scripts.download_data --dataset davis     --output data/real/

Pre-generated AI videos  (skips step 01 — no GPU/model required)
-----------------------------------------------------------------
    # Pexels stock videos — requires a free API key from pexels.com/api
    python -m videonoise.scripts.download_data --dataset pexels \\
        --api_key YOUR_KEY --n 90 --query "people walking" \\
        --output data/generated/pexels_natural/

    # Hugging Face generated-video dataset (no key needed)
    python -m videonoise.scripts.download_data --dataset hf_generated \\
        --n 90 --output data/generated/hf_modelscope_gaussian/

DAVIS has ~90 sequences; use --n 90 (the default) to match it.
"""
import argparse
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Real reference datasets
# ─────────────────────────────────────────────────────────────────────────────

def download_davis(output_dir: str) -> None:
    url       = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
    zip_path  = Path(output_dir) / "DAVIS-2017-trainval-480p.zip"
    davis_dir = Path(output_dir) / "DAVIS"

    if not davis_dir.exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print("Downloading DAVIS 2017 (~2 GB)…")
        urllib.request.urlretrieve(
            url, zip_path,
            reporthook=lambda b, bs, ts: print(
                f"\r  {b*bs/1e6:.1f}/{ts/1e6:.1f} MB", end="", flush=True),
        )
        print()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        zip_path.unlink()

    seq_dir   = davis_dir / "JPEGImages" / "480p"
    video_dir = davis_dir / "Videos" / "480p"
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting sequences → {video_dir}")
    for seq in sorted(seq_dir.iterdir()):
        if not seq.is_dir():
            continue
        out = video_dir / f"{seq.name}.mp4"
        if out.exists():
            continue
        result = subprocess.run(
            ["ffmpeg", "-y", "-framerate", "24",
             "-pattern_type", "glob", "-i", str(seq / "*.jpg"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out)],
            capture_output=True,
        )
        print(f"  {'OK' if result.returncode == 0 else 'FAIL'}: {out.name}")


def generate_synthetic(output_dir: str, n: int = 5) -> None:
    import cv2
    import numpy as np

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Generating {n} synthetic videos in {output_dir}…")
    for i in range(n):
        out = Path(output_dir) / f"synthetic_{i:02d}.mp4"
        if out.exists():
            continue
        T, H, W = 32, 128, 128
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 8, (W, H))
        for t in range(T):
            x, y  = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
            phase = t / T * 2 * np.pi
            frame = np.sin(2 * np.pi * x + phase) * np.cos(2 * np.pi * y + phase * 0.5)
            frame = ((frame + 1) / 2 * 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        writer.release()
        print(f"  Created {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-generated AI videos — Pexels
# ─────────────────────────────────────────────────────────────────────────────

def download_pexels(
    output_dir: str,
    n: int,
    api_key: str,
    query: str = "nature landscape",
    min_dur: int = 3,
    max_dur: int = 12,
) -> None:
    """
    Download n short video clips from the Pexels API.

    Free API key: https://www.pexels.com/api/  (instant, no credit card)
    Rate limits : 200 requests/hour, 20 000/month (more than enough).

    Args:
        query   : Search terms, e.g. "people walking", "ocean waves", "city night"
        min_dur : Minimum clip duration in seconds (default 3)
        max_dur : Maximum clip duration in seconds (default 12)
    """
    import datetime

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Pexels download: query='{query}'  n={n}  → {out}/")

    downloaded  = 0
    page        = 1
    per_page    = min(n, 80)        # Pexels max per page
    videos_meta = []

    while downloaded < n:
        url = (
            f"https://api.pexels.com/videos/search"
            f"?query={urllib.parse.quote(query)}"
            f"&per_page={per_page}&page={page}"
            f"&min_duration={min_dur}&max_duration={max_dur}"
        )
        req = urllib.request.Request(url, headers={"Authorization": api_key})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            print(f"  [error] Pexels HTTP {e.code}: {e.reason}")
            if e.code == 401:
                raise SystemExit(
                    "Invalid API key. Get a free key at https://www.pexels.com/api/")
            break

        videos = data.get("videos", [])
        if not videos:
            print(f"  [warn] No more results (got {downloaded}/{n})")
            break

        for v in videos:
            if downloaded >= n:
                break

            # Pick the file closest to 480p (height ≤ 540)
            files = sorted(v.get("video_files", []),
                           key=lambda f: abs(f.get("height", 0) - 480))
            chosen = next(
                (f for f in files if f.get("height", 0) <= 540 and f.get("height", 0) >= 240),
                files[0] if files else None,
            )
            if not chosen:
                continue

            vid_path = out / f"video_{downloaded:03d}.mp4"
            if not vid_path.exists():
                print(f"  [{downloaded+1}/{n}] {v['id']}  "
                      f"{chosen.get('width')}×{chosen.get('height')}  "
                      f"{v.get('duration', '?')}s  → {vid_path.name}")
                try:
                    urllib.request.urlretrieve(chosen["link"], vid_path)
                except Exception as exc:
                    print(f"    [skip] download failed: {exc}")
                    continue
            else:
                print(f"  [{downloaded+1}/{n}] {vid_path.name} already exists, skipping")

            videos_meta.append({
                "file":     vid_path.name,
                "index":    downloaded,
                "pexels_id": v["id"],
                "url":      v.get("url"),
                "duration": v.get("duration"),
                "width":    chosen.get("width"),
                "height":   chosen.get("height"),
                "query":    query,
            })
            downloaded += 1

        page += 1
        if page > data.get("total_results", 1) // per_page + 2:
            break   # avoid infinite loop if query has few results

    _write_metadata(out, {
        "source":     "pexels",
        "query":      query,
        "min_dur":    min_dur,
        "max_dur":    max_dur,
        "n_requested": n,
        "n_downloaded": downloaded,
        "timestamp":  _now(),
        "videos":     videos_meta,
    })
    print(f"\n  Done. {downloaded} video(s) in {out}/")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-generated AI videos — Hugging Face Hub
# ─────────────────────────────────────────────────────────────────────────────

# Known HuggingFace repos that contain MP4 generated video files.
# Each entry: (repo_id, repo_type, glob_pattern, description)
_HF_SOURCES = {
    "vbench_modelscope": (
        "Vchitect/VBench_full_info",
        "dataset",
        "*/modelscope_t2v/*.mp4",
        "VBench evaluation — ModelScope T2V outputs",
    ),
    "vbench_lavie": (
        "Vchitect/VBench_full_info",
        "dataset",
        "*/LaVie/*.mp4",
        "VBench evaluation — LaVie outputs",
    ),
    "vbench_cogvideo": (
        "Vchitect/VBench_full_info",
        "dataset",
        "*/CogVideo/*.mp4",
        "VBench evaluation — CogVideo outputs",
    ),
    "vbench_videocrafter": (
        "Vchitect/VBench_full_info",
        "dataset",
        "*/VideoCrafter2/*.mp4",
        "VBench evaluation — VideoCrafter2 outputs",
    ),
}
_HF_DEFAULT = "vbench_modelscope"


def download_hf_generated(
    output_dir: str,
    n: int,
    source_key: str = _HF_DEFAULT,
    repo_id: str = None,
    glob_pattern: str = None,
) -> None:
    """
    Download AI-generated video files from a HuggingFace Hub dataset.

    No API key required for public repos.  Requires huggingface_hub:
        pip install huggingface_hub   (already available via transformers)

    Built-in sources (--hf_source):
        vbench_modelscope    ModelScope T2V outputs from VBench evaluation
        vbench_lavie         LaVie outputs
        vbench_cogvideo      CogVideo outputs
        vbench_videocrafter  VideoCrafter2 outputs

    Or specify --hf_repo and --hf_glob directly for any public HF dataset.
    """
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub\n"
            "(or: pip install transformers)")

    # Resolve source
    if repo_id is None:
        if source_key not in _HF_SOURCES:
            raise ValueError(
                f"Unknown --hf_source '{source_key}'. "
                f"Choices: {list(_HF_SOURCES)} or use --hf_repo + --hf_glob")
        repo_id, repo_type, default_glob, description = _HF_SOURCES[source_key]
        glob_pattern = glob_pattern or default_glob
    else:
        repo_type    = "dataset"
        glob_pattern = glob_pattern or "**/*.mp4"
        description  = f"Custom: {repo_id}"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"HuggingFace download: {repo_id}")
    print(f"  source      : {source_key if repo_id is None else repo_id}")
    print(f"  description : {description}")
    print(f"  pattern     : {glob_pattern}")
    print(f"  n           : {n}")
    print(f"  output      : {out}/")

    # List matching files
    import fnmatch
    all_files = list(list_repo_files(repo_id, repo_type=repo_type))
    mp4_files = [f for f in all_files if fnmatch.fnmatch(f, glob_pattern)]

    if not mp4_files:
        raise SystemExit(
            f"No files matched '{glob_pattern}' in {repo_id}.\n"
            f"Try a different --hf_source or adjust --hf_glob.")

    print(f"  Found {len(mp4_files)} matching files, downloading {min(n, len(mp4_files))}…")
    mp4_files = mp4_files[:n]

    videos_meta = []
    for i, hf_path in enumerate(mp4_files):
        vid_path = out / f"video_{i:03d}.mp4"
        if not vid_path.exists():
            print(f"  [{i+1}/{len(mp4_files)}] {Path(hf_path).name}")
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                repo_type=repo_type,
            )
            import shutil
            shutil.copy2(cached, vid_path)
        else:
            print(f"  [{i+1}/{len(mp4_files)}] {vid_path.name} already exists")

        videos_meta.append({
            "file":    vid_path.name,
            "index":   i,
            "hf_path": hf_path,
            "repo_id": repo_id,
        })

    _write_metadata(out, {
        "source":       "hf_generated",
        "hf_source":    source_key,
        "repo_id":      repo_id,
        "glob_pattern": glob_pattern,
        "description":  description,
        "n_requested":  n,
        "n_downloaded": len(videos_meta),
        "timestamp":    _now(),
        "videos":       videos_meta,
    })
    print(f"\n  Done. {len(videos_meta)} video(s) in {out}/")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_metadata(out_dir: Path, meta: dict) -> None:
    path = out_dir / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {path}")


def _now() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    davis_seq_count = 90  # DAVIS 2017 trainval 480p has ~90 sequences

    parser = argparse.ArgumentParser(
        description=(
            "Download or generate video datasets.\n\n"
            "Real reference videos:\n"
            "  --dataset davis       DAVIS 2017 480p (~2 GB, ~90 sequences)\n"
            "  --dataset synthetic   fast procedural videos (no network)\n\n"
            "Pre-generated AI videos (skips step 01):\n"
            "  --dataset pexels       short clips via Pexels API (free key)\n"
            "  --dataset hf_generated generated videos from HuggingFace Hub (no key)\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["davis", "synthetic", "pexels", "hf_generated"],
        default="synthetic",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output directory.\n"
            "Defaults: real → data/real/\n"
            "          generated → data/generated/<source>_<query|source_key>/"
        ),
    )
    parser.add_argument(
        "--n", type=int, default=davis_seq_count,
        help=f"Number of videos (default: {davis_seq_count} to match DAVIS)",
    )

    # Pexels options
    pexels = parser.add_argument_group("Pexels options  (--dataset pexels)")
    pexels.add_argument(
        "--api_key", default=None,
        help="Pexels API key — free at https://www.pexels.com/api/",
    )
    pexels.add_argument(
        "--query", default="nature landscape",
        help="Search query, e.g. 'people walking', 'ocean waves', 'city night'",
    )
    pexels.add_argument("--min_dur", type=int, default=3,  help="Min clip duration (s)")
    pexels.add_argument("--max_dur", type=int, default=12, help="Max clip duration (s)")

    # HuggingFace options
    hf = parser.add_argument_group("HuggingFace options  (--dataset hf_generated)")
    hf.add_argument(
        "--hf_source", default=_HF_DEFAULT,
        choices=list(_HF_SOURCES),
        help=(
            "Pre-configured HF source:\n" +
            "\n".join(f"  {k}: {v[3]}" for k, v in _HF_SOURCES.items())
        ),
    )
    hf.add_argument(
        "--hf_repo", default=None,
        help="Override: any public HuggingFace dataset repo ID",
    )
    hf.add_argument(
        "--hf_glob", default=None,
        help="Glob pattern for MP4 files within the repo (e.g. '**/*.mp4')",
    )

    # Legacy compat
    parser.add_argument("--n_synthetic", type=int, default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Resolve n (--n_synthetic takes priority for backward compat)
    n = args.n_synthetic if args.n_synthetic is not None else args.n

    # ── Real reference datasets ──────────────────────────────────────────────
    if args.dataset == "davis":
        output = args.output or "data/real/"
        download_davis(output)

    elif args.dataset == "synthetic":
        output = args.output or "data/real/"
        generate_synthetic(output, n=n)

    # ── Pre-generated AI videos ──────────────────────────────────────────────
    elif args.dataset == "pexels":
        if not args.api_key:
            raise SystemExit(
                "[error] --api_key is required for Pexels.\n"
                "  Get a free key at https://www.pexels.com/api/\n"
                "  Then: python -m videonoise.scripts.download_data "
                "--dataset pexels --api_key YOUR_KEY"
            )
        slug   = args.query.lower().replace(" ", "_")[:20]
        output = args.output or f"data/generated/pexels_{slug}/"
        download_pexels(
            output, n=n, api_key=args.api_key,
            query=args.query, min_dur=args.min_dur, max_dur=args.max_dur,
        )

    elif args.dataset == "hf_generated":
        slug   = (args.hf_repo or args.hf_source).replace("/", "_").replace("-", "_")
        output = args.output or f"data/generated/{slug}/"
        download_hf_generated(
            output, n=n,
            source_key=args.hf_source,
            repo_id=args.hf_repo,
            glob_pattern=args.hf_glob,
        )


if __name__ == "__main__":
    main()
