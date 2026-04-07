"""
CLI: download or generate video datasets.

Real reference videos
---------------------
    python -m videonoise.scripts.download_data --dataset synthetic --n 8 --output data/real/
    python -m videonoise.scripts.download_data --dataset davis     --output data/real/

Pre-generated AI videos  (skips step 01 — no GPU/model required)
-----------------------------------------------------------------
    # Coverr — free stock videos, no API key needed
    python -m videonoise.scripts.download_data --dataset archive \\
        --n 90 --query "nature" --output data/generated/archive_nature/

    # Pexels — requires a free API key from pexels.com/api
    python -m videonoise.scripts.download_data --dataset pexels \\
        --api_key YOUR_KEY --n 90 --query "people walking" \\
        --output data/generated/pexels_people/

    # HuggingFace — AI-generated video evaluation sets (no key needed)
    python -m videonoise.scripts.download_data --dataset hf_generated \\
        --hf_source vbench_sampled --n 90 \\
        --output data/generated/hf_vbench/

DAVIS has ~90 sequences; use --n 90 (the default) to match it.
"""
import argparse
import json
import re
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
             "-i", str(seq / "%05d.jpg"),
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
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
                    _download_file(chosen["link"], vid_path)
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
# Stock / reference videos — Internet Archive (no API key)
# ─────────────────────────────────────────────────────────────────────────────

# Curated Internet Archive collections that contain short MP4 clips.
# Each entry: identifier prefix or subject tag used for search.
_IA_COLLECTIONS = {
    "nature":    "subject:nature AND mediatype:movies",
    "city":      "subject:city AND mediatype:movies",
    "people":    "subject:people AND mediatype:movies",
    "animals":   "subject:animals AND mediatype:movies",
    "ocean":     "subject:ocean AND mediatype:movies",
    "sports":    "subject:sports AND mediatype:movies",
    "timelapse": "subject:timelapse AND mediatype:movies",
    "dance":     "subject:dance AND mediatype:movies",
}

def download_archive(
    output_dir: str,
    n: int,
    query: str = "nature",
) -> None:
    """
    Download free video clips from the Internet Archive (archive.org).

    No API key required.  Uses the open Archive.org search API to find
    short MP4/MKV clips matching the query, then downloads them directly.

    The Internet Archive hosts millions of public-domain and Creative-Commons
    licensed videos.  Clips vary in length (typically 5–120 s) and resolution.

    Args:
        query : Free-text search or subject keyword, e.g.:
                'nature', 'city street', 'ocean waves', 'wildlife'
                Or one of the preset keys: nature, city, people, animals,
                ocean, sports, timelapse, dance
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve query: preset key or raw search string
    search_q = _IA_COLLECTIONS.get(query.lower(), f"{query} mediatype:movies")

    print(f"Internet Archive download: query='{query}'  n={n}  → {out}/")

    # Step 1 — search for item identifiers
    search_url = (
        "https://archive.org/advancedsearch.php"
        f"?q={urllib.parse.quote(search_q)}"
        f"&fl[]=identifier,title,format"
        f"&rows={min(n * 4, 200)}"   # fetch extra to account for items with no MP4
        "&sort[]=downloads+desc"     # most-downloaded first → higher quality
        "&output=json"
    )
    try:
        req = urllib.request.Request(
            search_url,
            headers={"User-Agent": "videonoise/1.0 (research)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            results = json.loads(resp.read())
    except Exception as exc:
        raise SystemExit(f"[error] Internet Archive search failed: {exc}")

    items = results.get("response", {}).get("docs", [])
    if not items:
        raise SystemExit(
            f"No results found for query '{query}'.\n"
            f"Try a different search term, e.g.: nature, city, ocean, people")

    print(f"  Found {len(items)} items in search, scanning for MP4s…")

    downloaded  = 0
    videos_meta = []

    for item in items:
        if downloaded >= n:
            break

        identifier = item.get("identifier", "")
        title      = item.get("title", identifier)[:50]

        # Step 2 — fetch the file list for this item
        meta_url = f"https://archive.org/metadata/{identifier}/files"
        try:
            with urllib.request.urlopen(meta_url, timeout=15) as resp:
                file_data = json.loads(resp.read())
        except Exception:
            continue

        # Pick the smallest MP4 file in this item (prefer 480p / small size)
        mp4_files = [
            f for f in file_data.get("result", [])
            if f.get("name", "").lower().endswith(".mp4")
            and f.get("source") == "original"    # skip derivatives
        ]
        if not mp4_files:
            # Fall back to any MP4 (including derivatives)
            mp4_files = [
                f for f in file_data.get("result", [])
                if f.get("name", "").lower().endswith(".mp4")
            ]
        if not mp4_files:
            continue

        # Sort by file size ascending; skip files > 100 MB to stay fast
        mp4_files.sort(key=lambda f: int(f.get("size", 0) or 0))
        chosen = next(
            (f for f in mp4_files if int(f.get("size", 0) or 0) < 100 * 1024 * 1024),
            mp4_files[0],
        )

        fname    = chosen["name"]
        dl_url   = f"https://archive.org/download/{identifier}/{urllib.parse.quote(fname)}"
        vid_path = out / f"video_{downloaded:03d}.mp4"

        if not vid_path.exists():
            size_mb = int(chosen.get("size", 0) or 0) / 1e6
            print(f"  [{downloaded+1}/{n}] {title}  ({size_mb:.1f} MB)  → {vid_path.name}")
            try:
                _download_file(dl_url, vid_path)
            except Exception as exc:
                print(f"    [skip] {exc}")
                if vid_path.exists():
                    vid_path.unlink()   # remove partial download
                continue
        else:
            print(f"  [{downloaded+1}/{n}] {vid_path.name} already exists")

        videos_meta.append({
            "file":       vid_path.name,
            "index":      downloaded,
            "identifier": identifier,
            "title":      title,
            "source_url": dl_url,
            "query":      query,
        })
        downloaded += 1

    _write_metadata(out, {
        "source":       "archive",
        "query":        query,
        "n_requested":  n,
        "n_downloaded": downloaded,
        "timestamp":    _now(),
        "videos":       videos_meta,
    })
    print(f"\n  Done. {downloaded} video(s) in {out}/")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-generated AI videos — Hugging Face Hub
# ─────────────────────────────────────────────────────────────────────────────

# Verified public HuggingFace repos containing MP4 video files.
# Format: key → (repo_id, repo_type, glob_pattern, description)
# All entries here are verified publicly downloadable without login.
# Format: key → (repo_id, repo_type, glob_pattern, description, n_videos_approx)
#
# Gated repos (list_repo_files works but hf_hub_download returns 403):
#   Vchitect/VBench_sampled_video    — visit the repo page to request access
#   Vchitect/VBench_full_info        — same
_HF_SOURCES = {
    # 30k MP4s from VBench 2.0 (CogVideo outputs, camera-motion prompts)
    "vbench2": (
        "Vchitect/VBench-2.0_sampled_videos",
        "dataset",
        "**/*.mp4",
        "VBench 2.0 — 30k CogVideo outputs with camera-motion prompts  [RECOMMENDED]",
    ),
    # 13 MP4s: CogVideoX-2b and CogVideoX-5b
    "cogvideox": (
        "jdelavande/text2video-energy-benchmark-generated-videos",
        "dataset",
        "**/*.mp4",
        "T2V energy benchmark — CogVideoX-2b / 5b outputs (78 videos)",
    ),
    # 121 MP4s: Wan T2V baseline
    "wan": (
        "YunjinZhang/generated_videos",
        "dataset",
        "**/*.mp4",
        "Wan T2V baseline outputs (121 videos)",
    ),
}
_HF_DEFAULT   = "vbench2"
_VBENCH2_REPO = "Vchitect/VBench-2.0_sampled_videos"


def download_hf_generated(
    output_dir: str,
    n: int,
    source_key: str = _HF_DEFAULT,
    repo_id: str = None,
    glob_pattern: str = None,
) -> None:
    """
    Download AI-generated video files from a HuggingFace Hub dataset.

    No API key required for public repos.  Requires huggingface_hub
    (installed automatically with transformers / diffusers).

    Built-in sources (--hf_source):
        vbench_sampled    VBench sampled evaluation videos  [default]
        vbench2_sampled   VBench 2.0 sampled videos
        t2v_bench         T2V energy benchmark outputs
        moviegen          Meta MovieGen benchmark videos

    Or pass --hf_repo / --hf_glob to target any public HF dataset.
    """
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub\n"
            "(already included with: pip install transformers)")

    import re
    import shutil

    def _matches(filepath: str, pattern: str) -> bool:
        """
        Match a repo file path against a glob pattern.
        Handles ** (any depth including zero) and * (within one segment).
        Works on Python 3.10.
        """
        filepath = filepath.strip("/")
        pattern  = pattern.strip("/")
        # Walk char by char to build a regex
        regex = ""
        i = 0
        while i < len(pattern):
            if pattern[i:i+2] == "**":
                regex += "(?:.*/)??"  # zero or more path segments, non-greedy
                i += 2
                # skip the slash after ** if present
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
            elif pattern[i] == "*":
                regex += "[^/]*"
                i += 1
            elif pattern[i] == "?":
                regex += "[^/]"
                i += 1
            else:
                regex += re.escape(pattern[i])
                i += 1
        return bool(re.fullmatch(regex, filepath))

    # Resolve source
    if repo_id is None:
        if source_key not in _HF_SOURCES:
            raise ValueError(
                f"Unknown --hf_source '{source_key}'. "
                f"Choices: {list(_HF_SOURCES)}  or use --hf_repo + --hf_glob")
        repo_id, repo_type, default_glob, description = _HF_SOURCES[source_key]
        glob_pattern = glob_pattern or default_glob
    else:
        repo_type    = "dataset"
        glob_pattern = glob_pattern or "**/*.mp4"
        description  = f"Custom: {repo_id}"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"HuggingFace download")
    print(f"  repo        : {repo_id}")
    print(f"  description : {description}")
    print(f"  pattern     : {glob_pattern}")
    print(f"  n           : {n}")
    print(f"  output      : {out}/")

    try:
        all_files = list(list_repo_files(repo_id, repo_type=repo_type))
    except Exception as exc:
        raise SystemExit(
            f"[error] Could not list files in '{repo_id}': {exc}\n"
            f"  • Check the repo ID is correct\n"
            f"  • If private/gated: run `huggingface-cli login` first")

    mp4_files = [f for f in all_files if _matches(f, glob_pattern)]
    if not mp4_files:
        raise SystemExit(
            f"No files matched '{glob_pattern}' in {repo_id}.\n"
            f"Try --hf_source with a different key, or --hf_repo + --hf_glob.")

    if source_key == "vbench2":
        mp4_files = _sample_vbench2_balanced(mp4_files, n)
        print(f"  Found {len(mp4_files)} files after balanced category sampling, "
              f"downloading {len(mp4_files)}…")
    else:
        mp4_files = mp4_files[:n]
        print(f"  Found {len(mp4_files)} matching files, "
              f"downloading {len(mp4_files)}…")

    videos_meta = []
    for i, hf_path in enumerate(mp4_files):
        vid_path = out / f"video_{i:03d}.mp4"
        if not vid_path.exists():
            print(f"  [{i+1}/{len(mp4_files)}] {Path(hf_path).name}")
            cached = hf_hub_download(
                repo_id=repo_id, filename=hf_path, repo_type=repo_type)
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
# Prompt-matched pairs  — VBench2 generated + Internet Archive real
# ─────────────────────────────────────────────────────────────────────────────

def _sample_vbench2_balanced(mp4_files: list, n: int) -> list:
    """
    Sample n files from a VBench2 file list balanced across categories.

    VBench2 path: <Model>/<Category>/<Prompt>-<idx>.mp4
    Prefers idx=0 files (one per unique prompt) to avoid prompt repetition.
    Shuffles within each category with a fixed seed for reproducibility.
    """
    import random
    from collections import defaultdict

    by_cat: dict = defaultdict(list)
    for f in mp4_files:
        parts = f.split("/")
        cat = parts[1] if len(parts) >= 3 else "unknown"
        by_cat[cat].append(f)

    cats = sorted(by_cat.keys())
    if not cats:
        return mp4_files[:n]

    per_cat = max(1, -(-n // len(cats)))  # ceiling division

    rng = random.Random(42)
    selected = []
    for cat in cats:
        files = by_cat[cat]
        # Prefer idx=0 — one video per unique prompt, avoids near-duplicate clips
        idx0 = [f for f in files if re.search(r"-0\.mp4$", f)]
        rest = [f for f in files if not re.search(r"-0\.mp4$", f)]
        rng.shuffle(idx0)
        rng.shuffle(rest)
        selected.extend((idx0 + rest)[:per_cat])

    rng.shuffle(selected)
    return selected[:n]


def _extract_vbench2_prompts(all_files: list) -> list:
    """
    Parse VBench2 file paths → list of dicts with keys:
        hf_path, model, category, prompt, idx

    VBench2 filename format: <Model>/<Category>/<Prompt text>-<idx>.mp4
    Example: CogVideo/Camera_Motion/Alhambra, First-person perspective.-0.mp4
    """
    results = []
    for fp in all_files:
        if not fp.lower().endswith(".mp4"):
            continue
        parts = fp.split("/")
        if len(parts) < 3:
            continue
        model    = parts[0]
        category = parts[1]
        filename = parts[-1]           # <Prompt>-<idx>.mp4
        m = re.match(r"^(.*)-(\d+)\.mp4$", filename)
        if not m:
            continue
        results.append({
            "hf_path":  fp,
            "model":    model,
            "category": category,
            "prompt":   m.group(1).strip(),
            "idx":      int(m.group(2)),
        })
    return results


def _ia_query_from_vbench2(prompt: str, category: str) -> str:
    """
    Derive a short Internet Archive search query from a VBench2 prompt.

    Strategy:
      1. Take the first clause (before the first comma or period)
      2. Strip stop words and short tokens
      3. Keep up to 4 meaningful words
      4. Fall back to humanised category name if nothing useful
    """
    stop = {
        "a", "an", "the", "is", "are", "in", "on", "of", "and", "to",
        "with", "from", "at", "by", "for", "as", "its", "be", "has",
        "this", "that", "then", "into", "over", "after", "before",
        "towards", "through", "between", "around",
    }
    clean     = prompt.strip(". -")
    first     = re.split(r"[,.]", clean)[0].strip()
    words     = [
        w for w in re.sub(r"[^a-zA-Z0-9 ]", " ", first).lower().split()
        if w not in stop and len(w) > 2
    ]
    query = " ".join(words[:4])
    if not query or len(query) < 4:
        query = category.replace("_", " ").lower()
    return query


def _download_one_archive(dest: Path, query: str) -> None:
    """
    Search Internet Archive for one video matching *query* and download it.
    Raises RuntimeError if nothing is found or downloadable.
    """
    search_url = (
        "https://archive.org/advancedsearch.php"
        f"?q={urllib.parse.quote(query + ' mediatype:movies')}"
        "&fl[]=identifier,title"
        "&rows=30"
        "&sort[]=downloads+desc"
        "&output=json"
    )
    req = urllib.request.Request(
        search_url, headers={"User-Agent": "videonoise/1.0 (research)"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        items = json.loads(resp.read()).get("response", {}).get("docs", [])

    if not items:
        raise RuntimeError(f"No IA results for '{query}'")

    for item in items:
        identifier = item.get("identifier", "")
        try:
            meta_url = f"https://archive.org/metadata/{identifier}/files"
            with urllib.request.urlopen(meta_url, timeout=15) as resp:
                file_data = json.loads(resp.read())
        except Exception:
            continue

        mp4_files = [
            f for f in file_data.get("result", [])
            if f.get("name", "").lower().endswith(".mp4")
        ]
        if not mp4_files:
            continue

        mp4_files.sort(key=lambda f: int(f.get("size", 0) or 0))
        chosen = next(
            (f for f in mp4_files if int(f.get("size", 0) or 0) < 100 * 1024 * 1024),
            mp4_files[0],
        )
        fname  = chosen["name"]
        dl_url = (
            f"https://archive.org/download/{identifier}/"
            f"{urllib.parse.quote(fname)}"
        )
        _download_file(dl_url, dest)
        return

    raise RuntimeError(f"No downloadable MP4 found for query '{query}'")


def download_matched_pairs(
    output_dir: str,
    n: int = 50,
    model_filter: str = None,
    category_filter: str = None,
) -> None:
    """
    Download N matched pairs of (generated, real) videos that share the same
    semantic prompt.

    Generated source : VBench 2.0  (Vchitect/VBench-2.0_sampled_videos, HF Hub)
    Real source      : Internet Archive  (free, no API key)

    For each selected VBench2 prompt:
      1. Download the -0 variant of the generated video from HF Hub.
      2. Extract keywords from the prompt and search Internet Archive.
      3. Download one matching real video clip.

    Prompts are sampled uniformly across VBench2 categories so that the
    distribution of scene types is balanced.

    Output layout::

        <output_dir>/
          generated/video_000.mp4  …  (VBench2 model outputs)
          real/video_000.mp4       …  (Internet Archive clips)
          pairs.json               ←  cross-reference with prompts + queries

    Args:
        n               : Number of pairs (default 50).
        model_filter    : Restrict generated videos to one VBench2 model,
                          e.g. "CogVideo" (default: any model, prefer CogVideo).
        category_filter : Restrict to one VBench2 category,
                          e.g. "Camera_Motion".
    """
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub is required: pip install huggingface_hub")

    import random
    import shutil
    from collections import defaultdict

    out      = Path(output_dir)
    gen_dir  = out / "generated"
    real_dir = out / "real"
    gen_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. List VBench2 files and parse prompts ──────────────────────────────
    print(f"Listing files in {_VBENCH2_REPO}…")
    try:
        all_files = list(list_repo_files(_VBENCH2_REPO, repo_type="dataset"))
    except Exception as exc:
        raise SystemExit(f"[error] Could not list VBench2 files: {exc}")

    entries = _extract_vbench2_prompts(all_files)
    print(f"  Parsed {len(entries)} MP4 entries")

    # ── 2. Keep only idx=0 (one video per prompt) ────────────────────────────
    seen: dict = {}
    for e in entries:
        if e["idx"] != 0:
            continue
        if model_filter and e["model"] != model_filter:
            continue
        if category_filter and e["category"] != category_filter:
            continue
        key = (e["category"], e["prompt"])
        if key not in seen:
            seen[key] = e

    unique = list(seen.values())
    print(f"  Unique prompts (idx=0, after filters): {len(unique)}")
    if not unique:
        raise SystemExit(
            "No prompts matched the filters.\n"
            "Try without --model_filter / --category_filter.")

    # ── 3. Balance across categories, seed for reproducibility ───────────────
    by_cat: dict = defaultdict(list)
    for e in unique:
        by_cat[e["category"]].append(e)

    cats    = sorted(by_cat.keys())
    per_cat = max(1, n // len(cats))

    random.seed(42)
    selected = []
    for cat in cats:
        items = by_cat[cat][:]
        random.shuffle(items)
        selected.extend(items[:per_cat])

    if len(selected) < n:
        selected_keys = {(x["category"], x["prompt"]) for x in selected}
        pool = [e for e in unique if (e["category"], e["prompt"]) not in selected_keys]
        random.shuffle(pool)
        selected.extend(pool[:n - len(selected)])

    selected = selected[:n]
    print(f"  Selected {len(selected)} prompts across {len(cats)} categories")

    # ── 4. Download pairs ────────────────────────────────────────────────────
    pairs = []
    for i, entry in enumerate(selected):
        print(f"\n  [{i+1}/{len(selected)}] [{entry['category']}]"
              f"  {entry['prompt'][:70]}…")

        gen_path  = gen_dir  / f"video_{i:03d}.mp4"
        real_path = real_dir / f"video_{i:03d}.mp4"

        # --- Generated video (VBench2 from HF Hub) --------------------------
        if not gen_path.exists():
            print(f"    ↓ generated  {Path(entry['hf_path']).name}")
            try:
                cached = hf_hub_download(
                    repo_id=_VBENCH2_REPO,
                    filename=entry["hf_path"],
                    repo_type="dataset",
                )
                shutil.copy2(cached, gen_path)
            except Exception as exc:
                print(f"    [skip] generated download failed: {exc}")
                continue
        else:
            print(f"    ✓ generated  {gen_path.name} already exists")

        # --- Real video (Internet Archive) ----------------------------------
        ia_query = _ia_query_from_vbench2(entry["prompt"], entry["category"])
        if not real_path.exists():
            print(f"    ↓ real  (IA query: '{ia_query}')")
            try:
                _download_one_archive(real_path, ia_query)
            except Exception as exc:
                print(f"    [skip] real download failed: {exc}")
                # Keep generated; mark real as missing
                real_path_str = None
            else:
                real_path_str = real_path.name
        else:
            print(f"    ✓ real  {real_path.name} already exists")
            real_path_str = real_path.name

        pairs.append({
            "idx":       i,
            "prompt":    entry["prompt"],
            "category":  entry["category"],
            "model":     entry["model"],
            "hf_path":   entry["hf_path"],
            "ia_query":  ia_query,
            "generated": gen_path.name,
            "real":      real_path_str,
        })

    # ── 5. Write pairs.json ──────────────────────────────────────────────────
    n_complete = sum(1 for p in pairs if p["real"] is not None)
    pairs_path = out / "pairs.json"
    with open(pairs_path, "w") as f:
        json.dump({
            "source_generated": _VBENCH2_REPO,
            "source_real":      "archive.org",
            "n_requested":      n,
            "n_pairs":          len(pairs),
            "n_complete":       n_complete,
            "model_filter":     model_filter,
            "category_filter":  category_filter,
            "timestamp":        _now(),
            "pairs":            pairs,
        }, f, indent=2)

    print(f"\n  Done. {len(pairs)} pairs ({n_complete} complete) in {out}/")
    print(f"  pairs.json → {pairs_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path) -> None:
    """Download url → dest with a compact progress indicator."""
    _last = [-1]
    def _hook(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            if pct != _last[0] and pct % 10 == 0:
                print(f"\r    {pct:3d}%", end="", flush=True)
                _last[0] = pct
    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print(f"\r    100%")


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
            "  --dataset davis        DAVIS 2017 480p (~2 GB, ~90 sequences)\n"
            "  --dataset synthetic    fast procedural videos (no network)\n\n"
            "Pre-generated / stock videos (no GPU needed — use instead of step 01):\n"
            "  --dataset archive       free stock MP4s from archive.co  [NO KEY]\n"
            "  --dataset pexels       short clips from pexels.com     [free API key]\n"
            "  --dataset hf_generated AI outputs from HuggingFace     [NO KEY]\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["davis", "synthetic", "archive", "pexels", "hf_generated",
                 "matched_pairs"],
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

    # Coverr / Pexels shared options
    parser.add_argument(
        "--query", default="nature",
        help="Search query for archive.org or pexels  (default: 'nature')",
    )

    # Pexels options
    pexels = parser.add_argument_group("Pexels options  (--dataset pexels)")
    pexels.add_argument(
        "--api_key", default=None,
        help="Pexels API key — free at https://www.pexels.com/api/",
    )
    pexels.add_argument("--min_dur", type=int, default=3,  help="Min clip duration (s)")
    pexels.add_argument("--max_dur", type=int, default=12, help="Max clip duration (s)")

    # HuggingFace options
    hf = parser.add_argument_group("HuggingFace options  (--dataset hf_generated)")
    hf.add_argument(
        "--hf_source", default=_HF_DEFAULT,
        choices=list(_HF_SOURCES),
        help=(
            "Pre-configured HF source (all verified public, no login needed):\n" +
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

    # matched_pairs options
    mp = parser.add_argument_group(
        "Matched-pairs options  (--dataset matched_pairs)\n"
        "  Downloads N pairs of (generated, real) videos sharing the same prompt.\n"
        "  Generated: VBench 2.0 (HF Hub, no key). Real: Internet Archive (no key)."
    )
    mp.add_argument(
        "--model_filter", default=None,
        help="Restrict VBench2 source to one model, e.g. 'CogVideo' (default: any)",
    )
    mp.add_argument(
        "--category_filter", default=None,
        help=(
            "Restrict VBench2 source to one category, "
            "e.g. 'Camera_Motion' (default: balanced across all categories)"
        ),
    )

    # Legacy compat
    parser.add_argument("--n_synthetic", type=int, default=None,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Resolve n (--n_synthetic takes priority for backward compat)
    n = args.n_synthetic if args.n_synthetic is not None else args.n

    query = args.query  # shared by archive and pexels

    # ── Real reference datasets ──────────────────────────────────────────────
    if args.dataset == "davis":
        output = args.output or "data/real/"
        download_davis(output)

    elif args.dataset == "synthetic":
        output = args.output or "data/real/"
        generate_synthetic(output, n=n)

    # ── Stock / pre-generated videos (no GPU) ───────────────────────────────
    elif args.dataset == "archive":
        slug   = query.lower().replace(" ", "_")[:20]
        output = args.output or f"data/generated/archive_{slug}/"
        download_archive(output, n=n, query=query)

    elif args.dataset == "pexels":
        if not args.api_key:
            raise SystemExit(
                "[error] --api_key is required for Pexels.\n"
                "  Get a free key at https://www.pexels.com/api/\n"
                "  Then: python -m videonoise.scripts.download_data "
                "--dataset pexels --api_key YOUR_KEY --query 'nature'"
            )
        slug   = query.lower().replace(" ", "_")[:20]
        output = args.output or f"data/generated/pexels_{slug}/"
        download_pexels(
            output, n=n, api_key=args.api_key,
            query=query, min_dur=args.min_dur, max_dur=args.max_dur,
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

    elif args.dataset == "matched_pairs":
        cat_slug   = (args.category_filter or "all").lower().replace(" ", "_")
        model_slug = (args.model_filter    or "mixed").lower()
        output     = args.output or f"data/matched_pairs/{model_slug}_{cat_slug}/"
        download_matched_pairs(
            output, n=n,
            model_filter=args.model_filter,
            category_filter=args.category_filter,
        )


if __name__ == "__main__":
    main()
