"""
Shared low-level utilities: device selection, JSON I/O, dict traversal.
"""
import json
from pathlib import Path
from typing import Any

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def extract_scalar(results: dict, *key_path: str) -> list[float]:
    """Walk each per-video entry and collect the scalar at *key_path."""
    values = []
    for vid_data in results.get("per_video", {}).values():
        node = vid_data
        for k in key_path:
            node = node.get(k, {}) if isinstance(node, dict) else {}
        if isinstance(node, (int, float)):
            values.append(float(node))
    return values
