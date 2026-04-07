#!/usr/bin/env python3
"""
Read scripts/config.yaml and print bash export statements.

Usage (inside a bash script):
    eval "$(python scripts/yaml_to_env.py)"
    eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"   # explicit path
"""
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    # PyYAML not installed — print a helpful error and exit
    print('echo "ERROR: PyYAML not found. Run: pip install pyyaml" >&2; exit 1')
    sys.exit(1)

config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scripts/config.yaml")

with open(config_path) as f:
    c = yaml.safe_load(f)


def _data_key(path_str: str) -> str:
    """
    Derive a short human-readable key from a data directory path.

    Takes the last two meaningful path components (ignoring the leading 'data'
    segment) so that, e.g.:
      data/matched_pairs/deepaction_all/real  →  deepaction_all__real
      data/real/DAVIS/Videos/480p            →  Videos__480p
    """
    parts = [p for p in Path(path_str.rstrip("/")).parts if p not in (".", "")]
    meaningful = [p for p in parts if p != "data"]
    if len(meaningful) >= 2:
        return "__".join(meaningful[-2:])
    return meaningful[0] if meaningful else "data"


resize = c.get("resize", [128, 128])

# Settings fingerprint: embedded in every results folder name so that changing
# max_frames or resize automatically creates a new folder instead of reusing
# stale results.
settings_suffix = f"{c['max_frames']}f_{resize[0]}x{resize[1]}"

# Real-side results key: dataset name + settings
real_key = _data_key(c["data_real"]) + "__" + settings_suffix

exports = {
    "DATA_REAL":        c["data_real"],
    "DATA_GEN":         c["data_gen"],
    "GEN_DIR_OVERRIDE": c.get("gen_dir") or "",
    "RESULTS":          c["results"],
    "REAL_KEY":         real_key,
    "SETTINGS_SUFFIX":  settings_suffix,
    "MAX_FRAMES":       c["max_frames"],
    "RESIZE":           f"{resize[0]} {resize[1]}",
    "MODEL":         c["model"],
    "MODEL_ID":      c.get("model_id") or "",
    "T2I_MODEL_ID":  c.get("t2i_model_id") or "",
    "NOISE_TYPE":    c["noise_type"],
    "ALPHA":         c["alpha"],
    "SIGMA":         c["sigma"],
    "N_VIDEOS":      c["n_videos"],
    "NUM_FRAMES":    c["num_frames"],
    "NUM_STEPS":     c["num_steps"],
    "SEED":          c["seed"],
    "PROMPT":        c["active_prompt"],
    "ACTIVE_PROMPT": c["active_prompt"],
}

for key, val in exports.items():
    # Escape single quotes in the value
    safe = str(val).replace("'", "'\\''")
    print(f"export {key}='{safe}'")
