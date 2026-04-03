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

resize = c.get("resize", [128, 128])
exports = {
    "DATA_REAL":     c["data_real"],
    "DATA_GEN":      c["data_gen"],
    "RESULTS":       c["results"],
    "MAX_FRAMES":    c["max_frames"],
    "RESIZE":        f"{resize[0]} {resize[1]}",
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
