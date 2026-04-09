#!/usr/bin/env python3
"""
Print bash export statements from scripts/config.yaml.

Only needed for custom shell scripts. The step scripts use
videonoise.config_loader directly via python -m videonoise.scripts.*.

Usage (inside a bash script):
    eval "$(python scripts/yaml_to_env.py)"
    eval "$(python scripts/yaml_to_env.py path/to/config.yaml)"
"""
import argparse
import sys

_p = argparse.ArgumentParser()
_p.add_argument("config_pos", nargs="?", default="scripts/config.yaml")
_pos = _p.parse_args()

sys.argv = ["yaml_to_env.py", "--config", _pos.config_pos]

from videonoise.config_loader import load_config
cfg = load_config()

exports = {
    "DATA_REAL":       cfg.data_real,
    "DATA_GEN":        cfg.data_gen,
    "GEN_DIR":         cfg.gen_dir_resolved,
    "RESULTS":         cfg.results,
    "REAL_KEY":        cfg.real_key,
    "GEN_KEY":         cfg.gen_key,
    "SETTINGS_SUFFIX": cfg.settings_suffix,
    "REAL_OUT":        cfg.real_out,
    "GEN_OUT":         cfg.gen_out,
    "MAX_FRAMES":      cfg.max_frames,
    "RESIZE":          f"{cfg.resize[0]} {cfg.resize[1]}",
    "MODEL":           cfg.model,
    "MODEL_ID":        cfg.model_id or "",
    "NOISE_TYPE":      cfg.noise_type,
    "ALPHA":           cfg.alpha,
    "SIGMA":           cfg.sigma,
    "N_VIDEOS":        cfg.n_videos,
    "NUM_FRAMES":      cfg.num_frames,
    "NUM_STEPS":       cfg.num_steps,
    "SEED":            cfg.seed,
    "ACTIVE_PROMPT":   cfg.active_prompt,
}

for key, val in exports.items():
    safe = str(val).replace("'", "'\\''")
    print(f"export {key}='{safe}'")
