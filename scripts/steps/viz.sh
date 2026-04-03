#!/usr/bin/env bash
# Intermediate visualisation — works after step 02 alone.
#
# Produces 4–5 organised PNG panels per dataset in results/plots/<label>/:
#
#   01_temporal_coherence.png  — TCS, Pearson, Spearman, PSNR
#   02_motion_dynamics.png     — optical-flow stats + temporal ACF
#   03_spectral.png            — power spectra, slope, R², 3D energy ratio
#   04_frame_content.png       — frame-level pixel statistics
#   05_noise_stats.png         — inverted-noise stats (only with --noise_stats)
#
# Usage:
#   bash scripts/steps/viz.sh              # real + generated (if available)
#   bash scripts/steps/viz.sh --real       # real only
#   bash scripts/steps/viz.sh --gen        # generated only
#   bash scripts/steps/viz.sh --both       # both separately (default)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
eval "$(python scripts/yaml_to_env.py scripts/config.yaml)"

MODE="${1:---both}"    # --real | --gen | --both
GEN_KEY="${MODEL}_${NOISE_TYPE}"

# run_viz <run_dir> <label>
#   run_dir : results/real/  or  results/modelscope_gaussian/
#   label   : used for figure titles only
run_viz() {
    local run_dir="$1"
    local label="$2"
    local metrics_json="${run_dir}metrics.json"
    local noise_json="${run_dir}noise_stats.json"
    local plots_dir="${run_dir}plots/"

    if [ ! -f "$metrics_json" ]; then
        echo "  Skipping $label — ${metrics_json} not found (run step 02 first)"
        return
    fi

    echo "  Visualising $label → ${plots_dir}"

    local cmd="python -m videonoise.scripts.viz_metrics \"$metrics_json\" --label \"$label\" --output \"$plots_dir\""
    if [ -f "$noise_json" ]; then
        cmd="$cmd --noise_stats \"$noise_json\""
        echo "    (+ noise stats from $noise_json)"
    fi
    eval "$cmd"
}

echo "================================================================"
echo "  Intermediate visualisation"
echo "  Mode      : $MODE"
echo "================================================================"

case "$MODE" in
    --real)
        run_viz "${RESULTS}real/"       "real"
        ;;
    --gen)
        run_viz "${RESULTS}${GEN_KEY}/" "$GEN_KEY"
        ;;
    --both|*)
        run_viz "${RESULTS}real/"       "real"
        run_viz "${RESULTS}${GEN_KEY}/" "$GEN_KEY"
        ;;
esac

echo ""
echo "  Done. Plots saved inside each results/<run_key>/plots/"
