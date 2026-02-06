#!/usr/bin/env bash
set -euo pipefail

CONFIG="tasks/node_removal/tmp/scale_dynamics/scale_dynamics_grid_config.json"
OUT_PATH="tasks/node_removal/tmp/scale_dynamics/train_vs_probe_grid.png"
PROBE_METRIC="top1_acc"
TRANSFORMER_METRIC="top1_acc"
SMOOTH_TRAIN=11
SMOOTH_PROBE=5
SMOOTH_TRANSFORMER=5
COLORMAP="viridis_r"
TRAIN_TITLE="TSP policy performance"
PROBE_TITLE_A="Linear probe: node removal"
PROBE_TITLE_B="Linear probe: edge forbid"
TRANSFORMER_TITLE_A="Transformer probe: node removal"
TRANSFORMER_TITLE_B="Transformer probe: edge forbid"
FIG_TITLE="Scaling: training vs probe accuracy"
TRAIN_LOGY=1
PROBE_LOGY=0
TRANSFORMER_LOGY=0
TRAIN_SUBOPTIMALITY=1
TRAIN_YLIM_MAX="18"
XLIM_MAX="590000"

# Per-run overrides keyed by label (stretch applies to all panels).
OVERRIDES_JSON='{
  "0.44M": {"stretch": 1.0, "downsample_train": 1, "downsample_probe": 1, "downsample_transformer": 1},
  "1.10M": {"stretch": 1.0, "downsample_train": 1, "downsample_probe": 1, "downsample_transformer": 1},
  "3.10M": {"stretch": 1.0, "downsample_train": 1, "downsample_probe": 1, "downsample_transformer": 1}
}'

RESOLVED_CONFIG="tasks/node_removal/tmp/scale_dynamics/scale_dynamics_grid_config_resolved.json"

python - "$CONFIG" "$RESOLVED_CONFIG" "$OVERRIDES_JSON" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
overrides = json.loads(sys.argv[3])

cfg = json.loads(config_path.read_text())
runs = cfg.get("runs", [])
for run in runs:
    label = str(run.get("label", ""))
    if not label or label not in overrides:
        continue
    override = overrides[label]
    if "stretch" in override:
        run["step_scale"] = override["stretch"]
    if "downsample_train" in override:
        run["downsample_train"] = override["downsample_train"]
    if "downsample_probe" in override:
        run["downsample_probe"] = override["downsample_probe"]
    if "downsample_transformer" in override:
        run["downsample_transformer"] = override["downsample_transformer"]

out_path.write_text(json.dumps(cfg, indent=2))
PY

args=(
  --config "$RESOLVED_CONFIG"
  --out_path "$OUT_PATH"
  --probe_metric "$PROBE_METRIC"
  --transformer_metric "$TRANSFORMER_METRIC"
  --smooth_train "$SMOOTH_TRAIN"
  --smooth_probe "$SMOOTH_PROBE"
  --smooth_transformer "$SMOOTH_TRANSFORMER"
  --colormap "$COLORMAP"
  --train_title "$TRAIN_TITLE"
  --probe_title_a "$PROBE_TITLE_A"
  --probe_title_b "$PROBE_TITLE_B"
  --transformer_title_a "$TRANSFORMER_TITLE_A"
  --transformer_title_b "$TRANSFORMER_TITLE_B"
  --title "$FIG_TITLE"
  --xlim_max "$XLIM_MAX"
)

if [[ -n "$TRAIN_YLIM_MAX" ]]; then
  args+=(--train_ylim_max "$TRAIN_YLIM_MAX")
fi
if [[ "$TRAIN_LOGY" -eq 1 ]]; then
  args+=(--train_logy)
fi
if [[ "$PROBE_LOGY" -eq 1 ]]; then
  args+=(--probe_logy)
fi
if [[ "$TRANSFORMER_LOGY" -eq 1 ]]; then
  args+=(--transformer_logy)
fi
if [[ "$TRAIN_SUBOPTIMALITY" -eq 1 ]]; then
  args+=(--train_suboptimality)
fi

python tasks/node_removal/dynamics/plot_scaling_panels_grid.py "${args[@]}"
