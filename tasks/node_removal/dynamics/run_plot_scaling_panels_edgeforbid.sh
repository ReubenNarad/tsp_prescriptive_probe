#!/usr/bin/env bash
set -euo pipefail

CONFIG="tasks/node_removal/tmp/scale_dynamics/scale_dynamics_edgeforbid_config.json"
OUT_PATH="tasks/node_removal/tmp/scale_dynamics/train_vs_probe_edgeforbid.png"
PROBE_METRIC="top1_acc"
SMOOTH_TRAIN=11
SMOOTH_PROBE=5
COLORMAP="viridis"
TRAIN_TITLE="TSP policy performance"
PROBE_TITLE="Edge-forbid probe performance ({probe_metric})"
FIG_TITLE="Scaling: training vs edge-forbid probe accuracy"
TRAIN_LOGY=1
PROBE_LOGY=0
TRAIN_SUBOPTIMALITY=1
TRAIN_YLIM_MAX="15"

# Per-run overrides keyed by label (stretch applies to both panels).
OVERRIDES_JSON='{
  "3.10M": {"stretch": 1.0, "downsample_train": 1, "downsample_probe": 1},
  "0.44M": {"stretch": 1.0, "downsample_train": 1, "downsample_probe": 1},
  "0.20M": {"stretch": 6.0, "downsample_train": 1, "downsample_probe": 1}
}'

RESOLVED_CONFIG="tasks/node_removal/tmp/scale_dynamics/scale_dynamics_edgeforbid_config_resolved.json"

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

out_path.write_text(json.dumps(cfg, indent=2))
PY

args=(
  --config "$RESOLVED_CONFIG"
  --out_path "$OUT_PATH"
  --probe_metric "$PROBE_METRIC"
  --smooth_train "$SMOOTH_TRAIN"
  --smooth_probe "$SMOOTH_PROBE"
  --colormap "$COLORMAP"
  --train_title "$TRAIN_TITLE"
  --probe_title "$PROBE_TITLE"
  --title "$FIG_TITLE"
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
if [[ "$TRAIN_SUBOPTIMALITY" -eq 1 ]]; then
  args+=(--train_suboptimality)
fi

python tasks/node_removal/dynamics/plot_scaling_panels.py "${args[@]}"
