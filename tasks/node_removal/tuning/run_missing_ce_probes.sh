#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REPS_PATH="${REPS_PATH:-$REPO_ROOT/tasks/node_removal/tmp/tuning/probe_reps_expdecay_layers0-4_plus_output.pt}"
DEVICE="${DEVICE:-cuda}"

echo "[run_missing_ce_probes] repo_root=$REPO_ROOT"
echo "[run_missing_ce_probes] reps_path=$REPS_PATH"
echo "[run_missing_ce_probes] device=$DEVICE"

python "$REPO_ROOT/tasks/node_removal/tuning/run_probe_multiseed.py" \
  --reps_path "$REPS_PATH" \
  --out_dir "$REPO_ROOT/tasks/node_removal/tmp/tuning/st_listwise_ce_ms20_d256L3_h4_ff512_do0p1_lr3e4_wd1e3_T1" \
  --device "$DEVICE" --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 --repeats 1 \
  --train_size 2500 --val_size 200 --test_size 300 \
  --standardize_x --standardize_y \
  --batch_size 64 --num_epochs 50 --lr 3e-4 --weight_decay 1e-3 \
  --model set_transformer --model_dim 256 --layers 3 --heads 4 --ff_dim 512 --dropout 0.1 --head_mlp_layers 2 \
  --loss listwise_ce --temperature 1.0 --select_metric val_top1

python "$REPO_ROOT/tasks/node_removal/tuning/run_probe_multiseed.py" \
  --reps_path "$REPS_PATH" \
  --out_dir "$REPO_ROOT/tasks/node_removal/tmp/tuning/st_listwise_ce_ms5_d256L4_h4_ff512_do0p1_lr3e4_wd1e3_T1" \
  --device "$DEVICE" --seeds 0,1,2,3,4 --repeats 1 \
  --train_size 2500 --val_size 200 --test_size 300 \
  --standardize_x --standardize_y \
  --batch_size 64 --num_epochs 50 --lr 3e-4 --weight_decay 1e-3 \
  --model set_transformer --model_dim 256 --layers 4 --heads 4 --ff_dim 512 --dropout 0.1 --head_mlp_layers 2 \
  --loss listwise_ce --temperature 1.0 --select_metric val_top1

echo "[run_missing_ce_probes] done"
