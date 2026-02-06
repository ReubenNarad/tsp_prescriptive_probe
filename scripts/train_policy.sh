#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"

cd "${REPO_ROOT}"

# Train config
num_epochs=250
num_instances=100_000
num_val=100
num_loc=100
batch_size=1024
temperature=0.8
lr=1e-4
checkpoint_freq=20
dropout=0.1
attention_dropout=0.1
clip_val=0.01

# Learning rate decay configuration
lr_decay="none"  # Options: "none", "cosine", "linear"
# min_lr=1e-6        # Minimum learning rate at end of training

# Load from checkpoint (set to empty string to start from scratch)
load_checkpoint=""
# reset_lr=true

# Model config
embed_dim=256
n_encoder_layers=4

# Run name
run_name="uniform_for_ablation"

# Construct the checkpoint and reset_lr arguments
if [ -n "$load_checkpoint" ]; then
    checkpoint_arg="--load_checkpoint $load_checkpoint"
    if [ "$reset_lr" = true ]; then
        checkpoint_arg="$checkpoint_arg --reset_optimizer"  # Add reset flag
    fi
else
    checkpoint_arg=""
fi

# Train
"${PYTHON}" -m policy.train_vanilla \
    --lr $lr \
    --num_epochs $num_epochs \
    --num_instances $num_instances \
    --num_val $num_val \
    --num_loc $num_loc \
    --run_name $run_name \
    --temperature $temperature \
    --embed_dim $embed_dim \
    --batch_size $batch_size \
    --n_encoder_layers $n_encoder_layers \
    --checkpoint_freq $checkpoint_freq \
    --dropout $dropout \
    --attention_dropout $attention_dropout \
    --clip_val $clip_val \
    --lr_decay $lr_decay \
    # --min_lr $min_lr \
    $checkpoint_arg

"${PYTHON}" tools/solve_td_with_concorde.py --run_name $run_name

"${PYTHON}" -m policy.eval --run_name $run_name --num_epochs $num_epochs
