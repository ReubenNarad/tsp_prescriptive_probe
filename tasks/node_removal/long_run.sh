#!/usr/bin/env bash
set -euo pipefail

if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

usage() {
  cat <<'EOF'
Usage:
  bash tasks/node_removal/long_run.sh <tsp_policy_path|run_name> <num_instances> [num_shards]

Runs a full what-if pipeline:
  1) Collect Concorde what-if dataset (shards -> merged dataset.pt)
  2) Validate tour-length monotonicity (removing a node never increases optimal length)
  3) Extract per-node policy representations aligned to labels
  4) Train linear probes (residual stream)

Arguments:
  <tsp_policy_path|run_name>  Path to a trained policy run dir (expects env.pkl), e.g. runs/TSP100_uniform_...
                             If a bare name is provided, it is resolved as runs/<name>.
  <num_instances>             Number of base instances to sample (each yields n+1 Concorde solves).
  [num_shards]                Number of shards (default: 1). Note: collect_dataset.sh runs shards sequentially.

Environment variables (optional):
  SEED                   Default: 0
  TAG                    Default: long_v1
  CONCORDE_TIMEOUT_SEC   Default: 60
  OVERWRITE              Default: 0 (set 1 to overwrite shards)

  ACTIVATION_KEY         Default: encoder_output
  ACTIVATION_KEYS        Default: (unset)  Comma-separated list; if set overrides ACTIVATION_KEY
  BATCH_SIZE_EXTRACT     Default: 16
  PROBE_TARGET           Default: length   (length|time|both)
  PROBE_OBJECTIVE        Default: regression (regression|best_node_ce)
  PROBE_MODEL            Default: linear   (linear|mlp)
  PROBE_MLP_HIDDEN_DIM   Default: 256
  PROBE_MLP_LAYERS       Default: 1
  PROBE_MLP_DROPOUT      Default: 0.0
  PROBE_NUM_EPOCHS       Default: 50
  PROBE_BATCH_SIZE       Default: 4096
  PROBE_LR               Default: 1e-2
  PROBE_WEIGHT_DECAY     Default: 0
  PROBE_L1_LAMBDA        Default: 0
  STANDARDIZE_X          Default: 0 (set 1 to standardize X)

  PYTHON                 Default: python

EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_ARG="$1"
NUM_INSTANCES="$2"
NUM_SHARDS="${3:-1}"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
TAG="${TAG:-long_v1}"
CONCORDE_TIMEOUT_SEC="${CONCORDE_TIMEOUT_SEC:-60}"
OVERWRITE="${OVERWRITE:-0}"

ACTIVATION_KEY="${ACTIVATION_KEY:-encoder_output}"
ACTIVATION_KEYS="${ACTIVATION_KEYS:-}"
BATCH_SIZE_EXTRACT="${BATCH_SIZE_EXTRACT:-16}"
PROBE_TARGET="${PROBE_TARGET:-length}"
PROBE_OBJECTIVE="${PROBE_OBJECTIVE:-regression}"
PROBE_MODEL="${PROBE_MODEL:-linear}"
PROBE_MLP_HIDDEN_DIM="${PROBE_MLP_HIDDEN_DIM:-256}"
PROBE_MLP_LAYERS="${PROBE_MLP_LAYERS:-1}"
PROBE_MLP_DROPOUT="${PROBE_MLP_DROPOUT:-0.0}"
PROBE_NUM_EPOCHS="${PROBE_NUM_EPOCHS:-50}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-4096}"
PROBE_LR="${PROBE_LR:-1e-2}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0}"
PROBE_L1_LAMBDA="${PROBE_L1_LAMBDA:-0}"
STANDARDIZE_X="${STANDARDIZE_X:-0}"

if [[ -d "${RUN_ARG}" ]]; then
  RUN_DIR="${RUN_ARG}"
elif [[ -d "${REPO_ROOT}/runs/${RUN_ARG}" ]]; then
  RUN_DIR="${REPO_ROOT}/runs/${RUN_ARG}"
else
  echo "ERROR: Could not resolve run dir from '${RUN_ARG}'"
  exit 1
fi

RUN_NAME="$(basename "${RUN_DIR}")"
DATASET_ID="${RUN_NAME}__n${NUM_INSTANCES}__seed${SEED}"
if [[ -n "${TAG}" ]]; then
  DATASET_ID="${DATASET_ID}__${TAG}"
fi

PROCESSED_ROOT="${PROCESSED_ROOT:-${REPO_ROOT}/tasks/node_removal/data/processed}"
DATA_DIR="${PROCESSED_ROOT}/${DATASET_ID}"

echo "[long-run] Run dir:      ${RUN_DIR}"
echo "[long-run] Run name:     ${RUN_NAME}"
echo "[long-run] Instances:    ${NUM_INSTANCES}"
echo "[long-run] Shards:       ${NUM_SHARDS}"
echo "[long-run] Dataset id:   ${DATASET_ID}"
echo "[long-run] Data dir:     ${DATA_DIR}"
if [[ -n "${ACTIVATION_KEYS}" ]]; then
  echo "[long-run] Activations:  ${ACTIVATION_KEYS}"
else
  echo "[long-run] Activation:   ${ACTIVATION_KEY}"
fi
echo "[long-run] Probe target: ${PROBE_TARGET}"
echo "[long-run] Probe obj:    ${PROBE_OBJECTIVE}"
echo "[long-run] Probe model:  ${PROBE_MODEL}"

# 1) Collect dataset
COLLECT_SH="${REPO_ROOT}/tasks/node_removal/collect_dataset.sh"

SEED="${SEED}" \
CONCORDE_TIMEOUT_SEC="${CONCORDE_TIMEOUT_SEC}" \
OVERWRITE="${OVERWRITE}" \
DO_MERGE=1 \
DO_SUMMARY=1 \
TAG="${TAG}" \
bash "${COLLECT_SH}" "${RUN_DIR}" "${NUM_INSTANCES}" "${NUM_SHARDS}"

# 2) Validate invariants (length should never increase)
"${PYTHON}" "${REPO_ROOT}/tasks/node_removal/collect/validate_dataset.py" --data_dir "${DATA_DIR}"

# 3) Extract per-node representations aligned to labels
EXTRACT_CMD=(
  "${PYTHON}" "${REPO_ROOT}/core/reps/extract_node_reps.py"
  --data_dir "${DATA_DIR}"
  --batch_size "${BATCH_SIZE_EXTRACT}"
)
if [[ -n "${ACTIVATION_KEYS}" ]]; then
  EXTRACT_CMD+=(--activation_keys "${ACTIVATION_KEYS}")
else
  EXTRACT_CMD+=(--activation_key "${ACTIVATION_KEY}")
fi
"${EXTRACT_CMD[@]}"

# 4) Train probes
TRAIN_CMD=(
  "${PYTHON}" "${REPO_ROOT}/core/probe/train_probes.py"
  --reps_path "${DATA_DIR}/probe_reps.pt"
  --target "${PROBE_TARGET}"
  --objective "${PROBE_OBJECTIVE}"
  --model "${PROBE_MODEL}"
  --mlp_hidden_dim "${PROBE_MLP_HIDDEN_DIM}"
  --mlp_layers "${PROBE_MLP_LAYERS}"
  --mlp_dropout "${PROBE_MLP_DROPOUT}"
  --num_epochs "${PROBE_NUM_EPOCHS}"
  --batch_size "${PROBE_BATCH_SIZE}"
  --lr "${PROBE_LR}"
  --weight_decay "${PROBE_WEIGHT_DECAY}"
  --l1_lambda "${PROBE_L1_LAMBDA}"
)
if [[ "${STANDARDIZE_X}" == "1" ]]; then
  TRAIN_CMD+=(--standardize_x)
fi
"${TRAIN_CMD[@]}"

echo "[long-run] Done."
