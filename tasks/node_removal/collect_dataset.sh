#!/usr/bin/env bash
set -euo pipefail

if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

usage() {
  cat <<'EOF'
Usage:
  bash tasks/node_removal/collect_dataset.sh <tsp_policy_path|run_name> <num_instances> [num_shards]

Arguments:
  <tsp_policy_path|run_name>  Path to a trained policy run dir (expects env.pkl), e.g. runs/uniform_for_ablation
                             If a bare name is provided, it is resolved as runs/<name>.
  <num_instances>             Number of base instances to sample (each yields n+1 Concorde solves).
  [num_shards]                Optional: number of shards to split collection into (default: 1).

Environment variables (optional):
  OUT_ROOT                 Default: tasks/node_removal/data/raw
  PROCESSED_ROOT           Default: tasks/node_removal/data/processed
  TMP_ROOT                 Default: tasks/node_removal/tmp/concorde
  DATASET_ID               Default: <run_name>__n<num_instances>__seed<seed>[__<tag>]
  TAG                      Optional suffix for DATASET_ID (e.g. "pilot")
  SEED                     Default: 0
  CONCORDE_TIMEOUT_SEC     Default: "" (no timeout)
  LOG_EVERY_SEC           Default: "" (collector default; set to e.g. 15 to print progress)
  MAX_REMOVED_NODES        Default: "" (solve removals for all nodes; set small for smoke tests)
  ASSERT_NUM_LOC           Default: "" (no check)
  OVERWRITE                Default: 0 (set 1 to overwrite shard files)
  DO_MERGE                 Default: 1
  DO_SUMMARY               Default: 1
  PYTHON                   Default: python

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
CONCORDE_TIMEOUT_SEC="${CONCORDE_TIMEOUT_SEC:-}"
LOG_EVERY_SEC="${LOG_EVERY_SEC:-}"
MAX_REMOVED_NODES="${MAX_REMOVED_NODES:-}"
ASSERT_NUM_LOC="${ASSERT_NUM_LOC:-}"
OVERWRITE="${OVERWRITE:-0}"
DO_MERGE="${DO_MERGE:-1}"
DO_SUMMARY="${DO_SUMMARY:-1}"
TAG="${TAG:-}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/tasks/node_removal/data/raw}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${REPO_ROOT}/tasks/node_removal/data/processed}"
TMP_ROOT="${TMP_ROOT:-${REPO_ROOT}/tasks/node_removal/tmp/concorde}"

if [[ -d "${RUN_ARG}" ]]; then
  RUN_DIR="${RUN_ARG}"
elif [[ -d "${REPO_ROOT}/runs/${RUN_ARG}" ]]; then
  RUN_DIR="${REPO_ROOT}/runs/${RUN_ARG}"
else
  echo "ERROR: Could not resolve run dir from '${RUN_ARG}'"
  echo "  Expected a directory or runs/<name>."
  exit 1
fi

if [[ ! -f "${RUN_DIR}/env.pkl" ]]; then
  echo "ERROR: Missing env.pkl at: ${RUN_DIR}/env.pkl"
  echo "  This run dir defines the sampling distribution (env.reset())."
  exit 1
fi

if ! command -v concorde >/dev/null 2>&1; then
  echo "ERROR: 'concorde' not found on PATH."
  exit 1
fi

RUN_NAME="$(basename "${RUN_DIR}")"
DATASET_ID="${DATASET_ID:-${RUN_NAME}__n${NUM_INSTANCES}__seed${SEED}}"
if [[ -n "${TAG}" ]]; then
  DATASET_ID="${DATASET_ID}__${TAG}"
fi

RAW_DIR="${OUT_ROOT}/${DATASET_ID}"
PROCESSED_DIR="${PROCESSED_ROOT}/${DATASET_ID}"

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}" "${TMP_ROOT}"

COLLECT_PY="${REPO_ROOT}/tasks/node_removal/collect/collect_dataset.py"
MERGE_PY="${REPO_ROOT}/tasks/node_removal/collect/merge_shards.py"
SUMMARY_PY="${REPO_ROOT}/tasks/node_removal/collect/summarize_dataset.py"

if [[ ! -f "${COLLECT_PY}" ]]; then
  echo "ERROR: Collector not found (not implemented yet): ${COLLECT_PY}"
  echo "  See: tasks/node_removal/PLAN.md"
  exit 1
fi

echo "[what-if] Run dir:      ${RUN_DIR}"
echo "[what-if] Instances:    ${NUM_INSTANCES}"
echo "[what-if] Shards:       ${NUM_SHARDS}"
echo "[what-if] Dataset id:   ${DATASET_ID}"
echo "[what-if] Raw dir:      ${RAW_DIR}"
echo "[what-if] Processed:    ${PROCESSED_DIR}"
echo "[what-if] Tmp root:     ${TMP_ROOT}"

for ((shard_idx=0; shard_idx<NUM_SHARDS; shard_idx++)); do
  shard_path="${RAW_DIR}/shard_$(printf "%04d" "${shard_idx}").pt"

  cmd=(
    "${PYTHON}" "${COLLECT_PY}"
    --run_dir "${RUN_DIR}"
    --num_instances "${NUM_INSTANCES}"
    --num_shards "${NUM_SHARDS}"
    --shard_idx "${shard_idx}"
    --out_path "${shard_path}"
    --tmp_root "${TMP_ROOT}"
    --seed "${SEED}"
  )

  if [[ -n "${CONCORDE_TIMEOUT_SEC}" ]]; then
    cmd+=(--concorde_timeout_sec "${CONCORDE_TIMEOUT_SEC}")
  fi
  if [[ -n "${LOG_EVERY_SEC}" ]]; then
    cmd+=(--log_every_sec "${LOG_EVERY_SEC}")
  fi
  if [[ -n "${MAX_REMOVED_NODES}" ]]; then
    cmd+=(--max_removed_nodes "${MAX_REMOVED_NODES}")
  fi
  if [[ -n "${ASSERT_NUM_LOC}" ]]; then
    cmd+=(--assert_num_loc "${ASSERT_NUM_LOC}")
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  echo "[what-if] Collect shard ${shard_idx}/${NUM_SHARDS} -> ${shard_path}"
  "${cmd[@]}"
done

if [[ "${DO_MERGE}" == "1" ]]; then
  if [[ ! -f "${MERGE_PY}" ]]; then
    echo "[what-if] Skip merge (missing): ${MERGE_PY}"
  else
    echo "[what-if] Merge shards -> ${PROCESSED_DIR}"
    "${PYTHON}" "${MERGE_PY}" --raw_dir "${RAW_DIR}" --out_dir "${PROCESSED_DIR}"
  fi
fi

if [[ "${DO_SUMMARY}" == "1" ]]; then
  if [[ ! -f "${SUMMARY_PY}" ]]; then
    echo "[what-if] Skip summary (missing): ${SUMMARY_PY}"
  else
    echo "[what-if] Summarize -> ${PROCESSED_DIR}"
    "${PYTHON}" "${SUMMARY_PY}" --data_dir "${PROCESSED_DIR}"
  fi
fi

echo "[what-if] Done."
