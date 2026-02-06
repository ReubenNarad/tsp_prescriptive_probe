#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/TSP100_uniform_expdecay_12-24_00:09:50}"

NUM_INSTANCES=3000
NUM_SHARDS=20
SEED=0
TAG="n3000_s20"

CONCORDE_TIMEOUT_SEC=60
LOG_EVERY_SEC=60
BATCH_SIZE_EXTRACT=16

PROBE_BATCH_SIZE=128
PROBE_LR=3e-3
PROBE_WEIGHT_DECAY=1e-4
PROBE_NUM_EPOCHS=200

PYTHON="${PYTHON:-python}"

RUN_NAME="$(basename "${RUN_DIR}")"
DATASET_ID="${RUN_NAME}__n${NUM_INSTANCES}__seed${SEED}__${TAG}"
RAW_DIR="${REPO_ROOT}/tasks/node_removal/data/raw/${DATASET_ID}"
PROCESSED_DIR="${REPO_ROOT}/tasks/node_removal/data/processed/${DATASET_ID}"
TMP_ROOT="${REPO_ROOT}/tasks/node_removal/tmp/concorde"

echo "[what-if] run_dir:       ${RUN_DIR}"
echo "[what-if] dataset_id:    ${DATASET_ID}"
echo "[what-if] raw_dir:       ${RAW_DIR}"
echo "[what-if] processed_dir: ${PROCESSED_DIR}"
echo "[what-if] shards:        ${NUM_SHARDS}"

command -v concorde >/dev/null 2>&1 || { echo "ERROR: concorde not found on PATH"; exit 1; }
[[ -f "${RUN_DIR}/env.pkl" ]] || { echo "ERROR: missing ${RUN_DIR}/env.pkl"; exit 1; }

mkdir -p "${RAW_DIR}" "${PROCESSED_DIR}" "${TMP_ROOT}"

if [[ ! -f "${PROCESSED_DIR}/dataset.pt" ]]; then
  pids=()
  for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_file="$(printf "shard_%04d.pt" "${shard_idx}")"
    out_path="${RAW_DIR}/${shard_file}"
    if [[ -f "${out_path}" ]]; then
      echo "[what-if] shard exists, skipping: ${out_path}"
      continue
    fi
    echo "[what-if] launch shard ${shard_idx}/${NUM_SHARDS} -> ${out_path}"
    "${PYTHON}" "${REPO_ROOT}/tasks/node_removal/collect/collect_dataset.py" \
      --run_dir "${RUN_DIR}" \
      --num_instances "${NUM_INSTANCES}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_idx "${shard_idx}" \
      --out_path "${out_path}" \
      --tmp_root "${TMP_ROOT}" \
      --seed "${SEED}" \
      --concorde_timeout_sec "${CONCORDE_TIMEOUT_SEC}" \
      --log_every_sec "${LOG_EVERY_SEC}" &
    pids+=("$!")
  done

  if [[ "${#pids[@]}" -gt 0 ]]; then
    echo "[what-if] waiting on ${#pids[@]} shard workers..."
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  fi

  echo "[what-if] merge shards -> ${PROCESSED_DIR}"
  "${PYTHON}" "${REPO_ROOT}/tasks/node_removal/collect/merge_shards.py" \
    --raw_dir "${RAW_DIR}" \
    --out_dir "${PROCESSED_DIR}"

  echo "[what-if] summarize -> ${PROCESSED_DIR}"
  "${PYTHON}" "${REPO_ROOT}/tasks/node_removal/collect/summarize_dataset.py" \
    --data_dir "${PROCESSED_DIR}"
else
  echo "[what-if] dataset.pt exists; skipping collection/merge/summary"
fi

echo "[what-if] validate -> ${PROCESSED_DIR}"
"${PYTHON}" "${REPO_ROOT}/tasks/node_removal/collect/validate_dataset.py" --data_dir "${PROCESSED_DIR}"

if [[ ! -f "${PROCESSED_DIR}/probe_reps.pt" ]]; then
  echo "[what-if] extract (encoder_output) -> ${PROCESSED_DIR}/probe_reps.pt"
  "${PYTHON}" "${REPO_ROOT}/core/reps/extract_node_reps.py" \
    --data_dir "${PROCESSED_DIR}" \
    --activation_key encoder_output \
    --batch_size "${BATCH_SIZE_EXTRACT}"
else
  echo "[what-if] probe_reps.pt exists; skipping extract"
fi

echo "[what-if] train CE probe (length, resid only) -> ${PROCESSED_DIR}/probe_artifacts_best_node_ce_std"
"${PYTHON}" "${REPO_ROOT}/core/probe/train_probes.py" \
  --reps_path "${PROCESSED_DIR}/probe_reps.pt" \
  --target length \
  --objective best_node_ce \
  --model linear \
  --standardize_x \
  --batch_size "${PROBE_BATCH_SIZE}" \
  --lr "${PROBE_LR}" \
  --weight_decay "${PROBE_WEIGHT_DECAY}" \
  --num_epochs "${PROBE_NUM_EPOCHS}" \
  --out_dir "${PROCESSED_DIR}/probe_artifacts_best_node_ce_std"

echo "[what-if] done"
