#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOM'
Usage:
  bash scripts/extract_activations.sh <task> [args...]

Tasks:
  node_removal   -> core/reps/extract_node_reps.py
  edge_forbid    -> core/reps/extract_edge_reps.py

Example:
  bash scripts/extract_activations.sh node_removal \
    --data_dir tasks/node_removal/data/processed/<dataset_id> \
    --run_dir runs/<run_name>

All remaining args are passed through to the underlying Python extractor.
EOM
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

TASK="$1"; shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"

case "${TASK}" in
  node_removal)
    EXTRACT_PY="${REPO_ROOT}/core/reps/extract_node_reps.py"
    ;;
  edge_forbid)
    EXTRACT_PY="${REPO_ROOT}/core/reps/extract_edge_reps.py"
    ;;
  *)
    echo "ERROR: Unknown task '${TASK}' (expected node_removal or edge_forbid)"
    usage
    exit 2
    ;;
 esac

"${PYTHON}" "${EXTRACT_PY}" "$@"
