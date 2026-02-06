#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/collect_labels_node_removal.sh <run_dir|run_name> <num_instances> [num_shards]"
  echo "This is a thin wrapper around tasks/node_removal/collect_dataset.sh"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

bash "${REPO_ROOT}/tasks/node_removal/collect_dataset.sh" "$@"
