#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOM'
Usage:
  bash scripts/train_probes.sh --reps_path <path> [args...]

This is a thin wrapper around core/probe/train_probes.py.

Example:
  bash scripts/train_probes.sh \
    --reps_path tasks/node_removal/data/processed/<dataset_id>/probe_reps.pt \
    --target length --objective regression --model transformer
EOM
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"

"${PYTHON}" "${REPO_ROOT}/core/probe/train_probes.py" "$@"
