#!/usr/bin/env bash
# Batch-generate LeRobotDataset for all tasks under a training data root.
# Does not modify existing generate.sh; simply loops and calls it.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="${TRAIN_ROOT:-${ROOT_DIR}/training_data/pi05_all}"
REPO_PREFIX="${REPO_PREFIX:-}"
FORCE="${FORCE:-0}"           # set FORCE=1 to overwrite existing HF cache dirs
GEN_SCRIPT="${GEN_SCRIPT:-${ROOT_DIR}/generate.sh}"

if [[ ! -d "${TRAIN_ROOT}" ]]; then
  echo "Training root not found: ${TRAIN_ROOT}" >&2
  exit 1
fi

if [[ ! -x "${GEN_SCRIPT}" ]]; then
  echo "generate.sh not found/executable at: ${GEN_SCRIPT}" >&2
  exit 1
fi

HF_HOME="${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/lerobot"

mapfile -t TASKS < <(find -L "${TRAIN_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "No tasks found under ${TRAIN_ROOT}" >&2
  exit 1
fi

SKIPPED_EMPTY=()
SKIPPED_EXISTS=()

for task in "${TASKS[@]}"; do
  repo_id="${REPO_PREFIX}${task}"
  raw_dir="${TRAIN_ROOT}/${task}"

  if [[ -d "${HF_HOME}/${repo_id}" && "${FORCE}" != "1" ]]; then
    echo "[skip] ${repo_id} already exists at ${HF_HOME}/${repo_id} (set FORCE=1 to regenerate)"
    SKIPPED_EXISTS+=("${repo_id}")
    continue
  fi

  first_h5=$(find -L "${raw_dir}" -type f -name "*.hdf5" -print -quit)
  if [[ -z "${first_h5}" ]]; then
    echo "[skip] ${repo_id}: no hdf5 files under ${raw_dir}"
    SKIPPED_EMPTY+=("${repo_id}")
    continue
  fi

  echo "==> Generating ${repo_id} from ${raw_dir}"
  bash "${GEN_SCRIPT}" "${raw_dir}" "${repo_id}"
done

echo "All done. Datasets reside under ${HF_HOME}"

if [[ ${#SKIPPED_EMPTY[@]} -gt 0 ]]; then
  echo "[summary] skipped (no hdf5): ${SKIPPED_EMPTY[*]}"
fi
if [[ ${#SKIPPED_EXISTS[@]} -gt 0 ]]; then
  echo "[summary] skipped (already exists): ${SKIPPED_EXISTS[*]}"
fi
