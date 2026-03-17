#!/usr/bin/env bash
# Traverse all tasks under data/data_new and convert them to processed_data + training_data layout.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVED_DATA_ROOT="$(readlink -f "${ROOT_DIR}/../../data/data_new2")"
DATA_ROOT="${DATA_ROOT:-${RESOLVED_DATA_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/processed_data}"
TRAIN_ROOT="${TRAIN_ROOT:-${ROOT_DIR}/training_data/pi05_all}"
DESC_TYPE="${DESC_TYPE:-seen}"

mkdir -p "${OUTPUT_ROOT}" "${TRAIN_ROOT}"

if [[ $# -gt 0 ]]; then
  mapfile -t TASKS < <(printf "%s\n" "$@")
else
  mapfile -t TASKS < <(find -L "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "No tasks found under ${DATA_ROOT}" >&2
  exit 1
fi

for task_dir in "${TASKS[@]}"; do
  echo "==> Processing ${task_dir}"
  python "${ROOT_DIR}/scripts/process_data_data_new.py" "${task_dir}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --desc-type "${DESC_TYPE}"

  latest_out=$(ls -dt "${OUTPUT_ROOT}/${task_dir}-"* 2>/dev/null | head -n1 || true)
  if [[ -z "${latest_out}" ]]; then
    echo "[warn] no output found for ${task_dir}, skipping copy"
    continue
  fi

  rsync -a "${latest_out}/" "${TRAIN_ROOT}/${task_dir}/"
done

echo "All done. Training data staged at ${TRAIN_ROOT}"
