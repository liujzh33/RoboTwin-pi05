#!/bin/bash
set -euo pipefail

# Auto-evaluate pi05 by reading tasks.yaml.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASKS_FILE_DEFAULT="${SCRIPT_DIR}/tasks.yaml"
TASK_CONFIG_DEFAULT="demo_clean"
TRAIN_CONFIG_NAME_DEFAULT="pi05_aloha_full_base_all"
MODEL_NAME_DEFAULT="pi05_ft_4a100"
SEED_DEFAULT="0"
GPU_IDS_DEFAULT=""
GPU_ID_DEFAULT="2" # fallback when GPU_IDS is unset
MIN_FREE_GB_DEFAULT="40"
MIN_FREE_RATIO_DEFAULT="0.5"

usage() {
  cat << EOF
Usage: bash auto_eval.sh

Environment overrides:
  TASKS_FILE           Path to tasks.yaml (default: ${TASKS_FILE_DEFAULT})
  TASK_CONFIG          Task config for eval.sh (default: ${TASK_CONFIG_DEFAULT})
  TRAIN_CONFIG_NAME    Train config name (default: ${TRAIN_CONFIG_NAME_DEFAULT})
  MODEL_NAME           Model name / ckpt setting (default: ${MODEL_NAME_DEFAULT})
  SEED                 Random seed (default: ${SEED_DEFAULT})
  GPU_IDS              Comma-separated GPU ids for parallel eval (auto-detect if empty)
  GPU_ID               Single GPU id (used only if GPU_IDS is empty; default: ${GPU_ID_DEFAULT})
  MIN_FREE_GB          Min free memory (GB) to treat GPU as available (default: ${MIN_FREE_GB_DEFAULT})
  MIN_FREE_RATIO       Min free/total ratio to treat GPU as available (default: ${MIN_FREE_RATIO_DEFAULT})
  LOG_DIR              Log output dir (default: ${SCRIPT_DIR}/eval_logs_YYYYmmdd_HHMMSS)
  DRY_RUN              Set to 1 to only print commands

Example:
  TASKS_FILE=/path/to/tasks.yaml \\
  GPU_IDS=0,1,2 bash auto_eval.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TASKS_FILE="${TASKS_FILE:-$TASKS_FILE_DEFAULT}"
TASK_CONFIG="${TASK_CONFIG:-$TASK_CONFIG_DEFAULT}"
TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-$TRAIN_CONFIG_NAME_DEFAULT}"
MODEL_NAME="${MODEL_NAME:-$MODEL_NAME_DEFAULT}"
SEED="${SEED:-$SEED_DEFAULT}"
GPU_ID="${GPU_ID:-$GPU_ID_DEFAULT}"
GPU_IDS_ENV="${GPU_IDS:-$GPU_IDS_DEFAULT}"
MIN_FREE_GB="${MIN_FREE_GB:-$MIN_FREE_GB_DEFAULT}"
MIN_FREE_RATIO="${MIN_FREE_RATIO:-$MIN_FREE_RATIO_DEFAULT}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/eval_logs_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "$TASKS_FILE" ]; then
  echo "Error: tasks file not found: $TASKS_FILE"
  exit 1
fi

if [ ! -f "${SCRIPT_DIR}/eval.sh" ]; then
  echo "Error: eval.sh not found in: ${SCRIPT_DIR}"
  exit 1
fi

have_nvidia_smi=0
if command -v nvidia-smi >/dev/null 2>&1; then
  have_nvidia_smi=1
fi

# Parse GPUs: prefer GPU_IDS if provided; otherwise auto-detect; fallback to single GPU_ID.
if [ -n "$GPU_IDS_ENV" ]; then
  GPU_IDS_ENV_CLEAN="$(echo "$GPU_IDS_ENV" | tr -d ' ')"
  IFS=',' read -ra GPU_IDS <<< "$GPU_IDS_ENV_CLEAN"
else
  if [ "$have_nvidia_smi" -eq 1 ]; then
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  else
    GPU_IDS=("$GPU_ID")
  fi
fi

if [ ${#GPU_IDS[@]} -eq 0 ]; then
  GPU_IDS=("$GPU_ID")
fi

echo "Using GPUs: ${GPU_IDS[*]}"
if [ "$have_nvidia_smi" -eq 1 ]; then
  echo "GPU availability threshold: free >= ${MIN_FREE_GB}GB or free/total >= ${MIN_FREE_RATIO}"
fi

declare -A gpu_pid
declare -A task_log

for gpu in "${GPU_IDS[@]}"; do
  gpu_pid[$gpu]=""
done

is_running() {
  [ -n "$1" ] && kill -0 "$1" 2>/dev/null
}

is_gpu_available() {
  local gpu_id=$1
  if [ "$have_nvidia_smi" -ne 1 ]; then
    return 0
  fi
  local total used free_mib ratio min_free_mib
  read -r total used < <(
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$gpu_id" \
      | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
  )
  if [ -z "$total" ] || [ -z "$used" ]; then
    return 1
  fi
  free_mib=$((total - used))
  ratio=$(awk -v f="$free_mib" -v t="$total" 'BEGIN{if (t==0) print 0; else print f/t}')
  min_free_mib=$((MIN_FREE_GB * 1024))
  awk -v fm="$free_mib" -v r="$ratio" -v min_mib="$min_free_mib" -v min_r="$MIN_FREE_RATIO" \
    'BEGIN{exit !((fm >= min_mib) || (r >= min_r))}'
}

get_free_gpu() {
  while true; do
    for gpu in "${GPU_IDS[@]}"; do
      if ! is_running "${gpu_pid[$gpu]}" && is_gpu_available "$gpu"; then
        echo "$gpu"
        return 0
      fi
    done
    sleep 2
  done
}

show_progress() {
  local current=$1
  local total=$2
  local percent=$((current * 100 / total))
  local bar_length=40
  local filled=$((percent * bar_length / 100))
  printf "\r["
  printf "%${filled}s" | tr ' ' '='
  printf "%$((bar_length - filled))s" | tr ' ' ' '
  printf "] %d%% (%d/%d)" "$percent" "$current" "$total"
}

mapfile -t all_tasks < <(
  awk '
    BEGIN{in_list=0}
    /^[[:space:]]*#/ {next}
    /^[[:space:]]*tasks[[:space:]]*:/ {in_list=1; next}
    /^[[:space:]]*-[[:space:]]*/ {
      line=$0
      sub(/^[[:space:]]*-[[:space:]]*/, "", line)
      if (line != "") print line
    }
  ' "$TASKS_FILE"
)

if [ ${#all_tasks[@]} -eq 0 ]; then
  echo "Error: no tasks found in $TASKS_FILE"
  exit 1
fi

mkdir -p "$LOG_DIR"
tasks_file_out="${LOG_DIR}/tasks_list.txt"
printf "%s\n" "${all_tasks[@]}" > "$tasks_file_out"

summary="${LOG_DIR}/evaluation_summary.txt"
cat > "$summary" << EOF
pi05 Auto Evaluation Summary
============================
Date: $(date)
Tasks file: $TASKS_FILE
Task config: $TASK_CONFIG
Train config: $TRAIN_CONFIG_NAME
Model name: $MODEL_NAME
Seed: $SEED
GPU IDs: ${GPU_IDS[*]:-$GPU_ID}
Min free GB: $MIN_FREE_GB
Min free ratio: $MIN_FREE_RATIO
Log dir: $LOG_DIR
Total tasks: ${#all_tasks[@]}

Task Results:
-------------
EOF

echo "Log directory: $LOG_DIR"
echo "Tasks: ${#all_tasks[@]} (saved to $tasks_file_out)"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY_RUN enabled; commands will be printed only."
fi

pids=()

for task in "${all_tasks[@]}"; do
  log_file="${LOG_DIR}/${task}.log"
  task_log["$task"]="$log_file"

  if [ "$DRY_RUN" -eq 1 ]; then
    gpu_id="${GPU_IDS[$(( ${#pids[@]} % ${#GPU_IDS[@]} ))]}"
    cmd=(bash "${SCRIPT_DIR}/eval.sh" "$task" "$TASK_CONFIG" "$TRAIN_CONFIG_NAME" "$MODEL_NAME" "$SEED" "$gpu_id")
    echo "DRY_RUN: ${cmd[*]}"
    continue
  fi

  gpu_id=$(get_free_gpu)
  cmd=(bash "${SCRIPT_DIR}/eval.sh" "$task" "$TASK_CONFIG" "$TRAIN_CONFIG_NAME" "$MODEL_NAME" "$SEED" "$gpu_id")

  echo "Launching: $task on GPU $gpu_id"
  (
    "${cmd[@]}" > "$log_file" 2>&1
    echo "EXIT_CODE=$?" >> "$log_file"
  ) &

  pid=$!
  gpu_pid[$gpu_id]=$pid
  pids+=("$pid")
  sleep 1
done

if [ "$DRY_RUN" -eq 0 ]; then
  echo "Waiting for completion..."
  completed=0
  total=${#pids[@]}
  for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else :; fi
    ((completed++))
    show_progress "$completed" "$total"
  done
  echo
fi

success=0
failed=0

for task in "${all_tasks[@]}"; do
  log_file="${task_log[$task]}"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf "%s\tDRY_RUN\n" "$task" >> "$summary"
    continue
  fi
  if [ ! -f "$log_file" ]; then
    printf "%s\tLOG_MISSING\n" "$task" >> "$summary"
    ((failed++))
    continue
  fi
  if grep -q "EXIT_CODE=0" "$log_file"; then
    printf "%s\tSUCCESS\n" "$task" >> "$summary"
    ((success++))
  else
    printf "%s\tFAILED\n" "$task" >> "$summary"
    ((failed++))
  fi
done

if [ "$DRY_RUN" -eq 0 ]; then
  cat >> "$summary" << EOF

Summary Statistics:
-------------------
SUCCESS: $success
FAILED: $failed
TOTAL: ${#all_tasks[@]}
EOF
fi

echo "Summary: $summary"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY_RUN enabled; no evaluations executed."
elif [ $failed -eq 0 ]; then
  echo "All tasks completed successfully."
else
  echo "Some tasks failed. Check logs for details."
fi
