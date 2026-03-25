#!/usr/bin/env bash
# 8 组评估：4 任务 × (demo_randomized + demo_clean)，统一 subtask_progress_from_base_exp_1；任一步失败则退出
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate local venv if present; otherwise keep current shell env.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

TRAIN="pi05_aloha_multi_task_5_v1_0_subtask_progress_from_pi05_base"
MODEL="subtask_progress_from_base_exp_1"
SEED=0
GPU=2
CKPT=20000

run_eval() {
  local task_name="$1"
  local task_config="$2"
  bash eval.sh "$task_name" "$task_config" "$TRAIN" "$MODEL" "$SEED" "$GPU" "$CKPT"
}

# echo "=== [1/8] blocks_ranking_rgb demo_randomized ==="
# run_eval blocks_ranking_rgb demo_randomized

# echo "=== [2/8] blocks_ranking_rgb demo_clean ==="
# run_eval blocks_ranking_rgb demo_clean

echo "=== [3/8] blocks_ranking_size demo_randomized ==="
run_eval blocks_ranking_size demo_randomized

echo "=== [4/8] blocks_ranking_size demo_clean ==="
run_eval blocks_ranking_size demo_clean

echo "=== [5/8] click_alarmclock demo_randomized ==="
run_eval click_alarmclock demo_randomized

echo "=== [6/8] click_alarmclock demo_clean ==="
run_eval click_alarmclock demo_clean

echo "=== [7/8] click_bell demo_randomized ==="
run_eval click_bell demo_randomized

echo "=== [8/8] click_bell demo_clean ==="
run_eval click_bell demo_clean

echo "All 8 evals finished OK."
