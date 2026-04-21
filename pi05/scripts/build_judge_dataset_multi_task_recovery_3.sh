#!/usr/bin/env bash
# 从 training_data/pi05_multi_task_recovery_3 构建三任务 Judge 数据集（每个 episode 一条）
#
# 用法（在 policy/pi05 下）:
#   bash scripts/build_judge_dataset_multi_task_recovery_3.sh
#
# 输出: judge_dataset/recovery_judge_3tasks/
#   images/
#   recovery_judge_errors.json
#   recovery_judge_success.json
#   recovery_judge_dataset_merged.json
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1
python scripts/extract_judge_dataset_multi_task_recovery_3.py "$@"
