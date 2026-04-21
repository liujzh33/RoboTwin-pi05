#!/usr/bin/env bash
# 将 dataset_recovery 下的错误恢复数据转为 processed_data，并保留 metadata。
# 输出到: /mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/<task_name>-<setting>-recovery-<N>/
# 命名含 recovery，避免覆盖原有 processed_data（如 beat_block_hammer-aloha-agilex_randomized_500-200）。
#
# 用法:
#   bash process_recovery_data_pi0.sh <task_name> <setting> [expert_data_num]
#   expert_data_num 可选，为 0 或省略时自动检测 data 目录下全部 episode。
#
# 示例（beat_block_hammer demo_clean，处理全部）:
#   cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
#   bash process_recovery_data_pi0.sh beat_block_hammer demo_clean 0
#
# 示例（只处理前 66 条）:
#   bash process_recovery_data_pi0.sh beat_block_hammer demo_clean 66

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

task_name=${1}
setting=${2}
expert_data_num=${3:-0}

if [ -z "$task_name" ] || [ -z "$setting" ]; then
  echo "Usage: bash process_recovery_data_pi0.sh <task_name> <setting> [expert_data_num]"
  echo "  expert_data_num: 0 = auto-detect all episodes in data folder"
  echo "Example: bash process_recovery_data_pi0.sh beat_block_hammer demo_clean 0"
  exit 1
fi

python scripts/process_data_recovery.py "$task_name" "$setting" "$expert_data_num"
