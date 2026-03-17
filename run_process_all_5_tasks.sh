#!/usr/bin/env bash
# 对 5 个任务分别处理 aloha-agilex_clean_50（50条）和 aloha-agilex_randomized_500（500条）
# 用法: cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05 && bash run_process_all_5_tasks.sh

set -e
cd "$(dirname "$0")"

TASKS=(beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell)

for task in "${TASKS[@]}"; do
  echo "========== $task: aloha-agilex_clean_50 (50) =========="
  bash process_data_pi0.sh "$task" aloha-agilex_clean_50 50
  echo "========== $task: aloha-agilex_randomized_500 (500) =========="
  bash process_data_pi0.sh "$task" aloha-agilex_randomized_500 500
done

echo "All 5 tasks x 2 settings done."
