#!/usr/bin/env bash
# 对 5 种任务 × 2 种 setting：先添加子任务描述（process_data_generic_v1），再批量画图。
# 同一任务的 clean_50 与 randomized_500 使用同一套 task_definitions 子任务逻辑。
#
# 用法:
#   cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
#   source .venv/bin/activate
#   export XDG_CACHE_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache
#   bash run_subtask_and_batch_plot_all_5_tasks.sh
#
# ---------- 所有命令行参考（每条先 process_data_generic_v1 再对应 batch_analyze）----------
#
# beat_block_hammer (clean_50)
#   python scripts/process_data_generic_v1.py --task_name beat_block_hammer --data_dir processed_data/beat_block_hammer-aloha-agilex_clean_50-50
#   python "scripts/batch_analyze_trajectories_beat_block_hammer .py" --data_dir processed_data/beat_block_hammer-aloha-agilex_clean_50-50 --output_dir analysis_results/analysis_results_beat_block_hammer_aloha-agilex_clean_50
#
# beat_block_hammer (randomized_500)
#   python scripts/process_data_generic_v1.py --task_name beat_block_hammer --data_dir processed_data/beat_block_hammer-aloha-agilex_randomized_500-500
#   python "scripts/batch_analyze_trajectories_beat_block_hammer .py" --data_dir processed_data/beat_block_hammer-aloha-agilex_randomized_500-500 --output_dir analysis_results/analysis_results_beat_block_hammer_aloha-agilex_randomized_500
#
# blocks_ranking_rgb (clean_50, 用 blocks_ranking_rgb_v1)
#   python scripts/process_data_generic_v1.py --task_name blocks_ranking_rgb_v1 --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_clean_50-50
#   python scripts/batch_analyze_trajectories_blocks_ranking_rgb.py --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_clean_50-50 --output_dir analysis_results/analysis_results_blocks_ranking_rgb_aloha-agilex_clean_50
#
# blocks_ranking_rgb (randomized_500, 用 blocks_ranking_rgb_v1)
#   python scripts/process_data_generic_v1.py --task_name blocks_ranking_rgb_v1 --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_randomized_500-500
#   python scripts/batch_analyze_trajectories_blocks_ranking_rgb.py --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_randomized_500-500 --output_dir analysis_results/analysis_results_blocks_ranking_rgb_aloha-agilex_randomized_500
#
# blocks_ranking_size (clean_50, 用 blocks_ranking_size_v1)
#   python scripts/process_data_generic_v1.py --task_name blocks_ranking_size_v1 --data_dir processed_data/blocks_ranking_size-aloha-agilex_clean_50-50
#   python scripts/batch_analyze_trajectories_blocks_ranking_size.py --data_dir processed_data/blocks_ranking_size-aloha-agilex_clean_50-50 --output_dir analysis_results/analysis_results_blocks_ranking_size_aloha-agilex_clean_50
#
# blocks_ranking_size (randomized_500, 用 blocks_ranking_size_v1)
#   python scripts/process_data_generic_v1.py --task_name blocks_ranking_size_v1 --data_dir processed_data/blocks_ranking_size-aloha-agilex_randomized_500-500
#   python scripts/batch_analyze_trajectories_blocks_ranking_size.py --data_dir processed_data/blocks_ranking_size-aloha-agilex_randomized_500-500 --output_dir analysis_results/analysis_results_blocks_ranking_size_aloha-agilex_randomized_500
#
# click_alarmclock (clean_50)
#   python scripts/process_data_generic_v1.py --task_name click_alarmclock --data_dir processed_data/click_alarmclock-aloha-agilex_clean_50-50
#   python scripts/batch_analyze_trajectories_click_alarmclock.py --data_dir processed_data/click_alarmclock-aloha-agilex_clean_50-50 --output_dir analysis_results/analysis_results_click_alarmclock_aloha-agilex_clean_50
#
# click_alarmclock (randomized_500)
#   python scripts/process_data_generic_v1.py --task_name click_alarmclock --data_dir processed_data/click_alarmclock-aloha-agilex_randomized_500-500
#   python scripts/batch_analyze_trajectories_click_alarmclock.py --data_dir processed_data/click_alarmclock-aloha-agilex_randomized_500-500 --output_dir analysis_results/analysis_results_click_alarmclock_aloha-agilex_randomized_500
#
# click_bell (clean_50)
#   python scripts/process_data_generic_v1.py --task_name click_bell --data_dir processed_data/click_bell-aloha-agilex_clean_50-50
#   python scripts/batch_analyze_trajectories_click_bell.py --data_dir processed_data/click_bell-aloha-agilex_clean_50-50 --output_dir analysis_results/analysis_results_click_bell_aloha-agilex_clean_50
#
# click_bell (randomized_500)
#   python scripts/process_data_generic_v1.py --task_name click_bell --data_dir processed_data/click_bell-aloha-agilex_randomized_500-500
#   python scripts/batch_analyze_trajectories_click_bell.py --data_dir processed_data/click_bell-aloha-agilex_randomized_500-500 --output_dir analysis_results/analysis_results_click_bell_aloha-agilex_randomized_500
#
# ---------- 以上为 5 任务 × 2 setting 的完整命令行 ----------

set -e
cd "$(dirname "$0")"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache}"

# 5 任务 × 2 setting： (task_name, setting, expert_data_num)
run_one() {
  local task_name="$1"
  local setting="$2"
  local num="$3"
  local data_dir="processed_data/${task_name}-${setting}-${num}"
  local out_dir="analysis_results/analysis_results_${task_name}_${setting}"

  # process_data_generic_v1 使用的 task_name（blocks_ranking 用 v1 处理器）
  local processor_name="$task_name"
  if [[ "$task_name" == "blocks_ranking_rgb" ]]; then
    processor_name="blocks_ranking_rgb_v1"
  elif [[ "$task_name" == "blocks_ranking_size" ]]; then
    processor_name="blocks_ranking_size_v1"
  fi

  echo "========== $task_name | $setting =========="
  if [[ ! -d "$data_dir" ]]; then
    echo "  Skip (data_dir not found): $data_dir"
    return 0
  fi

  echo "  [1/2] Adding subtask descriptions (processor: $processor_name)..."
  python scripts/process_data_generic_v1.py \
    --task_name "$processor_name" \
    --data_dir "$data_dir"

  echo "  [2/2] Batch plotting..."
  case "$task_name" in
    beat_block_hammer)
      python "scripts/batch_analyze_trajectories_beat_block_hammer .py" \
        --data_dir "$data_dir" \
        --output_dir "$out_dir"
      ;;
    blocks_ranking_rgb)
      python scripts/batch_analyze_trajectories_blocks_ranking_rgb.py \
        --data_dir "$data_dir" \
        --output_dir "$out_dir"
      ;;
    blocks_ranking_size)
      python scripts/batch_analyze_trajectories_blocks_ranking_size.py \
        --data_dir "$data_dir" \
        --output_dir "$out_dir"
      ;;
    click_alarmclock)
      python scripts/batch_analyze_trajectories_click_alarmclock.py \
        --data_dir "$data_dir" \
        --output_dir "$out_dir"
      ;;
    click_bell)
      python scripts/batch_analyze_trajectories_click_bell.py \
        --data_dir "$data_dir" \
        --output_dir "$out_dir"
      ;;
    *)
      echo "  Unknown task: $task_name"
      return 1
      ;;
  esac
  echo "  Done: $out_dir"
}

# 执行 10 组：5 任务 × (clean_50, randomized_500)
TASKS=(beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell)
for task in "${TASKS[@]}"; do
  run_one "$task" "aloha-agilex_clean_50" 50
  run_one "$task" "aloha-agilex_randomized_500" 500
done

echo "All 5 tasks x 2 settings (subtask + batch plot) done."
