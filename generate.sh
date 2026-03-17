# 将 processed_data（含 instructions.json 的 subtasks/phase_info）转为 LeRobot HF 数据集
# 与 blocks_ranking_rgb_pi05_200 一致：会写入 instructions、subtasks、frame_idx、phase_info
# 用法: bash generate.sh <processed_data_dir> <repo_id>
#
# 单任务示例:
#   bash generate.sh ./processed_data/blocks_ranking_size-aloha-agilex_randomized_500-200 blocks_ranking_size_pi05_200
#   bash generate.sh ./processed_data/click_alarmclock-aloha-agilex_randomized_500-200 click_alarmclock_pi05_200
#   bash generate.sh ./processed_data/click_bell-aloha-agilex_randomized_500-200 click_bell_pi05_200
#
# 多任务训练（两种方式二选一）:
# 方式 A - 从已转换好的 5 个 LeRobot 数据集合并（无需重新跑 generate，推荐）:
#   export XDG_CACHE_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache
#   uv run scripts/merge_lerobot_datasets.py --output_repo pi05_multi_task_5 \
#     --source_repos beat_hammer_pi05_200 blocks_ranking_rgb_pi05_200 blocks_ranking_size_pi05_200 click_alarmclock_pi05_200 click_bell_pi05_200
# 方式 B - 从 processed_data 合并目录再转（推荐，比在 LeRobot 上逐帧合并更快）:
#   mkdir -p training_data/pi05_multi_task_5
#   cp -r processed_data/beat_block_hammer-aloha-agilex_randomized_500-200 \
#         processed_data/blocks_ranking_rgb-aloha-agilex_randomized_500-200 \
#         processed_data/blocks_ranking_size-aloha-agilex_randomized_500-200 \
#         processed_data/click_alarmclock-aloha-agilex_randomized_500-200 \
#         processed_data/click_bell-aloha-agilex_randomized_500-200 \
#         training_data/pi05_multi_task_5/
#   bash generate.sh ./training_data/pi05_multi_task_5 pi05_multi_task_5
# 然后: uv run scripts/compute_norm_stats.py --config-name pi05_aloha_full_base_multi_task_5
# 再: bash finetune.sh pi05_aloha_full_base_multi_task_5 <your_exp_name> <gpu_ids>
#
data_dir=${1}
repo_id=${2}
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir "$data_dir" --repo_id "$repo_id"

