# 10 组命令行参考（5 任务 × 2 setting：先 process_data_generic_v1，再 batch_analyze）

在项目根目录执行前请先：
```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
source .venv/bin/activate
export XDG_CACHE_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache
```

---

## 1. beat_block_hammer — clean_50

```bash
python scripts/process_data_generic_v1.py --task_name beat_block_hammer --data_dir processed_data/beat_block_hammer-aloha-agilex_clean_50-50
python "scripts/batch_analyze_trajectories_beat_block_hammer .py" --data_dir processed_data/beat_block_hammer-aloha-agilex_clean_50-50 --output_dir analysis_results/analysis_results_beat_block_hammer_aloha-agilex_clean_50
```

## 2. beat_block_hammer — randomized_500

```bash
python scripts/process_data_generic_v1.py --task_name beat_block_hammer --data_dir processed_data/beat_block_hammer-aloha-agilex_randomized_500-500
python "scripts/batch_analyze_trajectories_beat_block_hammer .py" --data_dir processed_data/beat_block_hammer-aloha-agilex_randomized_500-500 --output_dir analysis_results/analysis_results_beat_block_hammer_aloha-agilex_randomized_500
```

## 3. blocks_ranking_rgb — clean_50（使用 blocks_ranking_rgb_v1 处理器，13 阶段）

```bash
python scripts/process_data_generic_v1.py --task_name blocks_ranking_rgb_v1 --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_clean_50-50
python scripts/batch_analyze_trajectories_blocks_ranking_rgb.py --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_clean_50-50 --output_dir analysis_results/analysis_results_blocks_ranking_rgb_aloha-agilex_clean_50
```

## 4. blocks_ranking_rgb — randomized_500（使用 blocks_ranking_rgb_v1 处理器）

```bash
python scripts/process_data_generic_v1.py --task_name blocks_ranking_rgb_v1 --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_randomized_500-500
python scripts/batch_analyze_trajectories_blocks_ranking_rgb.py --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_randomized_500-500 --output_dir analysis_results/analysis_results_blocks_ranking_rgb_aloha-agilex_randomized_500
```

## 5. blocks_ranking_size — clean_50（使用 blocks_ranking_size_v1 处理器，13 阶段）

```bash
python scripts/process_data_generic_v1.py --task_name blocks_ranking_size_v1 --data_dir processed_data/blocks_ranking_size-aloha-agilex_clean_50-50
python scripts/batch_analyze_trajectories_blocks_ranking_size.py --data_dir processed_data/blocks_ranking_size-aloha-agilex_clean_50-50 --output_dir analysis_results/analysis_results_blocks_ranking_size_aloha-agilex_clean_50
```

## 6. blocks_ranking_size — randomized_500（使用 blocks_ranking_size_v1 处理器）

```bash
python scripts/process_data_generic_v1.py --task_name blocks_ranking_size_v1 --data_dir processed_data/blocks_ranking_size-aloha-agilex_randomized_500-500
python scripts/batch_analyze_trajectories_blocks_ranking_size.py --data_dir processed_data/blocks_ranking_size-aloha-agilex_randomized_500-500 --output_dir analysis_results/analysis_results_blocks_ranking_size_aloha-agilex_randomized_500
```

## 7. click_alarmclock — clean_50

```bash
python scripts/process_data_generic_v1.py --task_name click_alarmclock --data_dir processed_data/click_alarmclock-aloha-agilex_clean_50-50
python scripts/batch_analyze_trajectories_click_alarmclock.py --data_dir processed_data/click_alarmclock-aloha-agilex_clean_50-50 --output_dir analysis_results/analysis_results_click_alarmclock_aloha-agilex_clean_50
```

## 8. click_alarmclock — randomized_500

```bash
python scripts/process_data_generic_v1.py --task_name click_alarmclock --data_dir processed_data/click_alarmclock-aloha-agilex_randomized_500-500
python scripts/batch_analyze_trajectories_click_alarmclock.py --data_dir processed_data/click_alarmclock-aloha-agilex_randomized_500-500 --output_dir analysis_results/analysis_results_click_alarmclock_aloha-agilex_randomized_500
```

## 9. click_bell — clean_50

```bash
python scripts/process_data_generic_v1.py --task_name click_bell --data_dir processed_data/click_bell-aloha-agilex_clean_50-50
python scripts/batch_analyze_trajectories_click_bell.py --data_dir processed_data/click_bell-aloha-agilex_clean_50-50 --output_dir analysis_results/analysis_results_click_bell_aloha-agilex_clean_50
```

## 10. click_bell — randomized_500

```bash
python scripts/process_data_generic_v1.py --task_name click_bell --data_dir processed_data/click_bell-aloha-agilex_randomized_500-500
python scripts/batch_analyze_trajectories_click_bell.py --data_dir processed_data/click_bell-aloha-agilex_randomized_500-500 --output_dir analysis_results/analysis_results_click_bell_aloha-agilex_randomized_500
```

---

说明：`beat_block_hammer` 的 batch 脚本文件名中含空格，故使用 `"scripts/batch_analyze_trajectories_beat_block_hammer .py"` 并加引号。
