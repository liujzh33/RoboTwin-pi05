# 10 个数据集转移 + generate 流程

## 1. 数据检查结论

已确认 10 个目录均存在，且 `instructions.json` 中均包含：
- **phase_info**（checkpoints、num_phases）
- **subtasks**（CycleVLA 风格多表述）

即：已加子任务描述（process_data_generic_v1）且已做语言丰富化（Qwen3-VL enrich）。

---

## 2. 从 pi05_multi_task_10 挪到 pi05_multi_task_5_v1.0（mv）

若 10 个任务已在 `training_data/pi05_multi_task_10` 内，且该目录里原有其它数据，可把这 10 个任务单独挪到新目录：

在 **pi05** 根目录执行：

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05

mkdir -p training_data/pi05_multi_task_5_v1.0

mv training_data/pi05_multi_task_10/beat_block_hammer-aloha-agilex_clean_50-50 \
   training_data/pi05_multi_task_10/beat_block_hammer-aloha-agilex_randomized_500-500 \
   training_data/pi05_multi_task_10/blocks_ranking_rgb-aloha-agilex_clean_50-50 \
   training_data/pi05_multi_task_10/blocks_ranking_rgb-aloha-agilex_randomized_500-500 \
   training_data/pi05_multi_task_10/blocks_ranking_size-aloha-agilex_clean_50-50 \
   training_data/pi05_multi_task_10/blocks_ranking_size-aloha-agilex_randomized_500-500 \
   training_data/pi05_multi_task_10/click_alarmclock-aloha-agilex_clean_50-50 \
   training_data/pi05_multi_task_10/click_alarmclock-aloha-agilex_randomized_500-500 \
   training_data/pi05_multi_task_10/click_bell-aloha-agilex_clean_50-50 \
   training_data/pi05_multi_task_10/click_bell-aloha-agilex_randomized_500-500 \
   training_data/pi05_multi_task_5_v1.0/
```

---

## 3. 转为 LeRobot 并生成数据集

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05

bash generate.sh ./training_data/pi05_multi_task_5_v1.0 pi05_multi_task_5_v1.0
```

（如需先激活环境：`source .venv/bin/activate`，或 generate.sh 内已用 `uv run` 则可不激活。）

---

## 4. 后续（按需）

- 计算 norm stats：  
  `uv run scripts/compute_norm_stats.py --config-name pi05_aloha_full_base_multi_task_5_v1.0`  
  （若配置名为 `pi05_multi_task_5_v1.0`，请与当前 config 名称一致。）
- 微调：  
  `bash finetune.sh pi05_aloha_full_base_multi_task_5_v1.0 <exp_name> <gpu_ids>`
