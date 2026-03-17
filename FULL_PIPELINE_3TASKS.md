# 完整任务流程（以 blocks_ranking_size、click_alarmclock、click_bell 为例）

从原始数据到微调的全流程命令行。原始数据需位于 `/mnt/data1/liujingzhi/dataset/{task_name}/aloha-agilex_randomized_500`，且包含 `data/`、`instructions/` 目录。

---

## 0. 环境准备（每开新终端执行一次）

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05  &&
source .venv/bin/activate  &&
export XDG_CACHE_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache
```

---

## 1. 原始数据 → processed_data（HDF5 + instructions.json）

每任务 200 条，输出目录：`processed_data/{task_name}-aloha-agilex_randomized_500-200`。

```bash
# blocks_ranking_size
bash process_data_pi0.sh blocks_ranking_size aloha-agilex_randomized_500 200

# click_alarmclock
bash process_data_pi0.sh click_alarmclock aloha-agilex_randomized_500 200

# click_bell
bash process_data_pi0.sh click_bell aloha-agilex_randomized_500 200
```

---

## 2. 标注子任务（阶段划分 + subtasks/phase_info 写入 instructions.json）

需存在 `task_definitions/{task_name}.py`（如 `blocks_ranking_size.py`、`click_alarmclock.py`、`click_bell.py`）。

```bash
python scripts/process_data_generic.py --task_name blocks_ranking_size --data_dir processed_data/blocks_ranking_size-aloha-agilex_randomized_500-200

python scripts/process_data_generic.py --task_name click_alarmclock --data_dir processed_data/click_alarmclock-aloha-agilex_randomized_500-200

python scripts/process_data_generic.py --task_name click_bell --data_dir processed_data/click_bell-aloha-agilex_randomized_500-200
```

---

## 3. 丰富子任务表述（多样化描述）

enrich 脚本内 `DATA_ROOT` 已写死为上述 processed_data 路径，直接执行即可。

```bash
python Qwen3-VL/enrich_blocks_ranking_size.py
python Qwen3-VL/enrich_click_alarmclock.py
python Qwen3-VL/enrich_click_bell.py
```

若你的 processed_data 路径不同，需修改各 `enrich_*.py` 顶部的 `DATA_ROOT`。

---

## 4. 多任务数据合并到 training_data 并转为 LeRobot

将 3 个任务的 processed_data 拷到同一目录，再跑一次 `generate.sh` 得到多任务 LeRobot 数据集。

```bash
mkdir -p training_data/pi05_multi_task_3
cp -r processed_data/blocks_ranking_size-aloha-agilex_randomized_500-200 \
      processed_data/click_alarmclock-aloha-agilex_randomized_500-200 \
      processed_data/click_bell-aloha-agilex_randomized_500-200 \
      training_data/pi05_multi_task_3/

bash generate.sh ./training_data/pi05_multi_task_3 pi05_multi_task_3
```

数据集会出现在：`.cache/huggingface/lerobot/pi05_multi_task_3/`。

---

## 5. 计算归一化统计（norm）

已提供配置 `pi05_aloha_full_base_multi_task_3`（`config.py` 中 `repo_id="pi05_multi_task_3"`）：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_full_base_multi_task_3
```

---

## 6. 微调

使用上一步的配置名与实验名、GPU：

```bash
bash finetune.sh pi05_aloha_full_base_multi_task_3 multi3_exp 5,7
```

---

## 一键执行脚本（按顺序跑完全流程）

下面脚本假设：  
- 已执行过「0. 环境准备」；  
- 原始数据已在 `/mnt/data1/liujingzhi/dataset/{task_name}/aloha-agilex_randomized_500`。  
配置 `pi05_aloha_full_base_multi_task_3` 已在 `config.py` 中提供。

```bash
# 1) 原始数据 → processed_data
bash process_data_pi0.sh blocks_ranking_size aloha-agilex_randomized_500 200
bash process_data_pi0.sh click_alarmclock aloha-agilex_randomized_500 200
bash process_data_pi0.sh click_bell aloha-agilex_randomized_500 200

# 2) 标注子任务
python scripts/process_data_generic.py --task_name blocks_ranking_size --data_dir processed_data/blocks_ranking_size-aloha-agilex_randomized_500-200
python scripts/process_data_generic.py --task_name click_alarmclock --data_dir processed_data/click_alarmclock-aloha-agilex_randomized_500-200
python scripts/process_data_generic.py --task_name click_bell --data_dir processed_data/click_bell-aloha-agilex_randomized_500-200

# 3) 丰富表述
python Qwen3-VL/enrich_blocks_ranking_size.py
python Qwen3-VL/enrich_click_alarmclock.py
python Qwen3-VL/enrich_click_bell.py

# 4) 合并并转为 LeRobot
mkdir -p training_data/pi05_multi_task_3
cp -r processed_data/blocks_ranking_size-aloha-agilex_randomized_500-200 \
      processed_data/click_alarmclock-aloha-agilex_randomized_500-200 \
      processed_data/click_bell-aloha-agilex_randomized_500-200 \
      training_data/pi05_multi_task_3/
bash generate.sh ./training_data/pi05_multi_task_3 pi05_multi_task_3

# 5) norm 计算
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_full_base_multi_task_3

# 6) 微调
bash finetune.sh pi05_aloha_full_base_multi_task_3 multi3_exp 5,7
```

若使用已有的 5 任务配置与数据，只需把上面的 `pi05_multi_task_3` 换成 `pi05_multi_task_5`，配置名换成 `pi05_aloha_full_base_multi_task_5`，并在第 4 步中复制 5 个任务的 processed_data 到 `training_data/pi05_multi_task_5` 即可。
