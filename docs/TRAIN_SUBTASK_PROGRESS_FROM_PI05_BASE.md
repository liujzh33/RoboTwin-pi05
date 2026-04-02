# 干净 `pi05_multi_task_5_v1.0` + 从 `pi05_base` 训练子任务与进度

## 说明

- **数据**：LeRobot 数据集 **`repo_id=pi05_multi_task_5_v1.0`**，本地目录应为  
  `{HF_LEROBOT_HOME}/pi05_multi_task_5_v1.0`  
  例如：  
  `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache/huggingface/lerobot/pi05_multi_task_5_v1.0`  
  则设置：  
  `export HF_LEROBOT_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache/huggingface/lerobot`

- **初始权重**：`/mnt/data/linbingqian/openpi/checkpoints11/pi05_base/params`  
  使用 **`CheckpointWeightLoaderWithDefaults`**：`pi05_base` 里没有的权重（如 **progress 头**）保持**随机初始化**并参与训练。

- **训练目标**：与现有 Pi0.5 一致，`pi05.py` 里 **`compute_loss`** 同时包含：  
  - 子任务 CE  
  - 动作 flow matching（MSE）  
  - 进度 MSE（需 `ComputeProgressLabel` 提供 `progress_label`）  
  **若只想训子任务+进度、关掉动作损失，需要改 `pi05.py` 或加开关**，当前配置**仍会训练动作分支**。

- **Config 名**：`pi05_aloha_multi_task_5_v1_0_subtask_progress_from_pi05_base`（见 `src/openpi/training/config.py`）

---

## 命令行

在 `policy/pi05` 下：

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05

# 让 LeRobot 找到本地数据集（按你实际路径）
export HF_LEROBOT_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache/huggingface/lerobot
# 若 train/compute_norm 还用 HuggingFace 缓存，可一并：
export XDG_CACHE_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache

source .venv/bin/activate
```

**1. Norm（必须先跑，否则会报 norm_stats 缺失）**

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_multi_task_5_v1_0_subtask_progress_from_pi05_base
```

**2. 训练**

```bash
bash finetune.sh pi05_aloha_multi_task_5_v1_0_subtask_progress_from_pi05_base subtask_progress_from_base_exp_1 0,1
```

将最后的 `0,1` 换成你的 GPU。

---

## 若 `pi05_base` 路径在本机不存在

把 `config.py` 里该 TrainConfig 的 `weight_loader` 的 `params_path` 改成你机器上真实的 `pi05_base/params`，或把权重拷到上述路径。
