# beat_block_hammer 干净数据 + 错误恢复数据 混合训练

## 思路是否可行

**可以。** 与「先训 50000 再在纯 recovery 上微调」相比，**同一 repo 里混合干净轨迹与 recovery 轨迹**，能避免策略只在「instruction + correction」分布上优化，从而减轻测试时「无 correction」下的分布偏移。

### 你描述的用法 vs 当前代码

| 数据类型 | 子任务 / Prompt | 进度 `progress_label` |
|----------|-----------------|------------------------|
| **错误恢复** | 丢弃 `[MASKED]` 段之后：`[Correction]` 拼进 **high_prompt**，`[Subtask]` 为 **low_prompt**（`LoadSubtaskFromInstructions`） | 若任意变体含 `[MASKED]`，则 `start_idx = masked_end`，`progress = (frame_idx - start_idx) / ((total_steps - 1) - start_idx)`（`ComputeProgressLabel`） |
| **干净（无错误）** | 无 `[Correction]`：按 phase 取子任务文本，**high = instruction**，**low = 子任务** | 无 `[MASKED]` 时 `start_idx = 0`，等价于 **从 episode 起点线性到 1**：`progress = frame_idx / (total_steps - 1)` |

说明：干净数据若写成 `current/total`，与代码里的 `(total_steps - 1)` 分母差一帧；**当前实现**是 `frame_idx ∈ [0, T-1]` 时 progress 从 0 到 1 的常用写法，**无需改代码**即可混合训练。

### `drop_masked_recovery_frames`

- 仅当 episode 的 `subtasks` 里出现 `[MASKED]` 且 `frame_idx < masked_end` 时才**重采样丢弃**。
- **干净 episode** 不含 `[MASKED]` → **不丢帧**。

---

## 数据准备（合并目录 → LeRobot）

在 `policy/pi05` 下，把四份 **processed_data**（或已整理好的同名目录）拷到同一 `training_data` 父目录，再 `generate.sh`。

目录名需与数据一致，例如：

- `beat_block_hammer-aloha-agilex_clean_50-50`
- `beat_block_hammer-aloha-agilex_randomized_500-500`
- `beat_block_hammer-demo_clean-recovery-66`
- `beat_block_hammer-demo_randomized-recovery-44`

若数据在 `processed_data/`，示例：

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05

mkdir -p training_data/beat_block_hammer_mixed_pi05
cp -r processed_data/beat_block_hammer-aloha-agilex_clean_50-50 \
      processed_data/beat_block_hammer-aloha-agilex_randomized_500-500 \
      processed_data/beat_block_hammer-demo_clean-recovery-66 \
      processed_data/beat_block_hammer-demo_randomized-recovery-44 \
      training_data/beat_block_hammer_mixed_pi05/

bash generate.sh ./training_data/beat_block_hammer_mixed_pi05 beat_block_hammer_mixed_pi05
```

生成 LeRobot 数据集 `repo_id=beat_block_hammer_mixed_pi05`（默认在 HF_LEROBOT_HOME / 项目 cache 下）。

---

## Norm

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
export XDG_CACHE_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache  # 若需要
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_beat_block_hammer_mixed_recovery
```

---

## 训练配置

- **Config 名**：`pi05_aloha_beat_block_hammer_mixed_recovery`
- **位置**：`src/openpi/training/config.py`（`TrainConfig`，`repo_id="beat_block_hammer_mixed_pi05"`）
- **初始权重**：默认与 recovery 单训相同：`progress_exp_1/15000/params`（可按需改成 multi_task 基座或其它 checkpoint）

启动：

```bash
bash finetune.sh pi05_aloha_beat_block_hammer_mixed_recovery mixed_recovery_exp_1 0,1
```

---

## 其它缓解「纯 recovery 训崩」的办法（回顾）

1. **混合训练**（本文）—— 首选。  
2. **降低 recovery 微调步数 / 提高干净比例**（数据层面调权或 oversample 干净 episode）。  
3. **两阶段但加 replay**：recovery 微调时混入一定比例的干净 beat_block_hammer batch（与混合 repo 等价，实现上二选一）。  
4. **学习率**：混合或 replay 时可略减小 LR（需在 `train.py`/optimizer 配置里改，当前未改）。  
5. **Norm**：混合后务必用 **同一混合数据集** 重算 `norm_stats`，不要沿用纯 recovery 的 stats。

---

## 辅助脚本

- `scripts/prepare_beat_block_hammer_mixed.sh`：一键拷贝 + `generate.sh`（路径可按机器修改）。
