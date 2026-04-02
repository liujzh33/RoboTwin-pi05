# Recovery 训练流程与「手臂乱动」排查

本文档说明 checkpoint `pi05_aloha_beat_block_hammer_recovery_from15000` / `recovery_exp_1/15000` 的**训练配置、数据、标注与命令**，以及可能导致**测试时手臂乱动、成功率 0** 的原因与排查位置。

---

## 1. 训练配置（代码位置）

| 项目 | 位置 | 说明 |
|------|------|------|
| **TrainConfig** | `src/openpi/training/config.py` **L754–792** | `name="pi05_aloha_beat_block_hammer_recovery_from15000"` |
| **数据 repo** | 同上 L759 | `repo_id="beat_block_hammer_recovery_pi05"` |
| **drop_masked** | 同上 L778–779 | `base_config=DataConfig(..., drop_masked_recovery_frames=True)` |
| **初始权重** | 同上 L783–786 | `CheckpointWeightLoaderWithDefaults(..., progress_exp_1/15000/params)` |
| **训练步数** | 同上 L787 | `num_train_steps=50001`, `batch_size=64`, `fsdp_devices=2` |

初始权重路径（来自 progress 多任务 checkpoint）：

```
/mnt/data1/liujingzhi/RoboTwin/policy/pi05/checkpoints/
  pi05_aloha_full_base_multi_task_5_v1_0_progress/progress_exp_1/15000/params
```

---

## 2. 数据来源与构建

- **LeRobot 数据集名**：`beat_block_hammer_recovery_pi05`（config 里写的 `repo_id`）。
- **原始数据**：由 `generate.sh` 从 **processed_data** 转成 LeRobot HF 数据集。通常由两段合并后一起转：
  - `processed_data/beat_block_hammer-demo_clean-recovery-66`
  - `processed_data/beat_block_hammer-demo_randomized-recovery-44`
- **合并与生成**（需在 `policy/pi05` 下执行）：

```bash
# 1) 合并 processed_data 到 training_data
mkdir -p training_data/beat_block_hammer_recovery_pi05
cp -r processed_data/beat_block_hammer-demo_clean-recovery-66 \
      processed_data/beat_block_hammer-demo_randomized-recovery-44 \
      training_data/beat_block_hammer_recovery_pi05/

# 2) 转为 LeRobot 格式
bash generate.sh ./training_data/beat_block_hammer_recovery_pi05 beat_block_hammer_recovery_pi05
```

- **generate 脚本**：`policy/pi05/generate.sh`，内部调用  
  `uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir ... --repo_id ...`

---

## 3. 标注与 Pipeline（关键逻辑）

### 3.1 字段从哪里来

- **instructions / subtasks / frame_idx / phase_info**：来自 processed_data 的 `instructions.json` 等，经 `convert_aloha_data_to_lerobot_robotwin.py` 写入 LeRobot 数据集。
- **Repack**：config L761–775 的 `RepackTransform` 把 LeRobot 的 key 映射到 `observation.*`, `action`, `instructions`, `subtasks`, `frame_idx`, `phase_info`。
- **Progress 标签**：不在磁盘里，在 dataloader 里**动态算**。

### 3.2 丢弃 [MASKED] 帧（drop_masked_recovery_frames）

| 位置 | 作用 |
|------|------|
| **data_loader.py** L65–159 | `_MaskedRecoveryFilteredTransformedDataset`：对每个 sample 先 repack，再调 `_should_drop_masked(repacked)` |
| **data_loader.py** L91–141 | `_should_drop_masked(data)`：若本条有 `[MASKED]` 阶段，算出该阶段结束帧 `masked_end`，**若 `frame_idx < masked_end` 则丢弃**（并重采样其它 index） |
| **data_loader.py** L325–331 | `transform_dataset()` 里当 `data_config.drop_masked_recovery_frames is True` 时用上述 wrapper |

效果：**只保留「masked 阶段结束之后」的帧**（即 recovery + 后续 normal），error attempt 段不参与训练。

### 3.3 子任务 / 高层 prompt（[Correction] / [Subtask]）

| 位置 | 作用 |
|------|------|
| **transforms.py** L455–524（约） | `LoadSubtaskFromInstructions`：用 `frame_idx` 和 `phase_info["checkpoints"]` 得到当前 `phase_idx`，取 `subtasks[idx][phase_idx]` 为 `low_prompt_raw` |
| 同上 | 若 `low_prompt_raw` 含 `[Correction]` 和 `[Subtask]`：拆成 `correction_text` 与 `subtask_text`；**high_prompt = instruction + correction_text**，**low_prompt = subtask_text**（仅子任务参与 subtask loss） |
| 同上 | 若为普通阶段：去掉 tag 后 `low_prompt` 为纯子任务文本；`use_first_instruction=False` 时随机选一条 instruction 变体（L469–472） |

### 3.4 进度标签（recovery 重归一化）

| 位置 | 作用 |
|------|------|
| **transforms.py** L528–590（约） | `ComputeProgressLabel`：若存在含 `[MASKED]` 的 phase，则取所有变体里**最早**的 `masked_end` 为 `start_idx`，**progress = (frame_idx - start_idx) / (total_steps - 1 - start_idx)**，使 recovery 段从 0 到 1 线性增长 |

---

## 4. 训练入口与命令

- **脚本**：`policy/pi05/finetune.sh`（内部调用 `scripts/train.py`）。
- **命令示例**（在 `policy/pi05` 下，使用 config 名 + 实验名 + GPU）：

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
source .venv/bin/activate   # 或 uv 环境
export XDG_CACHE_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache

# 训练 recovery（两卡示例）
bash finetune.sh pi05_aloha_beat_block_hammer_recovery_from15000 recovery_exp_1 0,1
```

- **finetune.sh** 实际执行（见 `finetune.sh` L5–7）：

```bash
export CUDA_VISIBLE_DEVICES=$gpu_use
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $train_config_name --exp-name=$model_name --overwrite
```

即：

```bash
uv run scripts/train.py pi05_aloha_beat_block_hammer_recovery_from15000 --exp-name=recovery_exp_1 --overwrite
```

- **训练前**需要先为该 config 算好 norm stats（否则会报错）：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_beat_block_hammer_recovery_from15000
```

norm stats 会写入该 config 使用的 assets 目录，并在训练/评测时从 checkpoint 的 assets 里读。

---

## 5. 为什么测试时「手臂乱动、成功率 0」可能原因

### 5.1 数据分布过窄（最可疑）

- 训练数据**只有** beat_block_hammer 的 **recovery 轨迹**（含 [MASKED] + [Correction] + 少量 normal）。
- 再丢弃 [MASKED] 段后，**实际参与训练的帧**里：
  - 很大比例是 **「instruction + correction」+「recovery 子任务」**；
  - 纯「正常执行、无 correction」的帧可能很少。
- 测试时是**纯正常指令、无 [Correction]**，和训练分布差异大，容易导致：
  - 策略对「无 correction」的 prompt 行为异常；
  - 或对「正常抓取 / 敲击」的动作分布遗忘 → **手臂乱动、成功率 0**。

**建议排查**：

- 统计 `beat_block_hammer_recovery_pi05` 里，**被保留的帧**中：
  - 带 `[Correction]` 的帧数 vs 纯 normal 的帧数；
  - 若 correction 帧占绝大多数，需要增加「纯正常」beat_block_hammer 数据一起训练，或减少 recovery 占比。

### 5.2 Norm stats 与 action 尺度

- Recovery 数据的 **state/action 分布**（尤其是 recovery 段）可能和「纯正常多任务」的 progress 基座不同。
- 若用 **仅 recovery 数据** 算的 norm_stats 训练，而测试环境或基座期望的尺度不同，可能表现为动作幅度异常（过大/过小/偏移）→ 看起来像「乱动」。

**建议排查**：

- 对比 `pi05_aloha_beat_block_hammer_recovery_from15000` 用的 norm stats（在对应 checkpoint 的 assets 里）与 `pi05_aloha_full_base_multi_task_5_v1_0_progress` 的 norm stats；
- 或尝试用 **progress 基座的 norm stats** 做 recovery 微调（若代码支持从 base config 继承 assets）。

### 5.3 过拟合 / 步数过多

- `num_train_steps=50001`，数据量若不大，容易**过拟合到 recovery 子集**，进一步削弱对「正常执行」的泛化。

**建议**：减小步数或加大数据（尤其是正常 beat_block_hammer 轨迹）。

### 5.4 初始权重与 head

- 从 **progress_exp_1/15000** 继续训，用的是 `CheckpointWeightLoaderWithDefaults`，**新参数**（若有）会随机初始化，其余从 15000 加载。
- 若 15000 的「正常多任务」能力本来就不稳，或和当前 data/config 不匹配，也可能在只喂 recovery 数据后快速退化。

---

## 6. 建议的下一步

1. **确认数据构成**：对 `beat_block_hammer_recovery_pi05` 做简单统计（按 phase / 是否含 [Correction] 统计保留帧数），确认 normal vs recovery 比例。
2. **对比评测**：用 **progress_exp_1/15000**（不训 recovery）在同一任务上跑相同 eval，确认是否「不加 recovery 训练就正常」。
3. **混合数据**：用 **beat_block_hammer 正常数据 + recovery 数据** 一起训一版（例如同一 repo 里既有正常轨迹也有 recovery 轨迹），看手臂是否还乱动。
4. **Norm / 超参**：检查 recovery 训练用的 norm_stats 与 base 是否一致或兼容；必要时减小 `num_train_steps` 做消融。

以上所有代码与命令位置均已在正文中标明，便于你直接打开对应文件与行号排查。
