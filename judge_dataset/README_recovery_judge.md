# Recovery Judge 数据集（Qwen2-VL Judge 微调）

## 模型输入 / 输出（约定）

### 输入（User）

- **6 张图**（与 pi05 三视角对齐，顺序固定）  
  - 前 3 张：**Start 状态** — `cam_high`, `cam_left_wrist`, `cam_right_wrist`（error 为 error_attempt 起始帧，success 为 episode 首帧）  
  - 后 3 张：**End 状态** — 同上三视角（error 为 error_attempt 结束帧，success 为某 checkpoint 帧）
- **文本**：高层任务指令（如 “Beat the block with the hammer.”）+ 固定提示（看首尾状态、按要求格式输出）

一条 user 内容在 ShareGPT 里形如：

```
<image><image><image><image><image><image>

Task: {high_instruction}

Look at the start state (first 3 images) and the current state (last 3 images). What went wrong? Output exactly in this format:
[error_type]: <one of: grasp_position_offset, grasp_orientation_mismatch, premature_close, grasp_slip, None>
[reflection]: <short reflection>
[correction]: <short correction>
```

### 输出（Assistant，三段式）

模型必须按行输出且仅包含以下三行（便于下游正则解析）：

```
[error_type]: {grasp_position_offset | grasp_orientation_mismatch | premature_close | grasp_slip | None}
[reflection]: {一句反思/诊断}
[correction]: {一句纠错动作 或 "[Correction] None"}
```

- **Error 样本**：`error_type` 为四类之一，`reflection` / `correction` 从对应词库随机。  
- **Success 样本**：`error_type` 为 `None`，`reflection` 从 SUCCESS_VARIANTS 随机，`correction` 固定为 `[Correction] None`。

---

## Success 样本数量

- 默认 **每个数据源（clean / rand）最多 50 条**，即 success 总数 ≤ 100（`--max_per_source 50`）。  
- 需要更多时可传 `--max_per_source 200` 等；需要全部时传 `--max_per_source 0` 或很大数值表示不限制。

---

## 生成与合并

见项目内 `scripts/build_judge_dataset_beat_block_hammer_full.sh` 或 `scripts/extract_judge_*.py` 的用法说明。

---

## 在 LLaMA-Factory 中训练

### 1. 数据集已注册

已在 **LLaMA-Factory** 的 `data/dataset_info.json` 中注册数据集（注意你的 LLaMA-Factory 路径是 `LlamaFactory`）：

- **数据集名称**：`recovery_judge_beat_block_hammer`
- **训练用文件**：`recovery_judge_dataset_merged.json`（198 条，含 success + error，已打乱）
- **格式**：ShareGPT，`conversations` + `images`，角色为 `user` / `assistant`

若你用的是别的 LLaMA-Factory 目录，在该目录的 `data/dataset_info.json` 最外层添加：

```json
"recovery_judge_beat_block_hammer": {
  "file_name": "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/judge_dataset/recovery_judge_beat_block_hammer/recovery_judge_dataset_merged.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "images": "images"
  },
  "tags": {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

### 2. 启动 WebUI 训练

```bash
cd /mnt/data1/liujingzhi/LlamaFactory
conda activate llama_factory   # 或你实际使用的环境名
llamafactory-cli webui
```

浏览器打开后：

- **模型**：选 **Qwen2-VL-7B**（或你本地的 Qwen2-VL 名称）
- **模型路径**：填 `Qwen2-VL-7B` 的本地路径，例如 `/mnt/data1/liujingzhi/Qwen-VL-7B`（若目录名不同请按实际路径填写）
- **数据集**：选 **recovery_judge_beat_block_hammer**
- 配置好 LoRA/全量、batch、epoch 等后，点击 **开始（Start）** 即可训练

### 3. 命令行训练（可选）

若用 YAML + 命令行，在 `dataset` 中写上 `recovery_judge_beat_block_hammer` 即可，例如：

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml  # 复制并改 dataset、model 等
```

确保 YAML 里 `dataset: recovery_judge_beat_block_hammer`，且 `model_name_or_path` 指向 Qwen2-VL 权重目录。
