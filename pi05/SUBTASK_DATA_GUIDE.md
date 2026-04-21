# PI0.5 子任务数据生成指南

## 📋 概述

本指南说明如何为 PI0.5 模型准备包含子任务信息的训练数据。对于 `beat_block_hammer` 任务，我们将任务自动分解为两个子任务：
1. **阶段1（抓取锤子）**：从开始到其中一个夹爪闭合
2. **阶段2（敲击积木）**：从夹爪闭合到结束

## 🔧 工具脚本

### 1. `detect_gripper_phases.py` - 夹爪状态检测和可视化

**功能：**
- 检测 HDF5 文件中的夹爪开闭状态
- 自动识别抓取阶段结束点
- 可视化夹爪状态和阶段分割

**使用方法：**

```bash
# 检测单个 episode
python scripts/detect_gripper_phases.py processed_data/beat_block_hammer-.../episode_0/episode_0.hdf5

# 检测整个数据集
python scripts/detect_gripper_phases.py processed_data/beat_block_hammer-.../

# 保存可视化图片
python scripts/detect_gripper_phases.py processed_data/beat_block_hammer-.../ --save-dir ./gripper_plots/

# 自定义阈值（默认 0.5）
python scripts/detect_gripper_phases.py processed_data/beat_block_hammer-.../ --threshold 0.4
```

**输出：**
- 控制台输出：阶段统计信息
- 可视化图片（可选）：夹爪状态曲线和阶段分割点

### 2. `add_subtasks_to_data.py` - 为已有数据添加子任务

**功能：**
- 为已处理的 `processed_data` 添加子任务信息
- 更新 `instructions.json`，添加 `subtasks` 和 `phase_info` 字段

**使用方法：**

```bash
# 为整个数据集添加子任务
python scripts/add_subtasks_to_data.py processed_data/beat_block_hammer-.../

# 为单个 episode 添加子任务
python scripts/add_subtasks_to_data.py processed_data/beat_block_hammer-.../episode_0/

# 自定义阈值
python scripts/add_subtasks_to_data.py processed_data/beat_block_hammer-.../ --threshold 0.4
```

**输出：**
- 更新后的 `instructions.json`，包含：
  - `subtasks`: 每个高级任务对应的子任务列表
  - `phase_info`: 阶段检测信息（分割点、步数等）

### 3. `process_data_with_subtasks.py` - 数据处理时自动添加子任务

**功能：**
- 在数据转换时自动检测夹爪状态并添加子任务
- 这是 `process_data.py` 的增强版本

**使用方法：**

```bash
python scripts/process_data_with_subtasks.py beat_block_hammer aloha-agilex_randomized_500-200 200 --gripper-threshold 0.5
```

## 📊 数据格式变化

### 原始格式 (`instructions.json`)

```json
{
  "instructions": [
    "Grab the hammer and beat the block",
    "Pick up the hammer, then strike the block"
  ]
}
```

### 新格式（包含子任务）

```json
{
  "instructions": [
    "Grab the hammer and beat the block",
    "Pick up the hammer, then strike the block"
  ],
  "subtasks": [
    ["Grab the hammer", "Strike the block with the hammer"],
    ["Grab the hammer", "Strike the block with the hammer"]
  ],
  "phase_info": {
    "grasp_end_idx": 45,
    "total_steps": 120,
    "phase1_steps": 45,
    "phase2_steps": 75,
    "threshold": 0.5
  }
}
```

## 🚀 完整工作流程

### 方案 A：为已有数据添加子任务

如果你已经有处理好的 `processed_data`：

```bash
# Step 1: 检测和可视化夹爪状态（可选，用于验证）
python scripts/detect_gripper_phases.py processed_data/beat_block_hammer-.../ --save-dir ./gripper_plots/

# Step 2: 添加子任务信息
python scripts/add_subtasks_to_data.py processed_data/beat_block_hammer-.../
```

### 方案 B：数据处理时自动添加子任务

如果你要从原始数据重新处理：

```bash
# 使用增强版数据处理脚本
python scripts/process_data_with_subtasks.py beat_block_hammer aloha-agilex_randomized_500-200 200
```

## 📝 训练数据加载修改

为了在训练时使用子任务信息，需要修改数据加载代码。参考 `openpi_subtask_generation` 的实现：

1. **修改 `transforms.py`**：添加 `TokenizeHighLowPrompt` transform
2. **修改数据加载器**：读取 `subtasks` 字段并使用 `tokenize_high_low_prompt()`

### 示例代码

```python
from openpi.models.tokenizer import PaligemmaTokenizer

# 加载 instructions.json
with open('episode_0/instructions.json', 'r') as f:
    data = json.load(f)

high_level_prompt = data['instructions'][0]  # 随机选择一个
low_level_prompt = data['subtasks'][0][0]  # 第一个子任务

# Tokenize
tokenizer = PaligemmaTokenizer(max_len=200)
tokens, mask, ar_mask, loss_mask = tokenizer.tokenize_high_low_prompt(
    high_level_prompt, low_level_prompt
)
```

## ⚙️ 参数说明

### `gripper_threshold` (默认: 0.5)

- **含义**：夹爪闭合阈值
- **范围**：0.0 (完全闭合) 到 1.0 (完全打开)
- **作用**：当 `min(left_gripper, right_gripper) < threshold` 时，认为抓取完成
- **调整建议**：
  - 如果检测过早：降低阈值（如 0.3-0.4）
  - 如果检测过晚：提高阈值（如 0.6-0.7）

## 🔍 验证和调试

### 1. 检查阶段检测结果

```bash
# 可视化所有 episode 的夹爪状态
python scripts/detect_gripper_phases.py processed_data/beat_block_hammer-.../ --save-dir ./plots/
```

查看生成的图片，确认阶段分割点是否合理。

### 2. 检查 JSON 文件

```bash
# 查看更新后的 instructions.json
cat processed_data/beat_block_hammer-.../episode_0/instructions.json | python -m json.tool
```

确认 `subtasks` 和 `phase_info` 字段已正确添加。

### 3. 统计信息

脚本会自动输出统计信息：
- 每个 episode 的阶段步数
- 平均阶段长度
- 阶段占比

## ⚠️ 注意事项

1. **夹爪阈值**：根据实际任务调整 `gripper_threshold`，确保阶段分割合理
2. **子任务描述**：当前使用固定描述，可以根据高级任务描述进行更智能的生成
3. **数据一致性**：确保所有 episode 的 `instructions.json` 格式一致
4. **备份数据**：在修改前建议备份原始数据

## 📚 下一步

1. **修改训练代码**：参考 `openpi_subtask_generation` 的实现，修改训练脚本以使用子任务数据
2. **修改模型代码**：使用支持子任务生成的 `pi05.py` 和 `tokenizer.py`
3. **训练模型**：使用包含子任务的数据进行训练

## 🔗 相关文件

- `/mnt/data1/liujingzhi/pi05_subtask_comparison.md` - 代码对比分析
- `/mnt/data1/liujingzhi/openpi_subtask_generation-main` - 参考实现

