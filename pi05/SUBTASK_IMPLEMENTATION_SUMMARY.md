# PI0.5 子任务实现总结

## 🎯 目标

为 `beat_block_hammer` 任务实现子任务分解功能，将任务自动分为两个阶段：
1. **阶段1（抓取锤子）**：从开始到其中一个夹爪闭合
2. **阶段2（敲击积木）**：从夹爪闭合到结束

## 📁 创建的文件

### 1. 检测和可视化脚本

- **`scripts/detect_gripper_phases.py`**
  - 功能：检测 HDF5 文件中的夹爪开闭状态
  - 输出：可视化图片和阶段统计信息
  - 用途：验证阶段分割是否合理

### 2. 数据增强脚本

- **`scripts/add_subtasks_to_data.py`**
  - 功能：为已处理的 `processed_data` 添加子任务信息
  - 输入：已有的 `processed_data` 目录
  - 输出：更新后的 `instructions.json`（包含 `subtasks` 和 `phase_info`）

- **`scripts/process_data_with_subtasks.py`**
  - 功能：在数据转换时自动添加子任务
  - 输入：原始 RoboTwin 数据
  - 输出：包含子任务信息的 `processed_data`

### 3. 文档

- **`SUBTASK_DATA_GUIDE.md`**：详细使用指南
- **`SUBTASK_IMPLEMENTATION_SUMMARY.md`**：本文档

## 🚀 使用步骤

### Step 1: 检测夹爪状态（验证阶段分割）

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05

# 检测单个 episode
python scripts/detect_gripper_phases.py \
    processed_data/beat_block_hammer-aloha-agilex_randomized_500-200/episode_0/episode_0.hdf5

# 检测整个数据集并保存可视化
python scripts/detect_gripper_phases.py \
    processed_data/beat_block_hammer-aloha-agilex_randomized_500-200/ \
    --save-dir ./gripper_plots/
```

**输出：**
- 控制台：阶段统计信息
- 图片：`gripper_plots/episode_X_gripper_phases.png`

### Step 2: 为已有数据添加子任务

```bash
# 为整个数据集添加子任务信息
python scripts/add_subtasks_to_data.py \
    processed_data/beat_block_hammer-aloha-agilex_randomized_500-200/
```

**结果：**
- 每个 `episode_X/instructions.json` 会添加：
  - `subtasks`: 子任务列表
  - `phase_info`: 阶段检测信息

### Step 3: 验证数据格式

```bash
# 查看更新后的 JSON
cat processed_data/beat_block_hammer-.../episode_0/instructions.json | python -m json.tool
```

**期望输出：**
```json
{
  "instructions": [...],
  "subtasks": [
    ["Grab the hammer", "Strike the block with the hammer"],
    ...
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

## 🔧 数据格式变化

### 原始格式

```json
{
  "instructions": ["Grab the hammer and beat the block"]
}
```

### 新格式（包含子任务）

```json
{
  "instructions": ["Grab the hammer and beat the block"],
  "subtasks": [["Grab the hammer", "Strike the block with the hammer"]],
  "phase_info": {
    "grasp_end_idx": 45,
    "total_steps": 120,
    "phase1_steps": 45,
    "phase2_steps": 75,
    "threshold": 0.5
  }
}
```

## 📊 阶段检测原理

### 检测方法

1. **提取夹爪值**：从 `qpos` 中提取左右夹爪的开闭值
2. **计算最小值**：`min_gripper = min(left_gripper, right_gripper)`
3. **检测闭合**：当 `min_gripper < threshold` 时，认为抓取完成
4. **分割点**：第一个满足条件的索引即为 `grasp_end_idx`

### 参数调整

- **`threshold`** (默认: 0.5)
  - 值越小：需要更紧的抓取才触发
  - 值越大：更早触发分割
  - 建议范围：0.3 - 0.7

## 🔄 完整工作流程

### 方案 A：为已有数据添加子任务（推荐）

```bash
# 1. 检测和可视化（验证）
python scripts/detect_gripper_phases.py \
    processed_data/beat_block_hammer-.../ \
    --save-dir ./gripper_plots/ \
    --threshold 0.5

# 2. 添加子任务信息
python scripts/add_subtasks_to_data.py \
    processed_data/beat_block_hammer-.../ \
    --threshold 0.5

# 3. 验证结果
cat processed_data/beat_block_hammer-.../episode_0/instructions.json | python -m json.tool
```

### 方案 B：重新处理数据（从原始数据开始）

```bash
# 使用增强版数据处理脚本
python scripts/process_data_with_subtasks.py \
    beat_block_hammer \
    aloha-agilex_randomized_500-200 \
    200 \
    --gripper-threshold 0.5
```

## 🎓 下一步：训练模型

### 1. 修改模型代码

参考 `/mnt/data1/liujingzhi/openpi_subtask_generation-main` 的实现：

- **复制关键文件**：
  - `src/openpi/models/pi05.py` → 支持子任务生成
  - `src/openpi/models/tokenizer.py` → 添加 `tokenize_high_low_prompt()`
  - `src/openpi/models/gemma_05.py` → 支持 KV cache

### 2. 修改数据加载

在数据加载时读取 `subtasks` 字段：

```python
# 读取 instructions.json
with open('episode_0/instructions.json', 'r') as f:
    data = json.load(f)

high_level = data['instructions'][0]
low_level = data['subtasks'][0][0]  # 第一个子任务

# 使用 tokenize_high_low_prompt
tokenizer = PaligemmaTokenizer(max_len=200)
tokens, mask, ar_mask, loss_mask = tokenizer.tokenize_high_low_prompt(
    high_level, low_level
)
```

### 3. 训练配置

修改训练配置以使用子任务数据，参考 `openpi_subtask_generation` 的训练代码。

## ⚠️ 注意事项

1. **阈值调整**：根据实际任务调整 `gripper_threshold`，确保阶段分割合理
2. **子任务描述**：当前使用固定描述，可以根据高级任务进行更智能的生成
3. **数据备份**：在修改前建议备份原始数据
4. **一致性**：确保所有 episode 的格式一致

## 📈 验证检查清单

- [ ] 夹爪状态可视化正常
- [ ] 阶段分割点合理（查看可视化图片）
- [ ] `instructions.json` 包含 `subtasks` 字段
- [ ] `instructions.json` 包含 `phase_info` 字段
- [ ] 所有 episode 的格式一致
- [ ] 阶段统计信息合理（Phase 1 和 Phase 2 的步数比例）

## 🔗 相关资源

- **代码对比分析**：`/mnt/data1/liujingzhi/pi05_subtask_comparison.md`
- **参考实现**：`/mnt/data1/liujingzhi/openpi_subtask_generation-main`
- **使用指南**：`SUBTASK_DATA_GUIDE.md`

## 💡 常见问题

### Q: 阶段分割点不合理怎么办？

A: 调整 `--threshold` 参数：
- 分割过早：降低阈值（如 0.3-0.4）
- 分割过晚：提高阈值（如 0.6-0.7）

### Q: 如何自定义子任务描述？

A: 修改 `generate_subtask_descriptions()` 函数，可以根据高级任务描述进行更智能的生成。

### Q: 数据格式不兼容怎么办？

A: 确保使用正确的脚本版本，并检查 JSON 格式是否符合预期。

## 📝 总结

通过以上步骤，你可以：

1. ✅ 自动检测夹爪状态并分割阶段
2. ✅ 为数据添加子任务信息
3. ✅ 可视化验证阶段分割
4. ✅ 准备用于子任务训练的数据格式

接下来需要修改模型代码和训练脚本，参考 `openpi_subtask_generation` 的实现。

