# 多阶段训练框架 - 快速开始

## 已实现的功能

✅ **多阶段任务定义框架** (`task_definitions/`)
- 基础接口 `BaseTaskProcessor`
- 轨迹分析工具 `TrajectoryAnalyzer`
- `beat_block_hammer` 任务实现（支持2-5阶段）

✅ **通用数据处理脚本** (`scripts/process_data_generic.py`)
- 支持任意数量的阶段切分
- 自动从 HDF5 数据中提取切分点
- 生成包含 `phase_info` 的 `instructions.json`

✅ **训练代码更新** (`src/openpi/transforms.py`)
- `LoadSubtaskFromInstructions` 支持多阶段
- 向后兼容旧的 `grasp_end_idx` 格式
- 自动根据 `checkpoints` 列表选择正确的子任务

✅ **可视化分析工具** (`scripts/analyze_trajectory.py`)
- 可视化夹爪状态、运动速度
- 自动建议阶段切分点

## 使用流程

### 步骤1: 分析数据特征（可选但推荐）

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
python scripts/analyze_trajectory.py \
    processed_data/beat_block_hammer-demo_clean-50/episode_0/episode_0.hdf5 \
    --save analysis.png
```

查看生成的图片，理解数据分布。

### 步骤2: 处理数据（添加多阶段信息）

```bash
# 使用通用脚本处理数据
python scripts/process_data_generic.py \
    --task_name beat_block_hammer \
    --data_dir processed_data/beat_block_hammer-demo_clean-50
```

这会自动：
- 检测夹爪闭合事件
- 切分多个阶段（默认4-5阶段）
- 更新每个 episode 的 `instructions.json`，添加 `phase_info`

### 步骤3: 生成 LeRobot 数据集

```bash
export XDG_CACHE_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache
bash generate.sh ./training_data/demo_clean/ demo_clean_repo
```

### 步骤4: 训练

```bash
bash finetune.sh pi05_aloha_full_base demo_randomized 4,5
```

训练代码会自动使用多阶段信息。

## 创建新任务定义

### 示例：创建 `stack_cube` 任务

1. **创建文件** `task_definitions/stack_cube.py`:

```python
from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer

class StackCubeProcessor(BaseTaskProcessor):
    def __init__(self):
        self.analyzer = TrajectoryAnalyzer()
    
    def get_phase_checkpoints(self, hdf5_data) -> list[int]:
        # 实现你的切分逻辑
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        grasp_idx = self.analyzer.detect_grasp_event(left_gripper, right_gripper)
        
        if grasp_idx is None:
            return []
        
        # 你的切分逻辑
        checkpoints = [grasp_idx]
        return checkpoints
    
    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Pick up the cube",
            "Place the cube on top"
        ]
```

2. **处理数据**:

```bash
python scripts/process_data_generic.py \
    --task_name stack_cube \
    --data_dir processed_data/stack_cube-demo_clean-50
```

## 关键改进

### 从2阶段到多阶段

**之前（硬编码）**:
- 只能切分为2阶段：抓取前 / 抓取后
- 切分逻辑写死在 `add_subtasks_to_data.py`
- 无法扩展

**现在（通用框架）**:
- 支持任意N阶段（2-5阶段或更多）
- 每个任务有独立的切分逻辑文件
- 易于扩展和维护

### beat_block_hammer 任务示例

**2阶段** (旧):
- Phase 0: Grab the hammer
- Phase 1: Strike the block

**5阶段** (新):
- Phase 0: Reach for the hammer
- Phase 1: Align with the hammer handle
- Phase 2: Grasp the hammer
- Phase 3: Lift the hammer up
- Phase 4: Strike the block with the hammer

## 数据格式

处理后的 `instructions.json` 包含：

```json
{
  "instructions": ["High-level instruction 1", ...],
  "subtasks": [
    ["Subtask for Phase 0", "Subtask for Phase 1", ...],
    ...
  ],
  "phase_info": {
    "checkpoints": [15, 50, 70],
    "total_steps": 200,
    "num_phases": 4
  }
}
```

训练时，`LoadSubtaskFromInstructions` 会根据 `frame_idx` 和 `checkpoints` 自动选择正确的子任务。

## 注意事项

1. **向后兼容**: 代码同时支持旧的 `grasp_end_idx` 格式
2. **验证**: 切分点会自动验证和清理
3. **灵活性**: 可以根据任务特点自定义切分逻辑

## 故障排除

### 问题: 找不到任务定义模块

**解决**: 确保 `task_definitions/{task_name}.py` 存在，且类名正确。

### 问题: 切分点数量与子任务描述不匹配

**解决**: 确保 `get_subtask_descriptions()` 返回的描述数量 = `len(checkpoints) + 1`

### 问题: 训练时找不到 phase_info

**解决**: 确保使用 `process_data_generic.py` 处理数据，而不是旧的脚本。

