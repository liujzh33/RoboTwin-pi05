# 多阶段任务定义框架

## 概述

本框架支持为不同任务定义多阶段切分逻辑，从原来的2阶段扩展到任意N阶段。

## 目录结构

```
task_definitions/
├── __init__.py              # 模块初始化
├── base_task.py             # 基础任务处理器接口
├── trajectory_analyzer.py   # 轨迹分析工具（提取运动特征）
├── beat_block_hammer.py    # beat_block_hammer 任务的具体实现
└── README.md               # 本文档
```

## 使用方法

### 1. 为现有任务添加多阶段支持

如果你已经有一个任务（例如 `beat_block_hammer`），可以直接使用：

```bash
# 处理数据，自动检测多阶段
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
python scripts/process_data_generic.py \
    --task_name beat_block_hammer \
    --data_dir processed_data/beat_block_hammer-demo_clean-50
```

### 2. 创建新任务定义

如果要为新的任务创建定义，按以下步骤：

#### 步骤1: 创建任务文件

在 `task_definitions/` 下创建新文件，例如 `my_new_task.py`：

```python
from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer

class MyNewTaskProcessor(BaseTaskProcessor):
    def __init__(self):
        self.analyzer = TrajectoryAnalyzer()
    
    def get_phase_checkpoints(self, hdf5_data) -> list[int]:
        # 实现你的切分逻辑
        # 返回切分点列表，例如 [50, 100] 表示3个阶段
        pass
    
    def get_subtask_descriptions(self) -> list[str]:
        # 返回每个阶段的子任务描述
        # 长度必须 = len(checkpoints) + 1
        pass
```

#### 步骤2: 分析数据特征

使用分析工具理解数据分布：

```bash
python scripts/analyze_trajectory.py processed_data/my_task/episode_0/episode_0.hdf5 --save analysis.png
```

查看生成的图片，确定切分规则。

#### 步骤3: 实现切分逻辑

在 `get_phase_checkpoints` 中实现你的切分算法，可以使用 `TrajectoryAnalyzer` 提供的工具：

- `extract_gripper_states()`: 提取夹爪状态
- `compute_velocity()`: 计算运动速度
- `detect_grasp_event()`: 检测抓取事件
- `detect_stop_points()`: 检测静止点
- `detect_velocity_peaks/valleys()`: 检测速度峰值/谷值

#### 步骤4: 处理数据

```bash
python scripts/process_data_generic.py \
    --task_name my_new_task \
    --data_dir processed_data/my_task
```

## 核心接口说明

### BaseTaskProcessor

所有任务处理器必须继承 `BaseTaskProcessor` 并实现两个方法：

1. **`get_phase_checkpoints(hdf5_data) -> list[int]`**
   - 输入: HDF5 文件对象
   - 输出: 切分点列表，例如 `[50, 100]` 表示：
     - Phase 0: 0-50帧
     - Phase 1: 51-100帧
     - Phase 2: 101-End帧

2. **`get_subtask_descriptions() -> list[str]`**
   - 输出: 子任务描述列表
   - 长度必须 = `len(checkpoints) + 1`

### TrajectoryAnalyzer

提供轨迹分析工具：

- **`extract_gripper_states(hdf5_data)`**: 提取左右夹爪状态
- **`compute_velocity(qpos, arm_indices)`**: 计算关节速度
- **`detect_grasp_event(left_gripper, right_gripper)`**: 检测抓取事件
- **`detect_stop_points(velocity)`**: 检测静止点
- **`analyze_movement_direction(...)`**: 分析运动方向

## 数据格式

处理后的 `instructions.json` 格式：

```json
{
  "instructions": ["High-level instruction 1", "High-level instruction 2", ...],
  "subtasks": [
    ["Subtask 1 for Phase 0", "Subtask 2 for Phase 1", ...],
    ["Subtask 1 for Phase 0", "Subtask 2 for Phase 1", ...],
    ...
  ],
  "phase_info": {
    "checkpoints": [50, 100],
    "total_steps": 200,
    "num_phases": 3
  }
}
```

## 训练流程

1. **数据处理**:
   ```bash
   mkdir processed_data && mkdir training_data
   bash process_data_pi0.sh ${task_name} ${task_config} ${expert_data_num}
   python scripts/process_data_generic.py --task_name ${task_name} --data_dir processed_data/...
   ```

2. **生成 LeRobot 数据集**:
   ```bash
   export XDG_CACHE_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache
   bash generate.sh ${hdf5_path} ${repo_id}
   ```

3. **训练**:
   ```bash
   bash finetune.sh pi05_aloha_full_base demo_randomized 4,5
   ```

## 示例：beat_block_hammer

`beat_block_hammer` 任务从2阶段扩展到4-5阶段：

- **Phase 0**: Reach for the hammer (伸手)
- **Phase 1**: Align with the hammer handle (对准)
- **Phase 2**: Grasp the hammer (抓取)
- **Phase 3**: Lift the hammer up (抬起)
- **Phase 4**: Strike the block with the hammer (敲击)

切分逻辑基于：
1. 夹爪闭合事件（核心切分点）
2. 抓取前的接近阶段（固定窗口）
3. 抓取后的抬起阶段（固定窗口）

## 注意事项

1. **向后兼容**: 代码同时支持旧的 `grasp_end_idx` 格式和新的 `checkpoints` 格式
2. **验证**: `validate_checkpoints` 方法会自动清理无效的切分点
3. **灵活性**: 可以根据任务特点自定义切分逻辑，不限于夹爪状态

