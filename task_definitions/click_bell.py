"""
click_bell 任务的多阶段切分逻辑
任务：先靠近物体（到达上方）→ 夹爪闭合 → 按铃。无“打开夹爪”状态，仅 2 阶段。
- 阶段 0：靠近物体（到达目标上方），直至夹爪完全关闭
- 阶段 1：点击物体（click）
"""
import numpy as np
from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class ClickBellProcessor(BaseTaskProcessor):
    def __init__(self, gripper_threshold: float = 0.5, velocity_threshold: float = 0.02):
        self.analyzer = TrajectoryAnalyzer(gripper_threshold, velocity_threshold)

    def get_phase_checkpoints(self, hdf5_data, active_side: str = None, external_z: np.ndarray = None) -> list[int]:
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)

        close_threshold = 0.2
        left_closed = left_gripper < close_threshold
        right_closed = right_gripper < close_threshold
        any_closed = np.logical_or(left_closed, right_closed)
        diff = np.diff(any_closed.astype(int), prepend=0)
        closes = np.where(diff == 1)[0]

        # 只有 2 阶段：夹爪首次完全关闭为分界点，之后不再打开
        checkpoints = []
        if len(closes) > 0:
            first_close = int(closes[0])
            if 0 < first_close < total_steps:
                checkpoints.append(first_close)
        else:
            mid = total_steps // 2
            if 0 < mid < total_steps:
                checkpoints.append(mid)

        return self.validate_checkpoints(checkpoints, total_steps)

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the bell (reach above it)",
            "Click the bell",
        ]

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> list[str]:
        base = self.get_subtask_descriptions()
        while len(base) < num_phases:
            base.append("Complete the task.")
        return base[:num_phases]
