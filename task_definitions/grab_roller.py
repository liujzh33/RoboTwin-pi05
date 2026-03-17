"""
grab_roller 任务的多阶段切分逻辑

任务：双臂靠近容器 → 双臂抓住容器。共 2 阶段。
- 阶段 0：双臂靠近容器（approach）
- 阶段 1：双臂抓住容器（grasp）

分界点：任一手夹爪首次闭合的帧（与 click_alarmclock 类似，用“任一闭合”即可）。
"""

import numpy as np
from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class GrabRollerProcessor(BaseTaskProcessor):
    def __init__(self, gripper_threshold: float = 0.2):
        self.analyzer = TrajectoryAnalyzer(gripper_threshold=gripper_threshold)

    def get_phase_checkpoints(
        self, hdf5_data, active_side: str = None, external_z: np.ndarray = None
    ) -> list:
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)

        close_threshold = getattr(
            self.analyzer, "gripper_threshold", 0.2
        )
        left_closed = left_gripper < close_threshold
        right_closed = right_gripper < close_threshold
        any_closed = np.logical_or(left_closed, right_closed)
        diff = np.diff(any_closed.astype(int), prepend=0)
        closes = np.where(diff == 1)[0]

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

    def get_subtask_descriptions(self) -> list:
        return [
            "Approach the container with both arms.",
            "Grasp the container with both grippers.",
        ]

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> list:
        base = self.get_subtask_descriptions()
        while len(base) < num_phases:
            base.append("Complete the task.")
        return base[:num_phases]
