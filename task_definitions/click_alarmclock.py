"""
click_alarmclock 任务的多阶段切分逻辑（夹爪 + Z 轴高度）

- Stage 1: 0 至 T1 — Move the gripper above the alarm clock.
  物理：夹爪保持张开（>0.95），XY 平移至目标上方。
- Stage 2: T1 至 T2 — Close the gripper to prepare for clicking.
  物理：夹爪开始收缩握拳（穿过 0.95），闭合动作 + 按压前悬停，Z 保持高位平稳。
- Stage 3: T2 至 末尾 — Click the alarm clock and return.
  物理：Z 轴高度开始明显下降的瞬间；下压触碰闹钟后抬升复位，夹爪保持闭合。

T1 由夹爪定义（首次从 >0.95 向闭合过渡），T2 由 Z 轴定义（明显下降起点）。
"""

from typing import List, Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class ClickAlarmclockProcessor(BaseTaskProcessor):
    def __init__(
        self,
        gripper_threshold: float = 0.5,
        velocity_threshold: float = 0.02,
        z_drop_threshold: float = 0.003,
    ):
        self.analyzer = TrajectoryAnalyzer(gripper_threshold, velocity_threshold)
        self.z_drop_threshold = z_drop_threshold  # Z 明显下降的阈值（单步）

    def get_phase_checkpoints(
        self,
        hdf5_data,
        active_side: str = None,
        external_z: Optional[np.ndarray] = None,
        external_eef_xyz: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        - T1: 夹爪从 open(>0.95) 首次向闭合过渡（OPEN→MID）。
        - T2: T1 之后，Z 轴高度开始明显下降的瞬间（用于“下压点击”起点）；
              若无 Z 数据则回退为夹爪到达 closed(<0.05) 的时刻。
        """
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)
        gripper = np.minimum(left_gripper, right_gripper)

        open_th = 0.95
        closed_th = 0.05
        state = np.zeros_like(gripper, dtype=int)
        state[gripper < closed_th] = 2
        state[(gripper >= closed_th) & (gripper <= open_th)] = 1

        checkpoints: List[int] = []

        # T1: 第一个 OPEN(0) -> MID(1)（夹爪穿过 0.95 开始闭合）
        for i in range(len(state) - 1):
            if state[i] == 0 and state[i + 1] == 1:
                checkpoints.append(i + 1)
                break
        if not checkpoints:
            first_closed = np.where(state == 2)[0]
            if len(first_closed) > 0 and 0 < first_closed[0] < total_steps:
                checkpoints.append(int(first_closed[0]))
            else:
                checkpoints.append(max(1, total_steps // 3))

        t1 = checkpoints[0]

        # Z 轴：优先 external_eef_xyz[:, 2]，否则 external_z，否则从 hdf5 近似
        z = self._get_z_series(hdf5_data, total_steps, external_z, external_eef_xyz)

        if z is not None and len(z) > t1 + 2:
            # T2: T1 之后 Z 开始明显下降的瞬间（单步下降超过阈值）
            dZ = np.diff(z)
            t2 = None
            for t in range(t1, min(len(dZ), len(z) - 1)):
                if dZ[t] <= -self.z_drop_threshold:
                    t2 = t + 1  # 索引对齐到 z 的帧
                    break
            if t2 is not None and t2 > t1:
                checkpoints.append(t2)
            else:
                _append_t2_from_gripper(state, t1, total_steps, checkpoints)
        else:
            _append_t2_from_gripper(state, t1, total_steps, checkpoints)

        return self.validate_checkpoints(sorted(checkpoints), total_steps)

    def _get_z_series(
        self,
        hdf5_data,
        total_steps: int,
        external_z: Optional[np.ndarray],
        external_eef_xyz: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if external_eef_xyz is not None and external_eef_xyz.shape[0] > 0 and external_eef_xyz.shape[1] >= 3:
            z = np.asarray(external_eef_xyz[:, 2])
        elif external_z is not None and len(external_z) > 0:
            z = np.asarray(external_z)
        else:
            eef = self.analyzer.extract_eef_pos(hdf5_data)
            if eef is not None and eef.ndim == 2 and eef.shape[1] >= 3:
                z = eef[:total_steps, 2]
            elif "observations/qpos" in hdf5_data or "qpos" in hdf5_data:
                qpos = hdf5_data.get("observations/qpos", hdf5_data.get("qpos"))[()]
                qpos = qpos[:total_steps]
                if qpos.shape[1] >= 14:
                    z = (qpos[:, 2] + qpos[:, 9]) / 2.0  # 双臂 Z 近似
                else:
                    z = None
            else:
                z = None
        if z is not None:
            z = z[:total_steps]
        return z

    def get_subtask_descriptions(self) -> List[str]:
        return [
            "Move the gripper above the alarm clock.",
            "Close the gripper to prepare for clicking.",
            "Click the alarm clock and return.",
        ]

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> List[str]:
        base = self.get_subtask_descriptions()
        if num_phases <= len(base):
            return base[:num_phases]
        while len(base) < num_phases:
            base.append("Complete the task.")
        return base[:num_phases]


def _append_t2_from_gripper(
    state: np.ndarray, t1: int, total_steps: int, checkpoints: List[int]
) -> None:
    """回退：T2 取 T1 之后第一次夹爪到达 CLOSED(2) 的帧。"""
    for i in range(t1, len(state)):
        if state[i] == 2:
            checkpoints.append(i)
            return
    t2_fallback = min(t1 + max(10, (total_steps - t1) // 2), total_steps - 1)
    if t2_fallback > t1:
        checkpoints.append(t2_fallback)
