"""
beat_block_hammer 任务的多阶段切分逻辑（夹爪 + 速度 + Z 轴）

- Stage 1: 0 至 T1 — Move the gripper above the hammer.
  物理：(寻物) 夹爪保持张开（> 0.95）。纯空间移动。结束于夹爪开始收缩（< 0.95）的瞬间。
- Stage 2: T1 至 T2 — Close the gripper to grasp the hammer.
  物理：(抓握) 包含夹爪收缩 (0.95 → 0.05) + 抓紧后的原地发呆期。结束于速度打破 0（> 0.015）的起步瞬间。
- Stage 3: T2 至 T3 — Move the hammer above the red block.
  物理：(搬运与瞄准) 夹爪死死锁定（< 0.05）。手持锤子进行 XY 平面移动。结束于准备敲击前，Z 轴开始明显急剧下降的瞬间。
- Stage 4: T3 至 末尾 — Move the hammer down to hit the red block.
  物理：(敲击) 夹爪锁定（< 0.05）。Z 轴全速向下俯冲执行敲击，直到当前 Episode 数据结束。

T1 由夹爪定义（首次从 >0.95 向闭合过渡），T2 由速度定义（>0.015 起步），T3 由 Z 轴定义（明显下降起点）。
"""

from typing import List, Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class BeatBlockHammerProcessor(BaseTaskProcessor):
    def __init__(
        self,
        gripper_threshold: float = 0.5,
        velocity_threshold: float = 0.015,
        z_drop_threshold: float = 0.003,
    ):
        self.analyzer = TrajectoryAnalyzer(gripper_threshold, velocity_threshold)
        self.velocity_start_threshold = 0.015  # 速度打破 0 的起步阈值（用于 T2 切分）
        self.velocity_break_threshold = 0.015  # 同上，供 analyze_trajectory 脚本绘图使用
        self.z_drop_threshold = z_drop_threshold  # Z 明显下降的阈值（单步）

    def get_phase_checkpoints(
        self,
        hdf5_data,
        active_side: str = None,
        external_z: Optional[np.ndarray] = None,
        external_eef_xyz: Optional[np.ndarray] = None,
        external_z_left: Optional[np.ndarray] = None,
        external_z_right: Optional[np.ndarray] = None,
        external_eef_xyz_left: Optional[np.ndarray] = None,
        external_eef_xyz_right: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        - T1: 夹爪从 open(>0.95) 首次向闭合过渡（OPEN→MID）。
        - T2: T1 之后，速度打破 0（>0.015）的起步瞬间（抓握后开始移动）。
        - T3: T2 之后，Z 轴高度开始明显下降的瞬间（敲击起点）；
              若无 Z 数据则回退为 T2 之后一定偏移。
        """
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)

        # 确定活动臂
        if active_side is None:
            left_range = np.max(left_gripper) - np.min(left_gripper)
            right_range = np.max(right_gripper) - np.min(right_gripper)
            active_side = "left" if left_range >= right_range else "right"

        gripper = left_gripper if active_side == "left" else right_gripper

        open_th = 0.95
        closed_th = 0.05
        state = np.zeros_like(gripper, dtype=int)
        state[gripper < closed_th] = 2
        state[(gripper >= closed_th) & (gripper <= open_th)] = 1
        # state == 0: open (>0.95)

        checkpoints: List[int] = []

        # T1: 第一个 OPEN(0) -> MID(1) 或 OPEN(0)->CLOSED(2)（夹爪穿过 0.95 开始闭合）
        for i in range(len(state) - 1):
            if state[i] == 0 and (state[i + 1] == 1 or state[i + 1] == 2):
                checkpoints.append(i + 1)
                break
        if not checkpoints:
            first_mid_or_closed = np.where(state >= 1)[0]
            if len(first_mid_or_closed) > 0 and 0 < first_mid_or_closed[0] < total_steps:
                checkpoints.append(int(first_mid_or_closed[0]))
            else:
                checkpoints.append(max(1, total_steps // 4))

        t1 = checkpoints[0]

        # T2: T1 之后，速度首次 > 0.015 的起步瞬间
        qpos_key = "observations/qpos" if "observations/qpos" in hdf5_data else "qpos"
        arm_indices = (0, 6) if active_side == "left" else (7, 13)
        if qpos_key in hdf5_data:
            qpos = hdf5_data[qpos_key][()]
            qpos = qpos[:total_steps]
            velocity = self.analyzer.compute_velocity(qpos, arm_indices)
            if len(velocity) > t1 + 1:
                t2 = None
                for t in range(t1, len(velocity)):
                    if velocity[t] > self.velocity_start_threshold:
                        t2 = t
                        break
                if t2 is not None and t2 > t1:
                    checkpoints.append(t2)
                else:
                    _append_t2_from_gripper(state, t1, total_steps, checkpoints)
            else:
                _append_t2_from_gripper(state, t1, total_steps, checkpoints)
        else:
            _append_t2_from_gripper(state, t1, total_steps, checkpoints)

        t2 = checkpoints[-1]

        # T3: T2 之后 Z 开始明显下降的瞬间
        z = self._get_z_series(
            hdf5_data, total_steps, active_side,
            external_z, external_eef_xyz,
            external_z_left, external_z_right,
            external_eef_xyz_left, external_eef_xyz_right,
        )

        if z is not None and len(z) > t2 + 2:
            dZ = np.diff(z)
            t3 = None
            for t in range(t2, min(len(dZ), len(z) - 1)):
                if dZ[t] <= -self.z_drop_threshold:
                    t3 = t + 1
                    break
            if t3 is not None and t3 > t2:
                checkpoints.append(t3)
            else:
                _append_t3_fallback(t2, total_steps, checkpoints)
        else:
            _append_t3_fallback(t2, total_steps, checkpoints)

        return self.validate_checkpoints(sorted(checkpoints), total_steps)

    def _get_z_series(
        self,
        hdf5_data,
        total_steps: int,
        active_side: str,
        external_z: Optional[np.ndarray],
        external_eef_xyz: Optional[np.ndarray],
        external_z_left: Optional[np.ndarray] = None,
        external_z_right: Optional[np.ndarray] = None,
        external_eef_xyz_left: Optional[np.ndarray] = None,
        external_eef_xyz_right: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        # 优先使用双臂数据中活动臂的 Z（process_data_generic_v1 传入）
        if active_side == "left" and external_eef_xyz_left is not None and external_eef_xyz_left.shape[0] > 0:
            z = np.asarray(external_eef_xyz_left[:, 2])
        elif active_side == "right" and external_eef_xyz_right is not None and external_eef_xyz_right.shape[0] > 0:
            z = np.asarray(external_eef_xyz_right[:, 2])
        elif active_side == "left" and external_z_left is not None and len(external_z_left) > 0:
            z = np.asarray(external_z_left)
        elif active_side == "right" and external_z_right is not None and len(external_z_right) > 0:
            z = np.asarray(external_z_right)
        elif external_eef_xyz is not None and external_eef_xyz.shape[0] > 0 and external_eef_xyz.shape[1] >= 3:
            z = np.asarray(external_eef_xyz[:, 2])
        elif external_z is not None and len(external_z) > 0:
            z = np.asarray(external_z)
        else:
            eef = self.analyzer.extract_eef_pos(hdf5_data)
            if eef is not None and eef.ndim == 2 and eef.shape[1] >= 3:
                z = eef[:total_steps, 2]
            else:
                z_key = "endpose/left_endpose" if active_side == "left" else "endpose/right_endpose"
                if z_key in hdf5_data:
                    endpose = hdf5_data[z_key][()]
                    if endpose.shape[0] >= total_steps:
                        z = endpose[:total_steps, 2]
                    else:
                        z = endpose[:, 2]
                else:
                    qpos_key = "observations/qpos" if "observations/qpos" in hdf5_data else "qpos"
                    if qpos_key in hdf5_data:
                        qpos = hdf5_data[qpos_key][()]
                        qpos_z_idx = 2 if active_side == "left" else 9
                        if qpos.shape[1] > qpos_z_idx:
                            z = qpos[:total_steps, qpos_z_idx]
                        else:
                            z = None
                    else:
                        z = None
        if z is not None:
            z = z[:total_steps]
        return z

    def get_subtask_descriptions(self) -> List[str]:
        return [
            "Move the gripper above the hammer.",
            "Close the gripper to grasp the hammer.",
            "Move the hammer above the red block.",
            "Move the hammer down to hit the red block.",
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
    t2_fallback = min(t1 + max(10, (total_steps - t1) // 3), total_steps - 1)
    if t2_fallback > t1:
        checkpoints.append(t2_fallback)


def _append_t3_fallback(t2: int, total_steps: int, checkpoints: List[int]) -> None:
    """回退：T3 取 T2 之后约 2/3 处或末尾前。"""
    t3_fallback = min(t2 + max(5, (total_steps - t2) * 2 // 3), total_steps - 1)
    if t3_fallback > t2:
        checkpoints.append(t3_fallback)
