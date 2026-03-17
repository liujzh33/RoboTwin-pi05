"""
blocks_ranking_size_v1 任务的多阶段切分逻辑（夹爪 + 速度）

- 13 阶段：T1 夹爪开始收缩；T2 速度打破 0 起步；T3 夹爪开始松开；T4 速度打破 0 起步。
- Move 阶段为纯空间移动，Close/Open 阶段含闭合/张开+原地发呆，结束于速度起步瞬间。
"""

from typing import List, Dict, Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class BlocksRankingSizeV1Processor(BaseTaskProcessor):
    """
    夹爪 + 速度精细边界：T1/T3 由夹爪定义，T2/T4 由“速度打破 0”的起步瞬间定义。
    """

    def __init__(
        self,
        gripper_threshold: float = 0.5,
        velocity_threshold: float = 0.02,
        close_threshold: float = 0.2,
        min_closed_len: int = 15,
    ):
        self.analyzer = TrajectoryAnalyzer(
            gripper_threshold=gripper_threshold,
            velocity_threshold=velocity_threshold,
        )
        self.current_arm_sequence: List[str] = []
        self.sizes = ["small", "medium", "large"]
        self.close_threshold = close_threshold
        self.min_closed_len = min_closed_len

    def get_phase_checkpoints(
        self,
        hdf5_data,
        active_side: str = None,
        external_z: Optional[np.ndarray] = None,
        external_eef_xyz: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        T1: 夹爪开始收缩 (<0.95)。T2: 抓紧后速度打破 0 的起步瞬间。T3: 夹爪开始松开 (>0.05)。T4: 放开后速度打破 0 的起步瞬间。
        """
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)
        gripper = np.minimum(left_gripper, right_gripper)

        qpos = self._get_qpos(hdf5_data, total_steps)
        vel_left = self.analyzer.compute_velocity(qpos, arm_indices=(0, 6))
        vel_right = self.analyzer.compute_velocity(qpos, arm_indices=(7, 13))
        velocity = np.maximum(vel_left, vel_right)

        open_th = 0.95
        closed_th = 0.05
        state = np.zeros_like(gripper, dtype=int)
        state[gripper < closed_th] = 2
        state[(gripper >= closed_th) & (gripper <= open_th)] = 1

        v_th = self.analyzer.velocity_threshold

        def find_cycles(
            state_arr: np.ndarray, vel: np.ndarray, max_cycles: int = 3
        ) -> List[Dict]:
            cycles: List[Dict] = []
            n = len(state_arr)
            idx = 0

            while idx < n and len(cycles) < max_cycles:
                while idx < n and state_arr[idx] != 0:
                    idx += 1
                if idx >= n:
                    break

                T1 = None
                while idx < n - 1:
                    if state_arr[idx] == 0 and state_arr[idx + 1] == 1:
                        T1 = idx + 1
                        idx = T1
                        break
                    idx += 1
                if T1 is None:
                    break

                t_closed = idx
                while t_closed < n and state_arr[t_closed] != 2:
                    t_closed += 1
                if t_closed >= n:
                    break
                T2 = t_closed
                for i in range(t_closed, n):
                    if vel[i] > v_th:
                        T2 = i
                        break

                T3 = None
                idx = t_closed
                while idx < n - 1:
                    if state_arr[idx] == 2 and state_arr[idx + 1] == 1:
                        T3 = idx + 1
                        idx = T3
                        break
                    idx += 1
                if T3 is None:
                    break

                t_open = idx
                while t_open < n and state_arr[t_open] != 0:
                    t_open += 1
                if t_open >= n:
                    T4 = n - 1
                else:
                    T4 = t_open
                    for i in range(t_open, n):
                        if vel[i] > v_th:
                            T4 = i
                            break

                cycles.append(
                    {"T1": int(T1), "T2": int(T2), "T3": int(T3), "T4": int(T4)}
                )
                idx = T4 + 1

            return cycles

        cycles = find_cycles(state, velocity, max_cycles=3)

        if len(cycles) < 1:
            left_segments = self._find_closed_segments(
                left_gripper, self.close_threshold, "left"
            )
            right_segments = self._find_closed_segments(
                right_gripper, self.close_threshold, "right"
            )
            all_segments = left_segments + right_segments
            all_segments.sort(key=lambda x: x["start"])
            valid_segments = [
                seg
                for seg in all_segments
                if (seg["end"] - seg["start"]) > self.min_closed_len
            ]
            target_segments = valid_segments[:3]
            checkpoints: List[int] = []
            for seg in target_segments:
                checkpoints.append(seg["start"])
                checkpoints.append(seg["end"])
            return self.validate_checkpoints(
                sorted(list(set(checkpoints))), total_steps
            )

        cycles = cycles[:3]
        checkpoints = []
        for cyc in cycles:
            checkpoints.extend([cyc["T1"], cyc["T2"], cyc["T3"], cyc["T4"]])

        return self.validate_checkpoints(
            sorted(list(set(checkpoints))), total_steps
        )

    def _get_qpos(self, hdf5_data, total_steps: int) -> np.ndarray:
        if "observations/qpos" in hdf5_data:
            qpos = hdf5_data["observations/qpos"][()]
        elif "qpos" in hdf5_data:
            qpos = hdf5_data["qpos"][()]
        else:
            raise ValueError("Cannot find qpos for velocity.")
        return qpos[:total_steps]

    def _find_closed_segments(
        self, gripper_data: np.ndarray, threshold: float, arm_name: str
    ) -> List[Dict]:
        segments: List[Dict] = []
        is_closed = gripper_data < threshold
        diff = np.diff(is_closed.astype(int), prepend=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        curr_start_idx = 0
        while curr_start_idx < len(starts):
            start_frame = starts[curr_start_idx]
            end_frame = next((e for e in ends if e > start_frame), len(gripper_data) - 1)
            segments.append(
                {"start": int(start_frame), "end": int(end_frame), "arm": arm_name}
            )
            curr_start_idx += 1
        return segments

    def get_subtask_descriptions(self) -> List[str]:
        """
        固定 13 条阶段描述（small→far right, medium→middle, large→far left）。
        """
        return [
            "Move the gripper above the small block.",
            "Close the gripper to grasp the small block.",
            "Move the gripper to the far right target position while holding the small block.",
            "Open the gripper to release the small block.",
            "Move the gripper above the medium block.",
            "Close the gripper to grasp the medium block.",
            "Move the gripper to the middle target position while holding the medium block.",
            "Open the gripper to release the medium block.",
            "Move the gripper above the large block.",
            "Close the gripper to grasp the large block.",
            "Move the gripper to the far left target position while holding the large block.",
            "Open the gripper to release the large block.",
            "Return to a neutral position.",
        ]

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> List[str]:
        base_descs = self.get_subtask_descriptions()
        if num_phases <= len(base_descs):
            return base_descs[:num_phases]
        extra = ["Return to a neutral position."] * (num_phases - len(base_descs))
        return base_descs + extra
