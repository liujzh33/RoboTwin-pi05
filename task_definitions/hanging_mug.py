"""
hanging_mug 任务的多阶段切分逻辑

任务：一臂抓马克杯 → 放到桌面中间 → 另一臂抓杯沿 → 悬挂到架子上。
5 阶段：
  0) 靠近马克杯
  1) 抓住马克杯
  2) 放置到桌面中间
  3) 另一臂抓住马克杯边缘
  4) 悬挂马克杯

分界点：先动臂闭 → 先动臂开 → 后动臂闭 → 后动臂开（4 个 checkpoint）。
"""

from typing import List

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


def _find_closed_segments(
    gripper: np.ndarray, threshold: float, min_len: int = 10
) -> List[dict]:
    is_closed = gripper < threshold
    diff = np.diff(is_closed.astype(int), prepend=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    segments = []
    for start_idx in starts:
        end_candidates = ends[ends > start_idx]
        end_idx = int(end_candidates[0]) if len(end_candidates) > 0 else len(gripper) - 1
        if end_idx - start_idx >= min_len:
            segments.append({"start": int(start_idx), "end": int(end_idx)})
    return segments


class HangingMugProcessor(BaseTaskProcessor):
    def __init__(self, gripper_threshold: float = 0.2):
        self.analyzer = TrajectoryAnalyzer(gripper_threshold=gripper_threshold)
        self._min_segment_len = 10

    def get_phase_checkpoints(
        self, hdf5_data, active_side: str = None, external_z: np.ndarray = None
    ) -> List[int]:
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)
        th = getattr(self.analyzer, "gripper_threshold", 0.2)

        left_segs = _find_closed_segments(left_gripper, th, self._min_segment_len)
        right_segs = _find_closed_segments(right_gripper, th, self._min_segment_len)

        left_first = left_segs[0]["start"] if left_segs else total_steps
        right_first = right_segs[0]["start"] if right_segs else total_steps

        if left_first <= right_first and left_segs:
            first_segs = left_segs
            second_segs = right_segs
        elif right_segs:
            first_segs = right_segs
            second_segs = left_segs
        else:
            return self.validate_checkpoints([total_steps // 2], total_steps)

        first_seg = first_segs[0]
        cp1 = first_seg["start"]  # 靠近结束 / 抓住马克杯开始
        cp2 = first_seg["end"]    # 放置到桌面中间结束（先动臂放开）

        # 后动臂在先动臂放开之后闭合（抓杯沿）
        second_seg = None
        for seg in second_segs:
            if seg["start"] > cp2:
                second_seg = seg
                break
        if second_seg is None:
            cp3 = min(cp2 + 1, total_steps - 1)
            cp4 = total_steps - 1
        else:
            cp3 = second_seg["start"]  # 另一臂抓住杯沿开始
            cp4 = second_seg["end"]    # 悬挂结束（后动臂放开）

        checkpoints = [cp1, cp2, cp3, cp4]
        return self.validate_checkpoints(checkpoints, total_steps)

    def get_subtask_descriptions(self) -> List[str]:
        return [
            "Approach the mug.",
            "Grasp the mug.",
            "Place the mug in the middle of the table.",
            "The other arm grasps the rim of the mug.",
            "Hang the mug on the rack.",
        ]

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> List[str]:
        base = self.get_subtask_descriptions()
        while len(base) < num_phases:
            base.append("Complete the task.")
        return base[:num_phases]
