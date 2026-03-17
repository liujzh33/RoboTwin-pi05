"""
handover_mic 任务的多阶段切分逻辑

任务：单臂先抓住麦克风，再双臂交接。
3 阶段：
  0) 靠近麦克风
  1) 抓住麦克风并移动到桌面中间
  2) 一臂保持不动，另一臂抓住麦克风并原臂放开

分界点：cp1 = 先闭合的那只臂闭合（靠近结束）；cp2 = 另一只臂闭合（第三阶段开始）。
"""

from typing import List

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


def _find_closed_segments(
    gripper: np.ndarray, threshold: float, min_len: int = 10
) -> List[dict]:
    """夹爪闭合片段"""
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


class HandoverMicProcessor(BaseTaskProcessor):
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

        # 先闭合的臂 = 先抓麦克风；后闭合的臂 = 交接时抓住麦克风
        left_first = left_segs[0]["start"] if left_segs else total_steps
        right_first = right_segs[0]["start"] if right_segs else total_steps

        if left_first <= right_first and left_segs:
            first_seg = left_segs[0]
            second_segs = right_segs
        elif right_segs:
            first_seg = right_segs[0]
            second_segs = left_segs
        else:
            return self.validate_checkpoints([total_steps // 2], total_steps)

        cp1 = first_seg["start"]  # 靠近结束 / 抓住麦克风开始

        # 第三阶段开始 = 另一只臂闭合（交接抓住）
        second_seg = None
        for seg in second_segs:
            if seg["start"] > cp1:
                second_seg = seg
                break
        if second_seg is None:
            cp2 = first_seg["end"]  # 兜底：另一臂未检测到则用原臂打开
        else:
            cp2 = second_seg["start"]

        checkpoints = [cp1, cp2]
        return self.validate_checkpoints(checkpoints, total_steps)

    def get_subtask_descriptions(self) -> List[str]:
        return [
            "Approach the microphone.",
            "Grasp the microphone and move it to the middle of the table.",
            "Keep one arm still; the other arm grasps the microphone and the first arm releases.",
        ]

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> List[str]:
        base = self.get_subtask_descriptions()
        while len(base) < num_phases:
            base.append("Complete the task.")
        return base[:num_phases]
