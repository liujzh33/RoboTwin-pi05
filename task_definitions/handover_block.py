"""
handover_block 任务的多阶段切分逻辑

任务：左臂抓红色木块 → 移动到中间 → 交接给右臂 → 右臂放到蓝色块上。
4 阶段：
  0) 左臂靠近红色木块
  1) 抓住红色木块并移动到中间
  2) 右臂抓住红块，左臂放开
  3) 右臂放置红色木块到蓝色块上

分界点：左闭 → 右闭 → 左开（共 3 个 checkpoint）。
"""

from typing import List, Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


def _find_closed_segments(
    gripper: np.ndarray, threshold: float, min_len: int = 10
) -> List[dict]:
    """夹爪闭合片段：[(start, end), ...]"""
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


class HandoverBlockProcessor(BaseTaskProcessor):
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

        if not left_segs:
            # 无左闭合则退化为单 checkpoint
            if right_segs:
                return self.validate_checkpoints([right_segs[0]["start"]], total_steps)
            return self.validate_checkpoints([total_steps // 2], total_steps)

        left_seg = left_segs[0]
        cp1 = left_seg["start"]   # 左靠近结束 / 左抓并移动到中间开始
        cp3 = left_seg["end"]     # 左放开（交接结束）

        # 右臂在左臂已闭合之后才闭合的那段（交接）
        right_seg = None
        for r in right_segs:
            if r["start"] > cp1:
                right_seg = r
                break
        if right_seg is None:
            right_seg = right_segs[0] if right_segs else {"start": cp1 + 1, "end": total_steps - 1}

        cp2 = right_seg["start"]  # 右抓住（交接开始）

        # 4 阶段只需 3 个 checkpoint：左闭、右闭、左开
        checkpoints = [cp1, cp2, cp3]
        return self.validate_checkpoints(checkpoints, total_steps)

    def get_subtask_descriptions(self) -> List[str]:
        return [
            "Approach the red block with the left arm.",
            "Grasp the red block and move it to the middle.",
            "Grasp the red block with the right gripper and release with the left.",
            "Place the red block on the blue block with the right arm.",
        ]

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> List[str]:
        base = self.get_subtask_descriptions()
        while len(base) < num_phases:
            base.append("Complete the task.")
        return base[:num_phases]
