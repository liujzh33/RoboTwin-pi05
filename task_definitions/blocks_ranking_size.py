"""
blocks_ranking_size 任务的多阶段切分逻辑
按积木尺寸排序（小/中/大），阶段划分与 blocks_ranking_rgb 类似，基于夹爪闭合片段。
"""
import numpy as np
from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class BlocksRankingSizeProcessor(BaseTaskProcessor):
    def __init__(self, gripper_threshold: float = 0.5, velocity_threshold: float = 0.02):
        self.analyzer = TrajectoryAnalyzer(gripper_threshold, velocity_threshold)
        self.current_arm_sequence = []
        self.sizes = ["small", "medium", "large"]

    def get_phase_checkpoints(self, hdf5_data, active_side: str = None, external_z: np.ndarray = None) -> list[int]:
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)

        close_threshold = 0.2

        left_segments = self._find_closed_segments(left_gripper, close_threshold, "left")
        right_segments = self._find_closed_segments(right_gripper, close_threshold, "right")

        all_segments = left_segments + right_segments
        all_segments.sort(key=lambda x: x["start"])

        valid_segments = [seg for seg in all_segments if (seg["end"] - seg["start"]) > 15]
        target_segments = valid_segments[:3]

        self.current_arm_sequence = [seg["arm"] for seg in target_segments]

        checkpoints = []
        for seg in target_segments:
            checkpoints.append(seg["start"])
            checkpoints.append(seg["end"])

        return self.validate_checkpoints(sorted(list(set(checkpoints))), total_steps)

    def _find_closed_segments(self, gripper_data: np.ndarray, threshold: float, arm_name: str) -> list[dict]:
        segments = []
        is_closed = gripper_data < threshold
        diff = np.diff(is_closed.astype(int), prepend=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        curr_start_idx = 0
        while curr_start_idx < len(starts):
            start_frame = starts[curr_start_idx]
            end_frame = next((e for e in ends if e > start_frame), len(gripper_data) - 1)
            segments.append({"start": int(start_frame), "end": int(end_frame), "arm": arm_name})
            curr_start_idx += 1
        return segments

    def get_subtask_descriptions(self) -> list[str]:
        # Order: smallest first → far right, then medium → middle, then largest → far left.
        place_locations = ["on the far right", "in the middle", "on the far left"]
        descriptions = []
        for i, _ in enumerate(self.current_arm_sequence):
            size = self.sizes[i] if i < len(self.sizes) else "next"
            descriptions.append(f"Pick up the {size} block.")
            loc = place_locations[i] if i < len(place_locations) else "in place"
            descriptions.append(f"Place the {size} block {loc}.")
        descriptions.append("Return to a neutral position.")
        return descriptions

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> list[str]:
        dynamic_descs = self.get_subtask_descriptions()
        while len(dynamic_descs) < num_phases:
            dynamic_descs.append("Complete the task.")
        return dynamic_descs[:num_phases]
