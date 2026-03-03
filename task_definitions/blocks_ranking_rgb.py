"""
blocks_ranking_rgb 任务的多阶段切分逻辑 (Threshold = 0.2)
路径: src/openpi/task_definitions/blocks_ranking_rgb.py
"""
import numpy as np
from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer

class BlocksRankingRgbProcessor(BaseTaskProcessor):
    def __init__(self, gripper_threshold: float = 0.5, velocity_threshold: float = 0.02):
        self.analyzer = TrajectoryAnalyzer(gripper_threshold, velocity_threshold)
        self.current_arm_sequence = []
        self.colors = ["red", "green", "blue"]

    def get_phase_checkpoints(self, hdf5_data, active_side: str = None, external_z: np.ndarray = None) -> list[int]:
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)

        # [修改点] 阈值改为 0.2
        close_threshold = 0.2
        
        left_segments = self._find_closed_segments(left_gripper, close_threshold, "left")
        right_segments = self._find_closed_segments(right_gripper, close_threshold, "right")

        all_segments = left_segments + right_segments
        all_segments.sort(key=lambda x: x['start'])

        # 过滤过短片段，保留前3个
        valid_segments = [seg for seg in all_segments if (seg['end'] - seg['start']) > 15]
        target_segments = valid_segments[:3]
        
        self.current_arm_sequence = [seg['arm'] for seg in target_segments]
        
        checkpoints = []
        for seg in target_segments:
            checkpoints.append(seg['start']) 
            checkpoints.append(seg['end'])   

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
            segments.append({'start': int(start_frame), 'end': int(end_frame), 'arm': arm_name})
            curr_start_idx += 1
        return segments

    def get_subtask_descriptions(self) -> list[str]:
        descriptions = []
        for i, _ in enumerate(self.current_arm_sequence):
            color = self.colors[i] if i < len(self.colors) else "next"
            descriptions.append(f"Pick up the {color} block.")
            prev_color = self.colors[i-1] if i > 0 else None
            loc = "on the far left" if i == 0 else f"next to the {prev_color} block"
            descriptions.append(f"Place the {color} block {loc}.")
        descriptions.append("Return to a neutral position.")
        return descriptions

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> list[str]:
        dynamic_descs = self.get_subtask_descriptions()
        while len(dynamic_descs) < num_phases:
            dynamic_descs.append("Complete the task.")
        return dynamic_descs[:num_phases]