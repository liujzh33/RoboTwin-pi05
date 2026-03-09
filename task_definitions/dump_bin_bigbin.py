"""
dump_bin_bigbin 任务的多阶段切分逻辑

核心语义（按你描述的行为）：
- 常规 3 阶段（单臂，主要是左臂）：
  1) 靠近容器
  2) 抓住容器
  3) 倾倒容器
- 扩展 6 阶段（双臂协作）：先右臂放中间，再左臂靠近/抓住/倾倒
  1) 右臂靠近容器
  2) 右臂抓住容器
  3) 右臂把容器放到中间
  4) 左臂靠近中间的容器
  5) 左臂抓住容器
  6) 左臂倾倒容器

倾倒起始帧：抓住后夹爪不会打开，无法用“夹爪打开”作为倾倒结束。改为：
  关闭夹爪的那只臂，在夹爪闭合后末端高度 Z 上升到第一个局部最高点，
  作为“倾倒”阶段的起始帧（对应图中 END-EFFECTOR HEIGHT 红线处）。
- 单臂：该臂闭合后第一个 Z 峰 = 倾倒起始。
- 双臂：后动的臂（左臂）闭合后第一个 Z 峰 = 倾倒起始。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


@dataclass
class GripperSegment:
    start: int
    end: int
    arm: str  # "left" or "right"


class DumpBinBigbinProcessor(BaseTaskProcessor):
    def __init__(self, gripper_threshold: float = 0.2, min_segment_len: int = 10):
        """
        Args:
            gripper_threshold: 小于该值视为“夹爪闭合”
            min_segment_len: 认为是有效抓取片段的最小长度（帧数）
        """
        self.analyzer = TrajectoryAnalyzer(gripper_threshold=gripper_threshold)
        self.min_segment_len = min_segment_len

    # 注意：签名需要兼容 process_data_generic.py 中的调用
    def get_phase_checkpoints(
        self,
        hdf5_data,
        active_side: str | None = None,
        external_z: np.ndarray | None = None,
    ) -> List[int]:
        """
        基于左右夹爪的闭合时序，返回阶段切分点：
        - 单臂：3 阶段 → 返回 [cp1, cp2]
        - 双臂：6 阶段 → 返回 [cp1, cp2, cp3, cp4, cp5]
        """
        # 1. 提取左右夹爪轨迹
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)

        # 防御：太短的 episode 直接均匀切 3 段
        if total_steps < 10:
            return self.validate_checkpoints(
                [total_steps // 3, 2 * total_steps // 3], total_steps
            )

        # 2. 查找左右手各自的“闭合片段”
        left_segments = self._find_closed_segments(left_gripper, "left")
        right_segments = self._find_closed_segments(right_gripper, "right")

        all_segments: List[GripperSegment] = left_segments + right_segments
        all_segments.sort(key=lambda s: s.start)

        # 3. 根据是否出现双臂抓取来决定是 3 阶段还是 6 阶段
        arms_used = {s.arm for s in all_segments}

        if not all_segments:
            # 完全没有明显抓取，回退为 3 阶段的均匀切分
            return self.validate_checkpoints(
                [total_steps // 3, 2 * total_steps // 3], total_steps
            )

        # 获取左右臂末端高度 Z（用于“闭合后第一个峰”= 倾倒起始）
        z_left, z_right = self._extract_left_right_z(hdf5_data, total_steps)

        if len(arms_used) == 1:
            # ===== 单臂情况：3 阶段 =====
            main_seg = all_segments[0]
            cp1 = main_seg.start  # 靠近结束（开始夹住）
            z_arm = z_left if main_seg.arm == "left" else z_right
            pour_start = self._first_z_peak_after(z_arm, cp1, total_steps)
            cp2 = pour_start if pour_start is not None else main_seg.end
            checkpoints = [cp1, cp2]
            return self.validate_checkpoints(checkpoints, total_steps)

        # ===== 双臂情况：先右后左，倾倒起始 = 左臂闭合后第一个 Z 峰 =====
        first_left = next((s for s in all_segments if s.arm == "left"), None)
        first_right = next((s for s in all_segments if s.arm == "right"), None)

        if first_left is None or first_right is None:
            main_seg = all_segments[0]
            cp1 = main_seg.start
            z_arm = z_left if main_seg.arm == "left" else z_right
            pour_start = self._first_z_peak_after(z_arm, cp1, total_steps)
            cp2 = pour_start if pour_start is not None else main_seg.end
            return self.validate_checkpoints([cp1, cp2], total_steps)

        if first_right.start <= first_left.start:
            first_arm_seg = first_right
            second_arm_seg = first_left  # 左臂后动，倾倒由左臂做
        else:
            first_arm_seg = first_left
            second_arm_seg = first_right

        # 6 阶段：R-Approach, R-Grasp, R-Place, L-Approach, L-Grasp, L-Pour → 5 个 checkpoint
        cp1 = first_arm_seg.start                                    # 右靠近结束 / 右抓开始
        z_first = z_right if first_arm_seg.arm == "right" else z_left
        right_grasp_end = self._first_z_peak_after(z_first, cp1, total_steps)
        cp2 = right_grasp_end if right_grasp_end is not None else (cp1 + (first_arm_seg.end - cp1) // 2)
        cp3 = first_arm_seg.end                                     # 右放置结束（右夹爪打开）
        cp4 = second_arm_seg.start                                  # 左靠近结束 / 左抓开始
        z_second = z_left if second_arm_seg.arm == "left" else z_right
        pour_start = self._first_z_peak_after(z_second, cp4, total_steps)
        cp5 = pour_start if pour_start is not None else second_arm_seg.end  # 左倾倒开始
        checkpoints = [cp1, cp2, cp3, cp4, cp5]
        return self.validate_checkpoints(checkpoints, total_steps)

    def _find_closed_segments(
        self, gripper: np.ndarray, arm_name: str
    ) -> List[GripperSegment]:
        """从夹爪轨迹中提取“闭合”片段"""
        is_closed = gripper < self.analyzer.gripper_threshold
        diff = np.diff(is_closed.astype(int), prepend=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        segments: List[GripperSegment] = []
        for start_idx in starts:
            # 找到第一个在 start_idx 之后的结束点
            end_candidates = ends[ends > start_idx]
            if len(end_candidates) == 0:
                end_idx = len(gripper) - 1
            else:
                end_idx = int(end_candidates[0])

            if end_idx - start_idx >= self.min_segment_len:
                segments.append(
                    GripperSegment(start=int(start_idx), end=int(end_idx), arm=arm_name)
                )

        return segments

    def _extract_left_right_z(
        self, hdf5_data, total_steps: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从 hdf5 读取左右臂末端高度 Z（优先 endpose，否则 qpos 近似）。"""
        z_left = None
        z_right = None
        if "endpose/left_endpose" in hdf5_data and "endpose/right_endpose" in hdf5_data:
            z_left = hdf5_data["endpose/left_endpose"][()][:total_steps, 2]
            z_right = hdf5_data["endpose/right_endpose"][()][:total_steps, 2]
        key_qpos = "observations/qpos" if "observations/qpos" in hdf5_data else "qpos"
        if key_qpos in hdf5_data:
            qpos = hdf5_data[key_qpos][()]
            if qpos.shape[1] >= 14:
                # 左臂抬升关节约 idx 2，右臂约 idx 9
                if z_left is None:
                    z_left = np.asarray(qpos[:total_steps, 2], dtype=np.float64)
                if z_right is None:
                    z_right = np.asarray(qpos[:total_steps, 9], dtype=np.float64)
        return (z_left, z_right)

    def _first_z_peak_after(
        self, z: np.ndarray | None, start_idx: int, total_steps: int
    ) -> Optional[int]:
        """
        在 start_idx 之后找 Z 的第一个局部最高点（闭合后抬起到第一个峰 = 倾倒起始）。
        要求该峰高于闭合时刻的 Z，避免误取闭合前的抖动。
        """
        if z is None or len(z) < start_idx + 3:
            return None
        z = np.asarray(z[:total_steps], dtype=np.float64)
        z_at_close = z[start_idx]
        for i in range(start_idx + 1, min(total_steps - 1, len(z) - 1)):
            if z[i] >= z[i - 1] and z[i] >= z[i + 1] and z[i] > z_at_close:
                return i
        return None

    def get_subtask_descriptions(self) -> List[str]:
        """
        返回一个“最大”描述列表，主要用于兜底。
        实际流程中更推荐使用 get_subtask_descriptions_for_phases。
        """
        return [
            # 单臂 3 阶段
            "Move the arm close to the container.",
            "Grasp the container firmly with the gripper.",
            "Tilt the container to pour out its contents.",
            # 双臂 6 阶段
            "Use the right arm to approach the container.",
            "Use the right arm to grasp the container.",
            "Place the container in the middle of the workspace.",
            "Move the left arm close to the container in the middle.",
            "Grasp the container with the left gripper.",
            "Tilt the container with the left arm to pour out its contents.",
        ]

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> List[str]:
        """
        根据阶段数量返回对应的子任务描述：
        - num_phases == 3 → 单臂：靠近 / 抓住 / 倾倒
        - num_phases == 6 → 双臂协作：右靠近 / 右抓 / 中间放置 / 左靠近 / 左抓 / 左倾倒
        其它情况做合理裁剪或填充。
        """
        if num_phases == 3:
            return [
                "Move the left arm close to the container.",
                "Grasp the container firmly with the gripper.",
                "Tilt the container to pour out its contents.",
            ]

        if num_phases == 6:
            return [
                "Use the right arm to approach the container.",
                "Use the right arm to grasp the container.",
                "Place the container in the middle of the workspace.",
                "Move the left arm close to the container in the middle.",
                "Grasp the container with the left gripper.",
                "Tilt the container with the left arm to pour out its contents.",
            ]

        # 其它阶段数：退化为简单的 3 阶段模板重复 / 截断
        base = [
            "Move the arm close to the container.",
            "Grasp the container.",
            "Tilt the container to pour.",
        ]
        descs: List[str] = []
        while len(descs) < num_phases:
            for d in base:
                if len(descs) < num_phases:
                    descs.append(d)
                else:
                    break
        return descs[:num_phases]

