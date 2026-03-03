"""
beat_block_hammer 任务的多阶段切分逻辑 (修复版)

修复说明：
1. 增加 active_side 参数，支持右手操作。
2. 动态选择 gripper 数据（左或右）。
3. 动态选择 Z 轴数据（左或右的 endpose/qpos）。
"""

import numpy as np
import h5py
from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class BeatBlockHammerProcessor(BaseTaskProcessor):
    """
    beat_block_hammer 任务处理器
    
    使用夹爪状态和运动速度来切分多个阶段
    """
    
    def __init__(self, 
                 gripper_threshold: float = 0.5,
                 velocity_threshold: float = 0.02):
        self.analyzer = TrajectoryAnalyzer(gripper_threshold, velocity_threshold)
    
    def get_phase_checkpoints(self, hdf5_data, active_side: str = None, external_z: np.ndarray = None) -> list[int]:
        """
        基于直观物理规则的四阶段划分，支持左右手自动切换。

        Args:
            hdf5_data: h5py.File 对象
            active_side: 'left' 或 'right'。如果不传，尝试自动检测。
            external_z: [新增参数] 如果外部已经读取了精准的 Z 轴数据（如 Raw Endpose），
                        直接传入，不再从 hdf5 读取。这样可以确保计算和可视化使用相同的数据源。

        Returns:
            [close_start_idx, close_end_idx, z_max_idx]
        """
        # 1. 提取双臂夹爪状态
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)

        # 2. 确定活动臂 (Active Side)
        if active_side is None:
            # 如果没有指定，通过方差/极差自动判断哪只手在动
            left_range = np.max(left_gripper) - np.min(left_gripper)
            right_range = np.max(right_gripper) - np.min(right_gripper)
            active_side = "left" if left_range >= right_range else "right"
            print(f"[BeatBlockHammerProcessor] Auto-detected active side: {active_side}")

        # 3. 选择主信号数据
        if active_side == "left":
            gripper = left_gripper
            z_key = "endpose/left_endpose"
            qpos_z_idx = 2  # 左臂 Z 轴通常是第 3 个关节 (0,1,2)
        else:
            gripper = right_gripper
            z_key = "endpose/right_endpose"
            # 注意：如果是一维 qpos (14维)，右臂通常从第 7 个开始，Z 是 7+2=9
            # 这里需要根据具体机器人的 qpos 结构调整，通常 Aloha 是 [left_6, right_6, ...]
            qpos_z_idx = 2 + 7  # 假设右臂从 index 7 开始 (0-6左, 7-13右)

        # 4. 找到所有“闭合”帧
        # 阈值：大于 open_threshold 认为是打开，小于 close_threshold 认为是闭合
        open_threshold = 0.9
        close_threshold = 0.1
        
        closed_indices = np.where(gripper < close_threshold)[0]
        if len(closed_indices) == 0:
            print(f"Warning: No closed {active_side} gripper state detected, returning midpoint")
            return [total_steps // 2]

        # 4.1 夹取结束点：第一次进入闭合状态的帧
        close_end_idx = int(closed_indices[0])

        # 4.2 夹取开始点：在 close_end_idx 之前，最后一次仍然“打开”的帧
        open_before_close = np.where(gripper[:close_end_idx] > open_threshold)[0]
        if len(open_before_close) == 0:
            close_start_idx = 0
        else:
            close_start_idx = int(open_before_close[-1])

        # 5. 提取 Z 轴数据 (针对 Active Side)
        # 优先级 0: 使用外部传入的精准 Z 数据（如 Raw Endpose）
        z = None
        
        if external_z is not None:
            # 确保长度对齐（截断到当前 episode 长度）
            if len(external_z) >= total_steps:
                z = external_z[:total_steps]
                print(f"[BeatBlockHammerProcessor] Using external Z data (length={len(external_z)}, truncated to {total_steps})")
            else:
                # 如果 external_z 比 total_steps 短，这是异常情况，打印警告
                print(f"Warning: external_z length ({len(external_z)}) < total_steps ({total_steps}), using shorter length")
                z = external_z
                # 更新 total_steps 以匹配 external_z 的长度
                total_steps = len(z)
        
        # 优先级 1: 尝试从 hdf5 读取 endpose (笛卡尔空间坐标)
        if z is None:
            if z_key in hdf5_data:
                endpose = hdf5_data[z_key][()]
                if endpose.shape[0] >= total_steps:
                    z = endpose[:total_steps, 2] # 取 Z 轴
                else:
                    z = endpose[:, 2]
                    print(f"[BeatBlockHammerProcessor] Using endpose from hdf5 (length={len(z)})")

        # 优先级 2: 降级使用 qpos (关节空间) 近似
        if z is None:
            # 尝试读取 qpos
            qpos_key = "observations/qpos" if "observations/qpos" in hdf5_data else "qpos"
            if qpos_key in hdf5_data:
                qpos = hdf5_data[qpos_key][()]
                # 检查维度是否足够
                if qpos.shape[1] > qpos_z_idx:
                    z = qpos[:total_steps, qpos_z_idx]
                    print(f"[BeatBlockHammerProcessor] Using qpos idx {qpos_z_idx} as Z approximation (WARNING: This is joint space, not end-effector position!)")

        if z is None:
            print(f"Warning: No Z data found for {active_side} arm; returning gripper splits only")
            return self.validate_checkpoints([close_start_idx, close_end_idx], total_steps)

        # 6. 通过 Z 轴变化精确划分
        
        # 6.1 夹取结束校准：
        # 抓取物体后，Z轴必须有实质性抬升才算 Phase 1 结束
        # 这样可以过滤掉“刚闭合但还在桌面上摩擦”的阶段
        z_min_idx = close_end_idx
        rise_threshold = 0.02 # 2cm 抬升阈值
        
        # 确保索引不越界
        safe_start_idx = min(close_end_idx, total_steps - 1)
        base_z = z[safe_start_idx]
        
        for i in range(safe_start_idx + 1, total_steps):
            if z[i] - base_z > rise_threshold:
                z_min_idx = i
                break

        # 6.2 Boundary 2：在 Boundary 1 (z_min_idx) 之后找 Z 值的最大值
        # 这表示抬起完成，准备向下敲击的时刻
        boundary1_idx = z_min_idx  # Boundary 1 是夹取结束、开始抬起的点
        
        # 从 Boundary 1 之后开始搜索（包含 boundary1_idx + 1 到末尾）
        search_start = boundary1_idx + 1
        search_end = total_steps
        
        if search_end <= search_start or search_end - search_start < 3:
            # 如果 Boundary 1 之后没有足够的数据，只返回两个切分点
            checkpoints = [close_start_idx, z_min_idx]
        else:
            # 在 Boundary 1 之后的所有 Z 值中找最大值
            post_boundary1_z = z[search_start:search_end]
            rel_max_idx = int(np.argmax(post_boundary1_z))
            z_max_idx = search_start + rel_max_idx
            
            # 返回三个切分点：[夹取开始, 夹取结束(抬起开始), 抬起完成(最高点)]
            checkpoints = [close_start_idx, z_min_idx, z_max_idx]

        return self.validate_checkpoints(checkpoints, total_steps)

    
    def get_subtask_descriptions(self) -> list[str]:
        """
        返回每个阶段的子任务描述
        
        注意：描述数量 = len(checkpoints) + 1
        这里返回最大可能的描述列表，实际使用时根据checkpoints数量选择
        """
        return [
            "Reach for the hammer",              # Phase 0
            "Align with the hammer handle",      # Phase 1
            "Grasp the hammer",                  # Phase 2
            "Lift the hammer up",                # Phase 3
            "Strike the block with the hammer"   # Phase 4
        ]
    
    def get_subtask_descriptions_for_phases(self, num_phases: int) -> list[str]:
        """
        根据阶段数量返回对应的子任务描述
        
        Args:
            num_phases: 阶段数量（2-5）
            
        Returns:
            子任务描述列表
        """
        all_descriptions = self.get_subtask_descriptions()
        
        if num_phases == 2:
            # 2阶段：简化为抓取和敲击
            return [all_descriptions[2], all_descriptions[4]]
        elif num_phases == 3:
            # 3阶段：抓取前、抓取、敲击
            return [all_descriptions[1], all_descriptions[2], all_descriptions[4]]
        elif num_phases == 4:
            # 4阶段：接近、抓取、抬起、敲击
            return [all_descriptions[0], all_descriptions[2], all_descriptions[3], all_descriptions[4]]
        elif num_phases == 5:
            # 5阶段：完整流程
            return all_descriptions
        else:
            # 默认返回所有描述
            return all_descriptions[:num_phases]

