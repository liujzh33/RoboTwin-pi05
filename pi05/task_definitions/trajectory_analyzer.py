"""
轨迹分析工具：从 HDF5 数据中提取运动特征

提供以下功能：
1. 提取夹爪状态（开/闭）
2. 计算末端执行器运动方向和速度
3. 检测静止点、方向变化点等关键事件
"""

import numpy as np
import h5py
from typing import Tuple, Optional


class TrajectoryAnalyzer:
    """轨迹分析器，用于提取运动特征"""
    
    def __init__(self, gripper_threshold: float = 0.5, velocity_threshold: float = 0.02):
        """
        Args:
            gripper_threshold: 夹爪闭合阈值（0.0=完全闭合, 1.0=完全打开）
            velocity_threshold: 速度阈值，低于此值认为是静止
        """
        self.gripper_threshold = gripper_threshold
        self.velocity_threshold = velocity_threshold
    
    def extract_gripper_states(self, hdf5_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取左右夹爪状态
        
        Args:
            hdf5_data: h5py.File 对象
            
        Returns:
            left_gripper: 左夹爪值数组
            right_gripper: 右夹爪值数组
        """
        # 尝试从不同路径读取
        if 'observations/qpos' in hdf5_data:
            qpos = hdf5_data['observations/qpos'][()]
            # 尝试读取维度信息
            if 'observations/left_arm_dim' in hdf5_data:
                left_arm_dim = int(hdf5_data['observations/left_arm_dim'][0])
                right_arm_dim = int(hdf5_data['observations/right_arm_dim'][0])
                left_gripper_idx = left_arm_dim
                right_gripper_idx = left_arm_dim + 1 + right_arm_dim
            else:
                # 默认假设：14维 = [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
                left_gripper_idx = 6
                right_gripper_idx = 13
        elif 'qpos' in hdf5_data:
            qpos = hdf5_data['qpos'][()]
            left_gripper_idx = 6
            right_gripper_idx = 13
        else:
            raise ValueError("Cannot find qpos data in HDF5 file")
        
        left_gripper = qpos[:, left_gripper_idx]
        right_gripper = qpos[:, right_gripper_idx]
        
        return left_gripper, right_gripper

    def extract_eef_pos(self, hdf5_data) -> Optional[np.ndarray]:
        """
        提取末端执行器位置（如果存在）

        优先使用:
        - observations/eef_pos: [T, 3] or [T, 7]

        如果不存在则返回 None，由上层代码回退到关节近似。
        """
        # 常见路径 1: observations/eef_pos
        if "observations/eef_pos" in hdf5_data:
            return hdf5_data["observations/eef_pos"][()]
        # 常见路径 2: observations/eef_position
        if "observations/eef_position" in hdf5_data:
            return hdf5_data["observations/eef_position"][()]
        # 其它路径可以在这里按需补充
        return None
    
    def compute_velocity(self, qpos: np.ndarray, arm_indices: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        计算关节速度（运动幅度）
        
        Args:
            qpos: 关节位置数据 [T, dim]
            arm_indices: (start_idx, end_idx) 指定要分析的关节范围，None表示使用所有关节
            
        Returns:
            velocity_magnitude: 速度模长数组 [T-1]
        """
        if arm_indices is None:
            # 默认使用前6个关节（左臂）
            arm_indices = (0, 6)
        
        start_idx, end_idx = arm_indices
        arm_qpos = qpos[:, start_idx:end_idx]
        
        # 计算差分（速度）
        joint_velocities = np.diff(arm_qpos, axis=0)
        
        # 计算L2范数（速度模长）
        velocity_magnitude = np.linalg.norm(joint_velocities, axis=1)
        
        # 在开头补0以对齐长度
        velocity_magnitude = np.insert(velocity_magnitude, 0, 0.0)
        
        return velocity_magnitude
    
    def detect_stop_points(self, velocity: np.ndarray) -> np.ndarray:
        """
        检测静止点（速度低于阈值的位置）
        
        Args:
            velocity: 速度数组
            
        Returns:
            is_stopped: 布尔数组，True表示静止
        """
        return velocity < self.velocity_threshold
    
    def detect_grasp_event(self, left_gripper: np.ndarray, right_gripper: np.ndarray) -> Optional[int]:
        """
        检测抓取事件（夹爪闭合的瞬间）
        
        Args:
            left_gripper: 左夹爪值数组
            right_gripper: 右夹爪值数组
            
        Returns:
            grasp_idx: 抓取事件发生的索引，如果未找到返回None
        """
        # 夹爪值越小表示闭合程度越高
        min_gripper = np.minimum(left_gripper, right_gripper)
        
        # 找到从大于阈值变为小于阈值的瞬间（闭合事件）
        is_closed = min_gripper < self.gripper_threshold
        transitions = np.where(np.diff(is_closed.astype(int)) == 1)[0]
        
        if len(transitions) > 0:
            return int(transitions[0])
        return None
    
    def detect_velocity_peaks(self, velocity: np.ndarray, min_peak_height: float = 0.1) -> np.ndarray:
        """
        检测速度峰值（快速运动阶段）
        
        Args:
            velocity: 速度数组
            min_peak_height: 最小峰值高度
            
        Returns:
            peak_indices: 峰值索引数组
        """
        from scipy.signal import find_peaks
        
        try:
            peaks, _ = find_peaks(velocity, height=min_peak_height)
            return peaks
        except ImportError:
            # 如果没有scipy，使用简单方法
            peaks = []
            for i in range(1, len(velocity) - 1):
                if velocity[i] > velocity[i-1] and velocity[i] > velocity[i+1] and velocity[i] > min_peak_height:
                    peaks.append(i)
            return np.array(peaks)
    
    def detect_velocity_valleys(self, velocity: np.ndarray, max_valley_height: float = 0.05) -> np.ndarray:
        """
        检测速度谷值（静止或换向点）
        
        Args:
            velocity: 速度数组
            max_valley_height: 最大谷值高度
            
        Returns:
            valley_indices: 谷值索引数组
        """
        # 反转速度数组，找峰值
        inverted_velocity = -velocity
        valleys = self.detect_velocity_peaks(inverted_velocity, min_peak_height=-max_valley_height)
        return valleys
    
    def analyze_movement_direction(self, qpos: np.ndarray, start_idx: int, end_idx: int, 
                                   arm_indices: Optional[Tuple[int, int]] = None) -> str:
        """
        分析运动方向（简化版：基于关节变化）
        
        Args:
            qpos: 关节位置数据
            start_idx: 起始索引
            end_idx: 结束索引
            arm_indices: 关节范围
            
        Returns:
            direction: 运动方向描述（"Forward", "Backward", "Lift", "Lower", "Hold"）
        """
        if arm_indices is None:
            arm_indices = (0, 6)
        
        start_idx, end_idx = max(0, start_idx), min(len(qpos), end_idx)
        segment = qpos[start_idx:end_idx, arm_indices[0]:arm_indices[1]]
        
        if len(segment) < 2:
            return "Hold"
        
        # 计算总位移
        displacement = segment[-1] - segment[0]
        
        # 简化判断：主要看前3个关节（通常对应X/Y/Z位置）
        if len(displacement) >= 3:
            dx, dy, dz = displacement[0], displacement[1], displacement[2]
            
            # 优先判断Z轴（上下）
            if abs(dz) > 0.05:
                return "Lift" if dz > 0 else "Lower"
            # 然后判断X轴（前后）
            elif abs(dx) > 0.05:
                return "Forward" if dx > 0 else "Backward"
            # 最后判断Y轴（左右）
            elif abs(dy) > 0.05:
                return "Right" if dy > 0 else "Left"
        
        return "Hold"

