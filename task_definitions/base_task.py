"""
基础任务处理器接口

所有任务定义都需要继承这个基类并实现两个方法：
1. get_phase_checkpoints: 从 HDF5 数据中提取阶段切分点
2. get_subtask_descriptions: 返回每个阶段的子任务描述
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseTaskProcessor(ABC):
    """
    基础任务处理器抽象类
    
    子类需要实现：
    - get_phase_checkpoints: 返回切分点列表，例如 [50, 100] 表示：
      Phase 0: 0-50帧
      Phase 1: 51-100帧  
      Phase 2: 101-End帧
    - get_subtask_descriptions: 返回子任务描述列表，长度 = len(checkpoints) + 1
    """
    
    @abstractmethod
    def get_phase_checkpoints(self, hdf5_data) -> list[int]:
        """
        从 HDF5 数据中提取阶段切分点
        
        Args:
            hdf5_data: h5py.File 对象或包含数据的字典
            
        Returns:
            checkpoints: 切分点列表，例如 [50, 100] 表示3个阶段
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_subtask_descriptions(self) -> list[str]:
        """
        返回每个阶段的子任务描述
        
        Returns:
            descriptions: 子任务描述列表，长度必须 = len(checkpoints) + 1
            例如 checkpoints=[50, 100]，则返回3个描述
        """
        raise NotImplementedError
    
    def validate_checkpoints(self, checkpoints: list[int], total_steps: int) -> list[int]:
        """
        验证并清理切分点，确保合法性
        
        Args:
            checkpoints: 原始切分点列表
            total_steps: 总步数
            
        Returns:
            清理后的切分点列表
        """
        # 过滤无效值
        valid = [int(cp) for cp in checkpoints if 0 < cp < total_steps]
        # 排序去重
        valid = sorted(list(set(valid)))
        return valid

