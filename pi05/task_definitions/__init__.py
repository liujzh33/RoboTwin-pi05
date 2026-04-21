"""
任务定义模块：为不同任务提供多阶段切分逻辑

每个任务需要实现 BaseTaskProcessor 接口，定义如何从 HDF5 数据中提取切分点
"""

from .base_task import BaseTaskProcessor

__all__ = ['BaseTaskProcessor']

