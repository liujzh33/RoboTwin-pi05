#!/usr/bin/env python3
"""
通用的多阶段数据处理脚本 (增强版 - 完全复刻 analyze_trajectory.py 的 Raw Z 读取逻辑)
"""

import os
import sys
import h5py
import json
import argparse
import importlib
import re
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task_definitions.base_task import BaseTaskProcessor

def get_raw_z_data_from_reference(episode_dir: Path, active_side: str):
    """
    [关键修改] 完全参考 analyze_trajectory.py 的方式读取 Raw Z 数据
    """
    # 1. 自动推断 raw path (逻辑同 reference)
    episode_num = None
    match = re.search(r"episode_(\d+)", str(episode_dir.name))
    if match:
        episode_num = match.group(1)
    
    if episode_num is None:
        return None

    # 硬编码路径 (同 reference)
    raw_episode_path = Path(
        "/mnt/data1/liujingzhi/dataset/beat_block_hammer/"
        f"aloha-agilex_randomized_500/data/episode{episode_num}.hdf5"
    )

    z_values = None

    # 2. 读取数据 (逻辑同 reference)
    if raw_episode_path.exists():
        try:
            with h5py.File(raw_episode_path, "r") as raw_f:
                # 检查 key 是否存在
                if "endpose/left_endpose" in raw_f and "endpose/right_endpose" in raw_f:
                    # 读取 raw endpose
                    left_endpose = raw_f["endpose/left_endpose"][()]
                    right_endpose = raw_f["endpose/right_endpose"][()]
                    
                    # 根据活动侧选择对应末端 Z 轴
                    if active_side == "left":
                        z_raw = left_endpose[:, 2]
                        # print(f"  [Info] Using Raw Left Z from {raw_episode_path.name}")
                    else:
                        z_raw = right_endpose[:, 2]
                        # print(f"  [Info] Using Raw Right Z from {raw_episode_path.name}")
                    
                    z_values = z_raw
                else:
                    print(f"  [Warn] Raw file found but missing 'endpose' keys: {raw_episode_path}")
        except Exception as e:
            print(f"  [Error] Failed to read raw file {raw_episode_path}: {e}")
    else:
        # print(f"  [Warn] Raw file not found: {raw_episode_path}")
        pass

    return z_values


def process_episode_generic(episode_dir: Path, task_processor: BaseTaskProcessor) -> bool:
    """
    通用的单集处理函数
    """
    # 查找 HDF5 文件
    hdf5_files = list(episode_dir.glob("episode_*.hdf5"))
    if len(hdf5_files) == 0:
        print(f"Warning: No HDF5 file found in {episode_dir}")
        return False
    
    hdf5_path = hdf5_files[0]
    
    # 读取或创建 instructions.json
    instructions_path = episode_dir / "instructions.json"
    
    # 读取现有的 instructions（如果存在）
    if instructions_path.exists():
        with open(instructions_path, 'r') as f:
            data = json.load(f)
    else:
        # 创建默认结构
        data = {
            "instructions": ["Default instruction"],
            "subtasks": []
        }
    
    # 从 HDF5 提取切分点
    try:
        with h5py.File(hdf5_path, 'r') as f:
            total_steps = f['action'].shape[0] if 'action' in f else f['observations/qpos'].shape[0]

            # ======================================================
            # [步骤 1] 确定 Active Side (为了正确读取 Raw Z)
            # ======================================================
            active_side = "left" # 默认兜底
            if hasattr(task_processor, 'analyzer') and hasattr(task_processor.analyzer, 'extract_gripper_states'):
                # 使用 analyzer 的逻辑提取夹爪数据
                left_gripper, right_gripper = task_processor.analyzer.extract_gripper_states(f)
                
                # 计算极差 (同 reference)
                left_delta = float(left_gripper.max() - left_gripper.min())
                right_delta = float(right_gripper.max() - right_gripper.min())
                if left_delta >= right_delta:
                    active_side = "left"
                else:
                    active_side = "right"
            
            # ======================================================
            # [步骤 2] 获取 Raw Z 数据 (完全参考 analyze_trajectory.py)
            # ======================================================
            external_z = get_raw_z_data_from_reference(episode_dir, active_side)
            
            # ======================================================
            # [步骤 3] 计算 Checkpoints
            # ======================================================
            # 将 Raw Z 传入处理函数
            checkpoints = task_processor.get_phase_checkpoints(f, active_side=active_side, external_z=external_z)

    except Exception as e:
        print(f"Error processing {hdf5_path}: {e}")
        return False
    
    # 验证切分点
    checkpoints = task_processor.validate_checkpoints(checkpoints, total_steps)
    num_phases = len(checkpoints) + 1
    
    # 获取子任务描述
    if hasattr(task_processor, 'get_subtask_descriptions_for_phases'):
        subtask_templates = task_processor.get_subtask_descriptions_for_phases(num_phases)
    else:
        all_descriptions = task_processor.get_subtask_descriptions()
        subtask_templates = all_descriptions[:num_phases]
    
    # 验证描述数量
    if len(subtask_templates) != num_phases:
        # print(f"Warning: Subtask count ({len(subtask_templates)}) != Phase count ({num_phases})")
        # 补齐或截断
        if len(subtask_templates) < num_phases:
            subtask_templates.extend([f"Phase {i}" for i in range(len(subtask_templates), num_phases)])
        else:
            subtask_templates = subtask_templates[:num_phases]
    
    # 获取高级指令（如果已存在）
    instructions = data.get("instructions", ["Default instruction"])
    if not isinstance(instructions, list) or len(instructions) == 0:
        instructions = ["Default instruction"]
    
    # 为每个指令生成对应的子任务列表
    subtasks_list = []
    for _ in instructions:
        subtasks_list.append(subtask_templates.copy())
    
    # 更新数据
    data["subtasks"] = subtasks_list
    data["phase_info"] = {
        "checkpoints": [int(cp) for cp in checkpoints],
        "total_steps": int(total_steps),
        "num_phases": num_phases
    }
    
    # 保存更新后的 JSON
    with open(instructions_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # 日志输出优化：显示是否使用了 Raw Z
    z_status = "Raw Z" if external_z is not None else "Fallback Qpos"
    print(f"✓ Processed {episode_dir.name} ({active_side}, {z_status}): {num_phases} phases, checkpoints={checkpoints}")
    return True


def load_task_processor(task_name: str) -> BaseTaskProcessor:
    """
    动态加载任务处理器
    """
    try:
        # 导入任务定义模块
        module = importlib.import_module(f"task_definitions.{task_name}")
        
        # 查找处理器类（约定：类名为 {TaskName}Processor）
        processor_class_name = f"{task_name.title().replace('_', '')}Processor"
        if hasattr(module, processor_class_name):
            processor_class = getattr(module, processor_class_name)
        else:
            # 尝试查找其他可能的类名
            classes = [cls for cls in dir(module) 
                      if isinstance(getattr(module, cls), type) 
                      and issubclass(getattr(module, cls), BaseTaskProcessor)
                      and cls != 'BaseTaskProcessor']
            if len(classes) > 0:
                processor_class = getattr(module, classes[0])
            else:
                raise ValueError(f"No processor class found in {task_name}")
        
        # 实例化处理器
        processor = processor_class()
        return processor
        
    except ImportError as e:
        print(f"Error: Cannot import task definition for '{task_name}'")
        print(f"Make sure task_definitions/{task_name}.py exists")
        raise
    except Exception as e:
        print(f"Error loading task processor: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="通用多阶段数据处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python process_data_generic.py --task_name beat_block_hammer --data_dir processed_data/beat_block_hammer-demo_clean-50
        """
    )
    parser.add_argument("--task_name", type=str, required=True,
                        help="任务名称（对应 task_definitions 下的模块名）")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据目录路径（包含多个 episode_XX 文件夹）")
    parser.add_argument("--episode_range", type=str, default=None,
                        help="处理的episode范围，例如 '0-10' 或 '0,5,10'")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return 1
    
    # 加载任务处理器
    print(f"Loading task processor for: {args.task_name}")
    try:
        task_processor = load_task_processor(args.task_name)
    except Exception as e:
        print(f"Failed to load task processor: {e}")
        return 1
    
    # 查找所有 episode 目录
    episode_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    
    if len(episode_dirs) == 0:
        print(f"Warning: No episode directories found in {data_dir}")
        return 1
    
    # 解析 episode 范围
    if args.episode_range:
        if '-' in args.episode_range:
            start, end = map(int, args.episode_range.split('-'))
            episode_indices = list(range(start, end + 1))
        else:
            episode_indices = [int(x) for x in args.episode_range.split(',')]
        episode_dirs = [episode_dirs[i] for i in episode_indices if 0 <= i < len(episode_dirs)]
    
    print(f"Found {len(episode_dirs)} episodes to process")
    
    # 处理每个 episode
    success_count = 0
    for episode_dir in episode_dirs:
        if process_episode_generic(episode_dir, task_processor):
            success_count += 1
    
    print(f"\n✓ Successfully processed {success_count}/{len(episode_dirs)} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())