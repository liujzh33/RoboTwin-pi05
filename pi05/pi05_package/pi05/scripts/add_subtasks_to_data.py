#!/usr/bin/env python3
"""
为已处理的数据添加子任务信息
基于夹爪状态检测结果，为每个 episode 生成子任务描述
"""

import os
import json
import h5py
import numpy as np
import argparse
from pathlib import Path
from detect_gripper_phases import load_hdf5_data, extract_gripper_values, detect_grasp_phase


def generate_subtask_descriptions(high_level_instruction, phase1_steps, phase2_steps, total_steps):
    """
    根据高级任务和阶段信息生成子任务描述
    
    Args:
        high_level_instruction: 高级任务描述
        phase1_steps: 阶段1的步数（抓取）
        phase2_steps: 阶段2的步数（敲击）
        total_steps: 总步数
    
    Returns:
        subtask1: 第一个子任务描述（抓取锤子）
        subtask2: 第二个子任务描述（敲击积木）
    """
    # 根据任务类型生成子任务描述
    # 对于 beat_block_hammer 任务，固定为两个子任务
    subtask1 = "Grab the hammer"
    subtask2 = "Strike the block with the hammer"
    
    # 可以根据高级任务描述进行更智能的生成
    # 这里使用简单的固定描述，你可以根据需要改进
    
    return subtask1, subtask2


def update_instructions_json(instructions_path, phase_info, threshold=0.5):
    """
    更新 instructions.json，添加子任务信息
    
    Args:
        instructions_path: instructions.json 文件路径
        phase_info: 阶段信息字典，包含 grasp_end_idx 等
        threshold: 夹爪阈值
    """
    # 读取现有的 instructions.json
    with open(instructions_path, 'r') as f:
        data = json.load(f)
    
    # 获取高级任务描述
    high_level_instructions = data.get('instructions', [])
    
    # 为每个高级任务描述生成子任务
    subtasks_list = []
    for high_level in high_level_instructions:
        subtask1, subtask2 = generate_subtask_descriptions(
            high_level,
            phase_info['phase1_steps'],
            phase_info['phase2_steps'],
            phase_info['total_steps']
        )
        subtasks_list.append([subtask1, subtask2])
    
    # 添加子任务信息到 JSON
    data['subtasks'] = subtasks_list
    data['phase_info'] = {
        'grasp_end_idx': int(phase_info['grasp_end_idx']),
        'total_steps': int(phase_info['total_steps']),
        'phase1_steps': int(phase_info['phase1_steps']),
        'phase2_steps': int(phase_info['phase2_steps']),
        'threshold': threshold
    }
    
    # 保存更新后的 JSON
    with open(instructions_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated {instructions_path} with subtask information")


def process_episode_directory(episode_dir, threshold=0.5):
    """处理单个 episode 目录"""
    episode_path = Path(episode_dir)
    
    # 查找 HDF5 文件
    hdf5_files = list(episode_path.glob("episode_*.hdf5"))
    if len(hdf5_files) == 0:
        print(f"No HDF5 file found in {episode_dir}")
        return False
    
    hdf5_path = hdf5_files[0]
    
    # 查找 instructions.json
    instructions_path = episode_path / "instructions.json"
    if not instructions_path.exists():
        print(f"No instructions.json found in {episode_dir}")
        return False
    
    # 检测阶段
    try:
        actions, qpos, left_arm_dim, right_arm_dim = load_hdf5_data(str(hdf5_path))
        left_gripper, right_gripper = extract_gripper_values(qpos, left_arm_dim, right_arm_dim)
        grasp_end_idx = detect_grasp_phase(left_gripper, right_gripper, threshold)
        
        phase_info = {
            'grasp_end_idx': grasp_end_idx,
            'total_steps': len(left_gripper),
            'phase1_steps': grasp_end_idx,
            'phase2_steps': len(left_gripper) - grasp_end_idx,
        }
        
        # 更新 instructions.json
        update_instructions_json(str(instructions_path), phase_info, threshold)
        return True
        
    except Exception as e:
        print(f"Error processing {episode_dir}: {e}")
        return False


def process_dataset(dataset_dir, threshold=0.5):
    """处理整个数据集"""
    dataset_path = Path(dataset_dir)
    episode_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    
    if len(episode_dirs) == 0:
        print(f"No episode directories found in {dataset_dir}")
        return
    
    print(f"Found {len(episode_dirs)} episodes to process")
    
    success_count = 0
    for episode_dir in episode_dirs:
        if process_episode_directory(str(episode_dir), threshold):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {success_count}/{len(episode_dirs)} episodes updated")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add subtask information to processed data")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory (e.g., processed_data/beat_block_hammer-...) or episode directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Gripper closure threshold (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    if dataset_path.is_dir():
        # 检查是否是 episode 目录（包含 episode_*.hdf5）
        hdf5_files = list(dataset_path.glob("episode_*.hdf5"))
        if hdf5_files:
            # 单个 episode 目录
            process_episode_directory(str(dataset_path), args.threshold)
        else:
            # 数据集目录
            process_dataset(str(dataset_path), args.threshold)
    else:
        print(f"Error: {args.dataset_path} is not a valid directory")

