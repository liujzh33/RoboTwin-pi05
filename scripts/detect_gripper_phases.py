#!/usr/bin/env python3
"""
检测 HDF5 文件中的夹爪开闭状态，并自动分割子任务阶段
用于 beat_block_hammer 任务：
- 阶段1：抓取锤子（从开始到其中一个夹爪闭合）
- 阶段2：敲击积木（从夹爪闭合到结束）
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_hdf5_data(hdf5_path):
    """加载 HDF5 文件数据"""
    with h5py.File(hdf5_path, 'r') as f:
        # 获取 action 数据
        actions = f['action'][()]
        # 获取 qpos 数据（包含夹爪状态）
        qpos = f['observations/qpos'][()]
        # 获取 left_arm_dim 和 right_arm_dim 来确定夹爪位置
        left_arm_dim = f['observations/left_arm_dim'][0]
        right_arm_dim = f['observations/right_arm_dim'][0]
        
    return actions, qpos, left_arm_dim, right_arm_dim


def extract_gripper_values(qpos, left_arm_dim, right_arm_dim):
    """
    从 qpos 中提取左右夹爪的开闭值
    qpos 格式: [left_arm, left_gripper, right_arm, right_gripper]
    """
    # 根据 process_data.py 的格式：left_arm + [left_gripper] + right_arm + [right_gripper]
    left_gripper_idx = left_arm_dim
    right_gripper_idx = left_arm_dim + 1 + right_arm_dim
    
    left_gripper = qpos[:, left_gripper_idx]
    right_gripper = qpos[:, right_gripper_idx]
    
    return left_gripper, right_gripper


def detect_grasp_phase(left_gripper, right_gripper, threshold=0.5):
    """
    检测抓取阶段
    当其中一个夹爪闭合（值小于 threshold）时，认为抓取完成
    
    Args:
        left_gripper: 左夹爪开闭值数组
        right_gripper: 右夹爪开闭值数组
        threshold: 夹爪闭合阈值（0.0=完全闭合, 1.0=完全打开）
    
    Returns:
        grasp_end_idx: 抓取阶段结束的索引（第一个夹爪闭合的位置）
    """
    # 夹爪值越小表示闭合程度越高
    # 当 min(left_gripper, right_gripper) < threshold 时，认为抓取完成
    min_gripper = np.minimum(left_gripper, right_gripper)
    
    # 找到第一个小于阈值的位置
    grasp_end_idx = np.where(min_gripper < threshold)[0]
    
    if len(grasp_end_idx) > 0:
        return grasp_end_idx[0]
    else:
        # 如果没有找到，返回中间位置作为默认分割点
        return len(left_gripper) // 2


def visualize_gripper_states(hdf5_path, left_gripper, right_gripper, grasp_end_idx, save_path=None):
    """可视化夹爪状态并标记阶段分割点"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    time_steps = np.arange(len(left_gripper))
    
    # 绘制左夹爪
    axes[0].plot(time_steps, left_gripper, 'b-', label='Left Gripper', linewidth=2)
    axes[0].axvline(x=grasp_end_idx, color='r', linestyle='--', linewidth=2, label='Grasp End')
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('Left Gripper Value', fontsize=12)
    axes[0].set_title('Left Gripper State', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Threshold (0.5)')
    
    # 标记阶段
    axes[0].axvspan(0, grasp_end_idx, alpha=0.2, color='green', label='Phase 1: Grasp')
    axes[0].axvspan(grasp_end_idx, len(left_gripper), alpha=0.2, color='orange', label='Phase 2: Strike')
    
    # 绘制右夹爪
    axes[1].plot(time_steps, right_gripper, 'g-', label='Right Gripper', linewidth=2)
    axes[1].axvline(x=grasp_end_idx, color='r', linestyle='--', linewidth=2, label='Grasp End')
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Right Gripper Value', fontsize=12)
    axes[1].set_title('Right Gripper State', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Threshold (0.5)')
    
    # 标记阶段
    axes[1].axvspan(0, grasp_end_idx, alpha=0.2, color='green', label='Phase 1: Grasp')
    axes[1].axvspan(grasp_end_idx, len(right_gripper), alpha=0.2, color='orange', label='Phase 2: Strike')
    
    plt.suptitle(f'Gripper States and Phase Detection\n{hdf5_path}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def process_episode(hdf5_path, threshold=0.5, visualize=True, save_dir=None):
    """处理单个 episode 的 HDF5 文件"""
    print(f"\n{'='*60}")
    print(f"Processing: {hdf5_path}")
    print(f"{'='*60}")
    
    # 加载数据
    actions, qpos, left_arm_dim, right_arm_dim = load_hdf5_data(hdf5_path)
    
    # 提取夹爪值
    left_gripper, right_gripper = extract_gripper_values(qpos, left_arm_dim, right_arm_dim)
    
    # 检测抓取阶段结束点
    grasp_end_idx = detect_grasp_phase(left_gripper, right_gripper, threshold)
    
    # 打印统计信息
    print(f"Total time steps: {len(left_gripper)}")
    print(f"Left arm dim: {left_arm_dim}, Right arm dim: {right_arm_dim}")
    print(f"Left gripper range: [{left_gripper.min():.3f}, {left_gripper.max():.3f}]")
    print(f"Right gripper range: [{right_gripper.min():.3f}, {right_gripper.max():.3f}]")
    print(f"\nPhase Detection Results:")
    print(f"  Phase 1 (Grasp): steps 0-{grasp_end_idx} ({grasp_end_idx} steps, {grasp_end_idx/len(left_gripper)*100:.1f}%)")
    print(f"  Phase 2 (Strike): steps {grasp_end_idx}-{len(left_gripper)} ({len(left_gripper)-grasp_end_idx} steps, {(len(left_gripper)-grasp_end_idx)/len(left_gripper)*100:.1f}%)")
    print(f"  Grasp end index: {grasp_end_idx}")
    print(f"  Left gripper at grasp end: {left_gripper[grasp_end_idx]:.3f}")
    print(f"  Right gripper at grasp end: {right_gripper[grasp_end_idx]:.3f}")
    
    # 可视化
    if visualize:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            episode_name = Path(hdf5_path).stem
            save_path = os.path.join(save_dir, f"{episode_name}_gripper_phases.png")
        else:
            save_path = None
        visualize_gripper_states(hdf5_path, left_gripper, right_gripper, grasp_end_idx, save_path)
    
    return {
        'grasp_end_idx': grasp_end_idx,
        'total_steps': len(left_gripper),
        'left_gripper': left_gripper,
        'right_gripper': right_gripper,
        'phase1_steps': grasp_end_idx,
        'phase2_steps': len(left_gripper) - grasp_end_idx,
    }


def process_dataset(dataset_dir, threshold=0.5, visualize=True, save_dir=None):
    """处理整个数据集的所有 episode"""
    dataset_path = Path(dataset_dir)
    episodes = sorted(dataset_path.glob("episode_*/episode_*.hdf5"))
    
    if len(episodes) == 0:
        print(f"No HDF5 files found in {dataset_dir}")
        return []
    
    print(f"Found {len(episodes)} episodes to process")
    
    results = []
    for episode_path in episodes:
        try:
            result = process_episode(str(episode_path), threshold, visualize, save_dir)
            result['episode_path'] = str(episode_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {episode_path}: {e}")
            continue
    
    # 打印汇总统计
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    if results:
        phase1_avg = np.mean([r['phase1_steps'] for r in results])
        phase2_avg = np.mean([r['phase2_steps'] for r in results])
        total_avg = np.mean([r['total_steps'] for r in results])
        
        print(f"Total episodes processed: {len(results)}")
        print(f"Average total steps: {total_avg:.1f}")
        print(f"Average Phase 1 (Grasp) steps: {phase1_avg:.1f} ({phase1_avg/total_avg*100:.1f}%)")
        print(f"Average Phase 2 (Strike) steps: {phase2_avg:.1f} ({phase2_avg/total_avg*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect gripper phases in HDF5 files")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset directory or single HDF5 file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Gripper closure threshold (default: 0.5)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save visualization images"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    if dataset_path.is_file() and dataset_path.suffix == '.hdf5':
        # 处理单个文件
        process_episode(str(dataset_path), args.threshold, not args.no_visualize, args.save_dir)
    elif dataset_path.is_dir():
        # 处理整个数据集
        process_dataset(str(dataset_path), args.threshold, not args.no_visualize, args.save_dir)
    else:
        print(f"Error: {args.dataset_path} is not a valid HDF5 file or directory")

