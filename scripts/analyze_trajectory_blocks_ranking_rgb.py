#!/usr/bin/env python3
"""
analyze_blocks_ranking.py
针对 Blocks Ranking RGB 任务的专用分析工具
特点：同时可视化双臂的 Z 轴高度，以验证交替操作逻辑。
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task_definitions.trajectory_analyzer import TrajectoryAnalyzer
# 使用 v1 版处理器（带精细阶段划分）
from task_definitions.blocks_ranking_rgb_v1 import BlocksRankingRgbV1Processor

def analyze_episode(hdf5_path: Path, save_path: Path = None, raw_episode_path: Path = None):
    analyzer = TrajectoryAnalyzer()
    task_processor = BlocksRankingRgbV1Processor()

    print(f"Analyzing: {hdf5_path.name}")

    # 1) 读取 processed 数据
    with h5py.File(hdf5_path, "r") as f:
        # 提取夹爪状态
        left_gripper, right_gripper = analyzer.extract_gripper_states(f)
        total_steps = len(left_gripper)
        
        # 提取关节位置 (用于计算速度)
        key_qpos = "observations/qpos" if "observations/qpos" in f else "qpos"
        qpos = f[key_qpos][()]

        # 计算双臂平均速度作为整体活力指标
        # 左臂: 0-6, 右臂: 7-13
        vel_left = analyzer.compute_velocity(qpos, arm_indices=(0, 6))
        vel_right = analyzer.compute_velocity(qpos, arm_indices=(7, 13))
        # 取最大值，看哪只手在动
        velocity = np.maximum(vel_left, vel_right)

        # ==========================================
        # [Raw Z 读取] 双臂版本
        # ==========================================
        z_left = None
        z_right = None
        
        # 自动推断 raw path（根据 data_dir 区分 clean_50 / randomized_500）
        if raw_episode_path is None:
            match = re.search(r"episode_(\d+)", str(hdf5_path.name))
            if match:
                episode_num = match.group(1)
                data_dir_name = hdf5_path.parent.parent.name
                setting = "aloha-agilex_clean_50" if "clean_50" in data_dir_name else "aloha-agilex_randomized_500"
                raw_episode_path = Path(
                    "/mnt/data1/liujingzhi/dataset/blocks_ranking_rgb/"
                    f"{setting}/data/episode{episode_num}.hdf5"
                )
        
        if raw_episode_path and raw_episode_path.exists():
            print(f"Loading Raw Z from: {raw_episode_path}")
            with h5py.File(raw_episode_path, "r") as raw_f:
                if "endpose/left_endpose" in raw_f:
                    left_ep = raw_f["endpose/left_endpose"][()]
                    right_ep = raw_f["endpose/right_endpose"][()]
                    raw_len = min(len(left_ep), len(right_ep))
                    z_left = np.asarray(left_ep[:raw_len, 2], dtype=float)
                    z_right = np.asarray(right_ep[:raw_len, 2], dtype=float)
                    # 对齐到 processed 长度，避免 plot 时 x/y 维度不一致
                    if raw_len < total_steps:
                        last_l = float(z_left[-1]) if len(z_left) > 0 else 0.0
                        last_r = float(z_right[-1]) if len(z_right) > 0 else 0.0
                        z_left = np.concatenate([z_left, np.full(total_steps - raw_len, last_l)])
                        z_right = np.concatenate([z_right, np.full(total_steps - raw_len, last_r)])
                    elif raw_len > total_steps:
                        z_left = z_left[:total_steps]
                        z_right = z_right[:total_steps]
        else:
            print(f"Warning: Raw file not found at {raw_episode_path}, plotting qpos approximations.")
            # Fallback: qpos idx 2 (left lift) and idx 9 (right lift)
            if qpos.shape[1] >= 14:
                z_left = qpos[:total_steps, 2]
                z_right = qpos[:total_steps, 9]

        # 2. 计算 Checkpoints（传入 velocity 对齐所需的信息）
        # 这里 active_side 对 v1 影响不大，传 None 即可
        checkpoints = task_processor.get_phase_checkpoints(f)

    # 3) 可视化 (4行)
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    time_steps = np.arange(total_steps)

    # Subplot 1: 夹爪 (最重要的信号)
    axes[0].plot(time_steps, left_gripper, "b-", label="Left Gripper", alpha=0.8)
    axes[0].plot(time_steps, right_gripper, "g-", label="Right Gripper", alpha=0.8)
    axes[0].axhline(y=0.4, color="k", linestyle="--", alpha=0.3, label="Threshold (0.4)")
    axes[0].set_ylabel("Gripper State")
    axes[0].set_title("Gripper States (Low=Closed, High=Open)", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Z 轴高度 (双臂对比)
    if z_left is not None:
        axes[1].plot(time_steps, z_left, "b-", label="Left Z (Height)", linewidth=1.5)
        axes[1].plot(time_steps, z_right, "g-", label="Right Z (Height)", linewidth=1.5)
        axes[1].set_ylabel("Height (Z)")
        axes[1].set_title("End-Effector Height", fontweight="bold")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

    # Subplot 3: 速度
    axes[2].plot(time_steps, velocity, "k-", alpha=0.6, label="Max Joint Velocity")
    axes[2].set_ylabel("Velocity")
    axes[2].set_title("Robot Movement Velocity", fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    # Subplot 4: 阶段划分
    axes[3].set_xlim(0, total_steps)
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    axes[3].set_title(f"Predicted Phases (Total: {len(checkpoints)+1})", fontweight="bold")
    
    # 绘制阶段色块
    phases = [0] + checkpoints + [total_steps]
    colors = ["#ffcccc", "#ccffcc", "#ccccff", "#ffffcc", "#ffccff", "#ccffff"] # 红绿蓝基调
    
    descriptions = task_processor.get_subtask_descriptions_for_phases(len(phases)-1)
    
    for i in range(len(phases) - 1):
        start, end = phases[i], phases[i+1]
        mid = (start + end) / 2
        color = colors[i % len(colors)]
        
        axes[3].axvspan(start, end, color=color, alpha=0.5)
        axes[3].axvline(x=start, color="k", linestyle="-", linewidth=1)
        
        # 添加文本描述
        desc_text = descriptions[i] if i < len(descriptions) else f"Phase {i}"
        # 简化显示：只显示 Pick/Place 关键词
        short_desc = "Unknown"
        if "Pick" in desc_text: short_desc = f"P{i}: Pick"
        elif "Place" in desc_text: short_desc = f"P{i}: Place"
        else: short_desc = f"P{i}"
            
        axes[3].text(mid, 0.5, short_desc, ha="center", va="center", fontsize=10, rotation=0, fontweight="bold")

    # 在前三个子图上绘制垂直红线以标记 checkpoints
    for cp in checkpoints:
        for ax in axes[:3]:
            ax.axvline(x=cp, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Blocks Ranking Analysis")
    parser.add_argument("hdf5_path", type=str)
    parser.add_argument("--save", type=str, default="analysis_result.png")
    args = parser.parse_args()

    analyze_episode(Path(args.hdf5_path), save_path=Path(args.save))

if __name__ == "__main__":
    main()