#!/usr/bin/env python3
"""
轨迹分析工具：可视化 HDF5 数据中的运动特征

用于帮助理解数据分布，确定多阶段切分规则
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task_definitions.trajectory_analyzer import TrajectoryAnalyzer
from task_definitions.beat_block_hammer import BeatBlockHammerProcessor


def analyze_episode(hdf5_path: Path, save_path: Path = None, raw_episode_path: Path = None):
    """
    分析单个 episode 的轨迹特征

    Args:
        hdf5_path: processed HDF5 文件路径（episode_0.hdf5）
        save_path: 保存图片的路径（可选）
        raw_episode_path: 原始 episode 文件路径（可选，用于读取 endpose 数据）
    """
    analyzer = TrajectoryAnalyzer()
    task_processor = BeatBlockHammerProcessor()

    # 1) 读取 processed 数据（用于 qpos / gripper / phase 计算）
    with h5py.File(hdf5_path, "r") as f:
        # 提取夹爪状态（基于 observations/qpos）
        left_gripper, right_gripper = analyzer.extract_gripper_states(f)

        # 选择“活动”的那只夹爪（变化幅度更大的一侧）
        left_delta = float(left_gripper.max() - left_gripper.min())
        right_delta = float(right_gripper.max() - right_gripper.min())
        if left_delta >= right_delta:
            active_side = "left"
        else:
            active_side = "right"

        # 提取关节位置
        if "observations/qpos" in f:
            qpos = f["observations/qpos"][()]
        elif "qpos" in f:
            qpos = f["qpos"][()]
        else:
            print("Warning: Cannot find qpos data")
            return

        # 计算速度：根据活动臂选择左臂或右臂关节
        if active_side == "left":
            velocity = analyzer.compute_velocity(qpos, arm_indices=(0, 6))
        else:
            # 右臂关节位于 qpos[:, 7:13]
            velocity = analyzer.compute_velocity(qpos, arm_indices=(7, 13))

        # 检测关键事件
        grasp_idx = analyzer.detect_grasp_event(left_gripper, right_gripper)
        stop_points = analyzer.detect_stop_points(velocity)

        # 获取总步数
        total_steps = len(left_gripper)

        # ==========================================
        # [关键修改]：先读取 Raw Z 数据，再计算 Checkpoints
        # 这样计算逻辑看到的数据就和画图看到的数据一模一样了！
        # ==========================================
        
        # 1. 先尝试获取 Raw Z 数据（用于计算和可视化）
        z_values = None
        z_source = "N/A"
        
        # 自动推断 raw path (如果未提供参数)
        if raw_episode_path is None:
            if "episode_" in str(hdf5_path):
                import re
                match = re.search(r"episode_(\d+)", str(hdf5_path))
                if match:
                    episode_num = match.group(1)
                    raw_episode_path = Path(
                        "/mnt/data1/liujingzhi/dataset/beat_block_hammer/"
                        f"aloha-agilex_randomized_500/data/episode{episode_num}.hdf5"
                    )
        
        if raw_episode_path and raw_episode_path.exists():
            with h5py.File(raw_episode_path, "r") as raw_f:
                if "endpose/left_endpose" in raw_f and "endpose/right_endpose" in raw_f:
                    # 读取 raw endpose
                    left_endpose = raw_f["endpose/left_endpose"][()]
                    right_endpose = raw_f["endpose/right_endpose"][()]
                    
                    # 根据活动侧选择对应末端 Z 轴
                    if active_side == "left":
                        z_raw = left_endpose[:, 2]
                        z_source = "raw endpose/left_endpose[:,2]"
                    else:
                        z_raw = right_endpose[:, 2]
                        z_source = "raw endpose/right_endpose[:,2]"
                    
                    # 对齐长度
                    if len(z_raw) >= total_steps:
                        z_values = z_raw[:total_steps]
                    else:
                        # 如果 raw 比 processed 短，使用 raw 的长度
                        z_values = z_raw
                        total_steps = len(z_values)
                        # 同时截断其他数组以匹配
                        left_gripper = left_gripper[:total_steps]
                        right_gripper = right_gripper[:total_steps]
                        velocity = velocity[:total_steps]
                        stop_points = stop_points[:total_steps]
                else:
                    print("Warning: no endpose/left_endpose in raw file, fallback to qpos[:,2]")
                    if qpos.shape[1] >= 3:
                        z_values = qpos[:total_steps, 2]
                        z_source = "processed qpos[:,2]"
                    else:
                        z_values = None
                        z_source = "N/A"
        else:
            if raw_episode_path:
                print(f"Warning: raw episode file not found: {raw_episode_path}")
            # 如果没有 raw 文件，回退到使用 processed qpos
            if qpos.shape[1] >= 3:
                z_values = qpos[:total_steps, 2]
                z_source = "processed qpos[:,2] (fallback)"
            else:
                z_values = None
                z_source = "N/A"

        # 2. 调用 get_phase_checkpoints，传入 external_z
        # 这样计算逻辑看到的数据就和画图看到的数据一模一样了！
        checkpoints = task_processor.get_phase_checkpoints(
            f, 
            active_side=active_side, 
            external_z=z_values  # <--- 关键修改：传入 Raw Z 数据
        )

    # 3) 创建可视化（4 行子图：gripper / velocity / Z / phases）
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    time_steps = np.arange(total_steps)

    # 子图1: 夹爪状态
    axes[0].plot(time_steps, left_gripper, "b-", label="Left Gripper", linewidth=2)
    axes[0].plot(time_steps, right_gripper, "g-", label="Right Gripper", linewidth=2)
    axes[0].axhline(
        y=analyzer.gripper_threshold,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Threshold ({analyzer.gripper_threshold})",
    )

    if grasp_idx is not None and grasp_idx < total_steps:
        axes[0].axvline(
            x=grasp_idx,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Grasp Event (t={grasp_idx})",
        )

    axes[0].set_xlabel("Time Step", fontsize=12)
    axes[0].set_ylabel("Gripper Value", fontsize=12)
    axes[0].set_title("Gripper States", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2: 运动速度
    axes[1].plot(time_steps, velocity, "b-", label="Joint Velocity", linewidth=2)
    axes[1].axhline(
        y=analyzer.velocity_threshold,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label=f"Stop Threshold ({analyzer.velocity_threshold})",
    )

    # 标记静止区域
    stop_regions = np.where(stop_points)[0]
    if len(stop_regions) > 0:
        axes[1].fill_between(
            time_steps,
            0,
            np.max(velocity),
            where=stop_points,
            color="gray",
            alpha=0.05,
            label="Stopped",
        )

    if grasp_idx is not None and grasp_idx < total_steps:
        axes[1].axvline(
            x=grasp_idx,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Grasp Event (t={grasp_idx})",
        )

    axes[1].set_xlabel("Time Step", fontsize=12)
    axes[1].set_ylabel("Velocity Magnitude", fontsize=12)
    axes[1].set_title("Arm Movement Velocity", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 子图3: 末端 Z 轴变化（来自原始 endpose）
    if z_values is not None:
        z_time = np.arange(z_values.shape[0])
        axes[2].plot(
            z_time,
            z_values,
            "m-",
            label=f"Z ({z_source})",
            linewidth=2,
        )
        if grasp_idx is not None and grasp_idx < z_values.shape[0]:
            axes[2].axvline(
                x=grasp_idx,
                color="r",
                linestyle="--",
                linewidth=1,
                label=f"Grasp Event (t={grasp_idx})",
            )
        axes[2].set_ylabel("Z", fontsize=12)
        axes[2].set_title("End-Effector Z Trajectory", fontsize=14, fontweight="bold")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(
            0.5,
            0.5,
            "No Z data available",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title(
            "End-Effector Z Trajectory (missing)", fontsize=14, fontweight="bold"
        )
        axes[2].set_axis_off()

    # 子图4: 阶段划分建议（使用 checkpoints）
    axes[3].set_xlim(0, total_steps)
    axes[3].set_ylim(-0.5, 0.5)
    axes[3].set_xlabel("Time Step", fontsize=12)
    axes[3].set_title("Suggested Phase Boundaries", fontsize=14, fontweight="bold")
    axes[3].grid(True, alpha=0.3)

    suggested_checkpoints = list(checkpoints)
    if len(suggested_checkpoints) > 0:
        colors = ["orange", "red", "purple", "blue", "brown"]
        labels = ["Boundary 0", "Boundary 1", "Boundary 2", "Boundary 3", "Boundary 4"]

        for i, cp in enumerate(suggested_checkpoints):
            if cp >= total_steps:
                continue
            color = colors[i % len(colors)]
            label = labels[i] if i < len(labels) else f"Boundary {i}"
            axes[3].axvline(
                x=cp,
                color=color,
                linestyle="-",
                linewidth=2,
                label=f"{label} (t={cp})",
            )

        # 标记阶段区域
        phases = [0] + [cp for cp in suggested_checkpoints if cp < total_steps] + [
            total_steps
        ]
        phase_colors = ["green", "yellow", "orange", "red", "blue"]
        for i in range(len(phases) - 1):
            axes[3].axvspan(
                phases[i],
                phases[i + 1],
                alpha=0.2,
                color=phase_colors[i % len(phase_colors)],
                label=f"Phase {i}",
            )

    axes[3].legend(loc="upper right")

    plt.suptitle(f"Trajectory Analysis: {hdf5_path.name}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    # 打印统计信息
    print("\n=== Trajectory Analysis Summary ===")
    print(f"Total steps (processed): {total_steps}")
    print(f"Grasp event: {grasp_idx if grasp_idx is not None else 'Not detected'}")
    print(f"Suggested checkpoints: {suggested_checkpoints}")
    print(f"Number of phases: {len(suggested_checkpoints) + 1}")
    print(f"Average velocity: {np.mean(velocity):.4f}")
    print(f"Max velocity: {np.max(velocity):.4f}")
    print(f"Stop points: {np.sum(stop_points)} ({100*np.sum(stop_points)/total_steps:.1f}%)")
    print(f"Z source: {z_source}")


def main():
    parser = argparse.ArgumentParser(
        description="分析 HDF5 轨迹数据，可视化运动特征",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("hdf5_path", type=str, help="processed HDF5 文件路径")
    parser.add_argument(
        "--save", type=str, default=None, help="保存图片的路径（可选）"
    )
    parser.add_argument(
        "--raw_episode", type=str, default=None,
        help="原始 episode 文件路径（可选，用于读取 endpose 数据，如果不提供会尝试自动推断）"
    )

    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        print(f"Error: File does not exist: {hdf5_path}")
        return 1

    save_path = Path(args.save) if args.save else None
    raw_episode_path = Path(args.raw_episode) if args.raw_episode else None

    try:
        analyze_episode(hdf5_path, save_path, raw_episode_path)
    except Exception as e:
        print(f"Error analyzing trajectory: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())