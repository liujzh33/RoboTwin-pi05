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

        # 双臂速度（与 blocks_ranking_size 一致）
        vel_left = analyzer.compute_velocity(qpos, arm_indices=(0, 6))
        vel_right = analyzer.compute_velocity(qpos, arm_indices=(7, 13))
        velocity_max = np.maximum(vel_left, vel_right)

        grasp_idx = analyzer.detect_grasp_event(left_gripper, right_gripper)
        stop_points = analyzer.detect_stop_points(velocity_max)

        # 获取总步数
        total_steps = len(left_gripper)

        # ==========================================
        # [关键修改]：先读取 Raw Z 数据，再计算 Checkpoints
        # 这样计算逻辑看到的数据就和画图看到的数据一模一样了！
        # ==========================================
        
        # 真实双臂 Z + 真实双臂 EEF 位置（与 blocks_ranking_rgb_v1 一致：raw endpose 优先）
        z_left = None
        z_right = None
        eef_xyz_left = None
        eef_xyz_right = None
        if raw_episode_path is None:
            import re
            match = re.search(r"episode_(\d+)", str(hdf5_path))
            if match:
                episode_num = match.group(1)
                # 从 processed 路径推断 setting：如 .../beat_block_hammer-aloha-agilex_clean_50-50/episode_0/...
                data_dir_name = hdf5_path.parent.parent.name  # e.g. beat_block_hammer-aloha-agilex_clean_50-50
                if "clean_50" in data_dir_name:
                    setting = "aloha-agilex_clean_50"
                else:
                    setting = "aloha-agilex_randomized_500"
                raw_episode_path = Path(
                    "/mnt/data1/liujingzhi/dataset/beat_block_hammer/"
                    f"{setting}/data/episode{episode_num}.hdf5"
                )
        if raw_episode_path and raw_episode_path.exists():
            with h5py.File(raw_episode_path, "r") as raw_f:
                if "endpose/left_endpose" in raw_f and "endpose/right_endpose" in raw_f:
                    left_ep = raw_f["endpose/left_endpose"][()]
                    right_ep = raw_f["endpose/right_endpose"][()]
                    raw_len = min(len(left_ep), len(right_ep))
                    z_left = left_ep[:raw_len, 2].copy()
                    z_right = right_ep[:raw_len, 2].copy()
                    eef_xyz_left = left_ep[:raw_len, :3].copy() if left_ep.shape[1] >= 3 else None
                    eef_xyz_right = right_ep[:raw_len, :3].copy() if right_ep.shape[1] >= 3 else None
                    # 对齐到 processed 长度，避免 plot 时 x/y 维度不一致（raw 与 processed 步数可能不同）
                    if raw_len < total_steps:
                        last_l = float(z_left[-1]) if len(z_left) > 0 else 0.0
                        last_r = float(z_right[-1]) if len(z_right) > 0 else 0.0
                        z_left = np.concatenate([z_left, np.full(total_steps - raw_len, last_l)])
                        z_right = np.concatenate([z_right, np.full(total_steps - raw_len, last_r)])
                        if eef_xyz_left is not None and eef_xyz_right is not None:
                            eef_xyz_left = np.vstack([eef_xyz_left, np.tile(eef_xyz_left[-1], (total_steps - raw_len, 1))])
                            eef_xyz_right = np.vstack([eef_xyz_right, np.tile(eef_xyz_right[-1], (total_steps - raw_len, 1))])
                    elif raw_len > total_steps:
                        z_left = z_left[:total_steps]
                        z_right = z_right[:total_steps]
                        if eef_xyz_left is not None and eef_xyz_right is not None:
                            eef_xyz_left = eef_xyz_left[:total_steps]
                            eef_xyz_right = eef_xyz_right[:total_steps]
        if z_left is None and qpos.shape[1] >= 14:
            z_left = qpos[:total_steps, 2]
            z_right = qpos[:total_steps, 9]

        checkpoints = task_processor.get_phase_checkpoints(
            f,
            active_side=active_side,
            external_z=None,
            external_eef_xyz=None,
            external_z_left=z_left,
            external_z_right=z_right,
            external_eef_xyz_left=eef_xyz_left,
            external_eef_xyz_right=eef_xyz_right,
        )

    # 3) 创建可视化（4 行子图：gripper / velocity / Z / phases）
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    time_steps = np.arange(total_steps)

    # 子图1: 夹爪状态
    axes[0].plot(time_steps, left_gripper, "b-", label="Left Gripper", linewidth=2)
    axes[0].plot(time_steps, right_gripper, "g-", label="Right Gripper", linewidth=2)
    axes[0].axhline(y=0.95, color="green", linestyle="--", alpha=0.5, label="Open (0.95)")
    axes[0].axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="Closed (0.05)")

    axes[0].set_xlabel("Time Step", fontsize=12)
    axes[0].set_ylabel("Gripper Value", fontsize=12)
    axes[0].set_title("Gripper States", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2: 双臂 Z 轴高度（真实 endpose，与 blocks_ranking_size 一致）
    if z_left is None:
        z_left = np.zeros(total_steps)
        z_right = np.zeros(total_steps)
    axes[1].plot(time_steps, z_left, "b-", label="Left Z (Height)", linewidth=1.5)
    axes[1].plot(time_steps, z_right, "g-", label="Right Z (Height)", linewidth=1.5)
    axes[1].set_ylabel("Height (Z)", fontsize=12)
    axes[1].set_title("End-Effector Height", fontsize=14, fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # 子图3: 双臂速度 + Max（与 blocks_ranking_size 一致）
    axes[2].plot(time_steps, vel_left, "b-", alpha=0.7, label="Left Arm Velocity")
    axes[2].plot(time_steps, vel_right, "g-", alpha=0.7, label="Right Arm Velocity")
    axes[2].plot(time_steps, velocity_max, "k-", alpha=0.4, linewidth=1, label="Max")
    axes[2].axhline(
        y=task_processor.velocity_break_threshold,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label=f"Break threshold ({task_processor.velocity_break_threshold})",
    )
    if len(stop_points) > 0:
        axes[2].fill_between(
            time_steps,
            0,
            np.max(velocity_max),
            where=stop_points,
            color="gray",
            alpha=0.05,
            label="Stopped",
        )
    axes[2].set_xlabel("Time Step", fontsize=12)
    axes[2].set_ylabel("Velocity", fontsize=12)
    axes[2].set_title("Robot Movement Velocity", fontsize=14, fontweight="bold")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    for cp in checkpoints:
        for ax in axes[:3]:
            ax.axvline(x=cp, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

    # 子图4: 4 阶段划分（T1/T2/T3 红线已在上面子图绘制）
    axes[3].set_xlim(0, total_steps)
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    axes[3].set_xlabel("Time Step", fontsize=12)
    axes[3].set_title(f"Predicted Phases (Total: {len(checkpoints)+1})", fontsize=14, fontweight="bold")
    axes[3].grid(True, alpha=0.3)

    phases = [0] + [cp for cp in checkpoints if cp < total_steps] + [total_steps]
    phase_colors = ["#ccffcc", "#ffffcc", "#ffcccc", "#ccccff"]
    phase_labels = [
        "P0: Above hammer",
        "P1: Close to grasp",
        "P2: Move above block",
        "P3: Hit block",
    ]
    for i in range(len(phases) - 1):
        axes[3].axvspan(
            phases[i],
            phases[i + 1],
            alpha=0.5,
            color=phase_colors[i % len(phase_colors)],
        )
        axes[3].axvline(x=phases[i], color="k", linestyle="-", linewidth=1)
        mid = (phases[i] + phases[i + 1]) / 2
        label = phase_labels[i] if i < len(phase_labels) else f"Phase {i}"
        axes[3].text(mid, 0.5, label, ha="center", va="center", fontsize=10, fontweight="bold")

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
    print(f"Checkpoints (T1,T2,T3): {checkpoints}")
    print(f"Number of phases: {len(checkpoints) + 1}")
    print(f"Average velocity (max): {np.mean(velocity_max):.4f}")
    print(f"Max velocity: {np.max(velocity_max):.4f}")
    print(f"Stop points: {np.sum(stop_points)} ({100*np.sum(stop_points)/total_steps:.1f}%)")


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