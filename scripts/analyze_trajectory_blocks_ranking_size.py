#!/usr/bin/env python3
"""
针对 Blocks Ranking Size 任务的轨迹分析工具
可视化夹爪、Z 轴高度、速度与阶段划分（小→最右，中→中间，大→最左）。
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import re

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task_definitions.trajectory_analyzer import TrajectoryAnalyzer
from task_definitions.blocks_ranking_size import BlocksRankingSizeProcessor


def analyze_episode(hdf5_path: Path, save_path: Path = None, raw_episode_path: Path = None):
    analyzer = TrajectoryAnalyzer()
    task_processor = BlocksRankingSizeProcessor()

    print(f"Analyzing: {hdf5_path.name}")

    with h5py.File(hdf5_path, "r") as f:
        left_gripper, right_gripper = analyzer.extract_gripper_states(f)
        total_steps = len(left_gripper)

        key_qpos = "observations/qpos" if "observations/qpos" in f else "qpos"
        qpos = f[key_qpos][()]

        vel_left = analyzer.compute_velocity(qpos, arm_indices=(0, 6))
        vel_right = analyzer.compute_velocity(qpos, arm_indices=(7, 13))
        velocity = np.maximum(vel_left, vel_right)

        z_left = None
        z_right = None

        if raw_episode_path is None:
            match = re.search(r"episode_(\d+)", str(hdf5_path.name))
            if match:
                episode_num = match.group(1)
                raw_episode_path = Path(
                    "/mnt/data1/liujingzhi/dataset/blocks_ranking_size/"
                    f"aloha-agilex_randomized_500/data/episode{episode_num}.hdf5"
                )

        if raw_episode_path and raw_episode_path.exists():
            print(f"Loading Raw Z from: {raw_episode_path}")
            with h5py.File(raw_episode_path, "r") as raw_f:
                if "endpose/left_endpose" in raw_f:
                    z_left = raw_f["endpose/left_endpose"][()][:, 2]
                    z_right = raw_f["endpose/right_endpose"][()][:, 2]
                    z_left = z_left[:total_steps]
                    z_right = z_right[:total_steps]
        else:
            print(f"Warning: Raw file not found at {raw_episode_path}, plotting qpos approximations.")
            if qpos.shape[1] >= 14:
                z_left = qpos[:total_steps, 2]
                z_right = qpos[:total_steps, 9]

        checkpoints = task_processor.get_phase_checkpoints(f)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    time_steps = np.arange(total_steps)

    axes[0].plot(time_steps, left_gripper, "b-", label="Left Gripper", alpha=0.8)
    axes[0].plot(time_steps, right_gripper, "g-", label="Right Gripper", alpha=0.8)
    axes[0].axhline(y=0.2, color="k", linestyle="--", alpha=0.3, label="Threshold (0.2)")
    axes[0].set_ylabel("Gripper State")
    axes[0].set_title("Gripper States (Low=Closed, High=Open)", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    if z_left is not None:
        axes[1].plot(time_steps, z_left, "b-", label="Left Z (Height)", linewidth=1.5)
        axes[1].plot(time_steps, z_right, "g-", label="Right Z (Height)", linewidth=1.5)
        axes[1].set_ylabel("Height (Z)")
        axes[1].set_title("End-Effector Height", fontweight="bold")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_steps, velocity, "k-", alpha=0.6, label="Max Joint Velocity")
    axes[2].set_ylabel("Velocity")
    axes[2].set_title("Robot Movement Velocity", fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_xlim(0, total_steps)
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    axes[3].set_title(f"Predicted Phases (Total: {len(checkpoints)+1}) [small→far right, medium→middle, large→far left]", fontweight="bold")

    phases = [0] + checkpoints + [total_steps]
    colors = ["#ffcccc", "#ccffcc", "#ccccff", "#ffffcc", "#ffccff", "#ccffff", "#f0e68c"]

    descriptions = task_processor.get_subtask_descriptions_for_phases(len(phases) - 1)

    for i in range(len(phases) - 1):
        start, end = phases[i], phases[i + 1]
        mid = (start + end) / 2
        color = colors[i % len(colors)]

        axes[3].axvspan(start, end, color=color, alpha=0.5)
        axes[3].axvline(x=start, color="k", linestyle="-", linewidth=1)

        desc_text = descriptions[i] if i < len(descriptions) else f"Phase {i}"
        if "Pick" in desc_text:
            short_desc = f"P{i}: Pick"
        elif "Place" in desc_text:
            short_desc = f"P{i}: Place"
        elif "Return" in desc_text:
            short_desc = f"P{i}: Return"
        else:
            short_desc = f"P{i}"

        axes[3].text(mid, 0.5, short_desc, ha="center", va="center", fontsize=10, rotation=0, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Blocks Ranking Size Analysis")
    parser.add_argument("hdf5_path", type=str)
    parser.add_argument("--save", type=str, default="analysis_result.png")
    parser.add_argument("--raw_episode", type=str, default=None, help="Optional raw episode HDF5 for Z")
    args = parser.parse_args()

    raw_path = Path(args.raw_episode) if args.raw_episode else None
    analyze_episode(Path(args.hdf5_path), save_path=Path(args.save), raw_episode_path=raw_path)


if __name__ == "__main__":
    main()
