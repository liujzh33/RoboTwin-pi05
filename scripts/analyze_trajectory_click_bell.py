#!/usr/bin/env python3
"""
click_bell 轨迹分析：2 阶段（靠近+闭合 → click），可视化夹爪、速度与阶段划分。
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task_definitions.trajectory_analyzer import TrajectoryAnalyzer
from task_definitions.click_bell import ClickBellProcessor


def analyze_episode(hdf5_path: Path, save_path: Path = None):
    analyzer = TrajectoryAnalyzer()
    task_processor = ClickBellProcessor()

    print(f"Analyzing: {hdf5_path.name}")

    with h5py.File(hdf5_path, "r") as f:
        left_gripper, right_gripper = analyzer.extract_gripper_states(f)
        total_steps = len(left_gripper)

        key_qpos = "observations/qpos" if "observations/qpos" in f else "qpos"
        qpos = f[key_qpos][()]

        vel_left = analyzer.compute_velocity(qpos, arm_indices=(0, 6))
        vel_right = analyzer.compute_velocity(qpos, arm_indices=(7, 13))
        velocity = np.maximum(vel_left, vel_right)

        checkpoints = task_processor.get_phase_checkpoints(f)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    time_steps = np.arange(total_steps)

    axes[0].plot(time_steps, left_gripper, "b-", label="Left Gripper", alpha=0.8)
    axes[0].plot(time_steps, right_gripper, "g-", label="Right Gripper", alpha=0.8)
    axes[0].axhline(y=0.2, color="k", linestyle="--", alpha=0.3, label="Threshold (0.2)")
    axes[0].set_ylabel("Gripper State")
    axes[0].set_title("Gripper (Low=Closed). Phase split at first close.", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_steps, velocity, "k-", alpha=0.6, label="Max Joint Velocity")
    axes[1].set_ylabel("Velocity")
    axes[1].set_title("Robot Movement Velocity", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlim(0, total_steps)
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_title(f"Phases (Total: {len(checkpoints)+1}) [Approach+Close → Click]", fontweight="bold")

    phases = [0] + checkpoints + [total_steps]
    colors = ["#ccffcc", "#ccccff"]
    descriptions = task_processor.get_subtask_descriptions_for_phases(len(phases) - 1)

    for i in range(len(phases) - 1):
        start, end = phases[i], phases[i + 1]
        mid = (start + end) / 2
        color = colors[i % len(colors)]
        axes[2].axvspan(start, end, color=color, alpha=0.5)
        axes[2].axvline(x=start, color="k", linestyle="-", linewidth=1)
        short = "P0: Approach" if i == 0 else "P1: Click"
        axes[2].text(mid, 0.5, short, ha="center", va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Click Bell Analysis (2 phases)")
    parser.add_argument("hdf5_path", type=str)
    parser.add_argument("--save", type=str, default="analysis_result.png")
    args = parser.parse_args()
    analyze_episode(Path(args.hdf5_path), save_path=Path(args.save))


if __name__ == "__main__":
    main()
