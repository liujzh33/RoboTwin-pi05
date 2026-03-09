#!/usr/bin/env python3
"""
click_alarmclock 轨迹分析：4 子图（夹爪、双臂 Z 高度、速度、阶段），
3 阶段纯夹爪边界 0.95/0.05，前三个子图绘制 checkpoint 红线。
"""

import sys
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task_definitions.trajectory_analyzer import TrajectoryAnalyzer
from task_definitions.click_alarmclock import ClickAlarmclockProcessor


def analyze_episode(hdf5_path: Path, save_path: Path = None, raw_episode_path: Path = None):
    analyzer = TrajectoryAnalyzer()
    task_processor = ClickAlarmclockProcessor()

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
                data_dir_name = hdf5_path.parent.parent.name
                setting = "aloha-agilex_clean_50" if "clean_50" in data_dir_name else "aloha-agilex_randomized_500"
                raw_episode_path = Path(
                    "/mnt/data1/liujingzhi/dataset/click_alarmclock/"
                    f"{setting}/data/episode{episode_num}.hdf5"
                )
        if raw_episode_path and raw_episode_path.exists():
            with h5py.File(raw_episode_path, "r") as raw_f:
                if "endpose/left_endpose" in raw_f:
                    left_ep = raw_f["endpose/left_endpose"][()]
                    right_ep = raw_f["endpose/right_endpose"][()]
                    raw_len = min(len(left_ep), len(right_ep))
                    z_left = np.asarray(left_ep[:raw_len, 2], dtype=float)
                    z_right = np.asarray(right_ep[:raw_len, 2], dtype=float)
                    if raw_len < total_steps:
                        last_l = float(z_left[-1]) if len(z_left) > 0 else 0.0
                        last_r = float(z_right[-1]) if len(z_right) > 0 else 0.0
                        z_left = np.concatenate([z_left, np.full(total_steps - raw_len, last_l)])
                        z_right = np.concatenate([z_right, np.full(total_steps - raw_len, last_r)])
                    elif raw_len > total_steps:
                        z_left = z_left[:total_steps]
                        z_right = z_right[:total_steps]
        else:
            if qpos.shape[1] >= 14:
                z_left = qpos[:total_steps, 2]
                z_right = qpos[:total_steps, 9]

        # T2 由 Z 轴明显下降定义：传入主动臂的 Z（夹爪变化更大的一侧）
        left_delta = float(left_gripper.max() - left_gripper.min())
        right_delta = float(right_gripper.max() - right_gripper.min())
        active_z = (z_left if left_delta >= right_delta else z_right) if (z_left is not None) else None
        if active_z is not None:
            active_z = active_z[:total_steps]
        checkpoints = task_processor.get_phase_checkpoints(
            f, external_z=active_z, external_eef_xyz=None
        )

    if z_left is None:
        z_left = np.zeros(total_steps)
        z_right = np.zeros(total_steps)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    time_steps = np.arange(total_steps)

    axes[0].plot(time_steps, left_gripper, "b-", label="Left Gripper", alpha=0.8)
    axes[0].plot(time_steps, right_gripper, "g-", label="Right Gripper", alpha=0.8)
    axes[0].axhline(y=0.95, color="green", linestyle="--", alpha=0.5, label="Open (0.95)")
    axes[0].axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="Closed (0.05)")
    axes[0].set_ylabel("Gripper State")
    axes[0].set_title("Gripper States (Low=Closed, High=Open)", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_steps, z_left, "b-", label="Left Z (Height)", linewidth=1.5)
    axes[1].plot(time_steps, z_right, "g-", label="Right Z (Height)", linewidth=1.5)
    axes[1].set_ylabel("Height (Z)")
    axes[1].set_title("End-Effector Height", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_steps, velocity, "k-", alpha=0.6, label="Max Joint Velocity")
    axes[2].set_ylabel("Velocity")
    axes[2].set_title("Robot Movement Velocity", fontweight="bold")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    for cp in checkpoints:
        for ax in axes[:3]:
            ax.axvline(x=cp, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

    axes[3].set_xlim(0, total_steps)
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    axes[3].set_title(f"Predicted Phases (Total: {len(checkpoints)+1})", fontweight="bold")

    phases = [0] + checkpoints + [total_steps]
    colors = ["#ccffcc", "#ffffcc", "#ccccff"]
    descriptions = task_processor.get_subtask_descriptions_for_phases(len(phases) - 1)

    for i in range(len(phases) - 1):
        start, end = phases[i], phases[i + 1]
        mid = (start + end) / 2
        color = colors[i % len(colors)]
        axes[3].axvspan(start, end, color=color, alpha=0.5)
        axes[3].axvline(x=start, color="k", linestyle="-", linewidth=1)
        desc_text = descriptions[i] if i < len(descriptions) else f"Phase {i}"
        if i == 0:
            short = "P0: Above alarm clock"
        elif i == 1:
            short = "P1: Close to prepare"
        else:
            short = "P2: Click and return"
        axes[3].text(mid, 0.5, short, ha="center", va="center", fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Click Alarmclock Analysis (3 phases)")
    parser.add_argument("hdf5_path", type=str)
    parser.add_argument("--save", type=str, default="analysis_result.png")
    args = parser.parse_args()
    analyze_episode(Path(args.hdf5_path), save_path=Path(args.save))


if __name__ == "__main__":
    main()
