#!/usr/bin/env python3
"""
hanging_mug 轨迹分析：4 面板图（夹爪、末端高度 Z、速度、阶段划分）。
5 阶段：靠近马克杯 → 抓住马克杯 → 放置到桌面中间 → 另一臂抓杯沿 → 悬挂马克杯。
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_definitions.hanging_mug import HangingMugProcessor
from task_definitions.trajectory_analyzer import TrajectoryAnalyzer

RAW_DATA_ROOT = "/mnt/data1/liujingzhi/dataset/hanging_mug/aloha-agilex_randomized_500/data"


def _get_raw_episode_path(hdf5_path: Path) -> Path | None:
    match = re.search(r"episode_(\d+)", hdf5_path.name)
    if not match:
        return None
    return Path(RAW_DATA_ROOT) / f"episode{match.group(1)}.hdf5"


def _short_phase_label(desc: str, i: int) -> str:
    if i == 0:
        return "Approach"
    if i == 1:
        return "Grasp"
    if i == 2:
        return "Place Middle"
    if i == 3:
        return "Other Grasp Rim"
    if i == 4:
        return "Hang"
    return f"P{i}"


def analyze_episode(hdf5_path: Path, save_path: Path | None = None):
    analyzer = TrajectoryAnalyzer()
    task_processor = HangingMugProcessor()

    print(f"Analyzing: {hdf5_path.name}")

    with h5py.File(hdf5_path, "r") as f:
        left_gripper, right_gripper = analyzer.extract_gripper_states(f)
        total_steps = len(left_gripper)

        key_qpos = "observations/qpos" if "observations/qpos" in f else "qpos"
        qpos = f[key_qpos][()]

        vel_left = analyzer.compute_velocity(qpos, arm_indices=(0, 6))
        vel_right = analyzer.compute_velocity(qpos, arm_indices=(7, 13))
        velocity = np.maximum(vel_left, vel_right)

        checkpoints = task_processor.get_phase_checkpoints(f, active_side=None, external_z=None)
        checkpoints = task_processor.validate_checkpoints(checkpoints, total_steps)

        z_left = None
        z_right = None
        raw_path = _get_raw_episode_path(hdf5_path)
        if raw_path and raw_path.exists():
            with h5py.File(raw_path, "r") as raw_f:
                if "endpose/left_endpose" in raw_f:
                    z_left = raw_f["endpose/left_endpose"][()][:, 2][:total_steps]
                    z_right = raw_f["endpose/right_endpose"][()][:, 2][:total_steps]
        if z_left is None and qpos.shape[1] >= 14:
            z_left = qpos[:total_steps, 2]
            z_right = qpos[:total_steps, 9]

    num_phases = len(checkpoints) + 1
    descriptions = task_processor.get_subtask_descriptions_for_phases(num_phases)
    time_steps = np.arange(total_steps)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))

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
    else:
        axes[1].set_title("End-Effector Height (no Z data)", fontweight="bold")

    axes[2].plot(time_steps, velocity, "k-", alpha=0.6, label="Max Joint Velocity")
    axes[2].set_ylabel("Velocity")
    axes[2].set_title("Robot Movement Velocity", fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_xlim(0, total_steps)
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    legend_note = "[Approach, Grasp, Place Middle, Other Grasp Rim, Hang]"
    axes[3].set_title(f"Predicted Phases (Total: {num_phases}) {legend_note}", fontweight="bold")

    phases = [0] + list(checkpoints) + [total_steps]
    colors = ["#ffcccc", "#ccffcc", "#ccccff", "#ffffcc", "#ffccff"]

    for i in range(len(phases) - 1):
        start, end = phases[i], phases[i + 1]
        mid = (start + end) / 2
        color = colors[i % len(colors)]
        axes[3].axvspan(start, end, color=color, alpha=0.5)
        axes[3].axvline(x=start, color="k", linestyle="-", linewidth=1)
        desc = descriptions[i] if i < len(descriptions) else f"Phase {i}"
        short = _short_phase_label(desc, i)
        axes[3].text(mid, 0.5, f"P{i}: {short}", ha="center", va="center", fontsize=9, fontweight="bold")

    # 红线：另一臂抓住杯沿（第三阶段开始，cp3）
    if len(checkpoints) >= 3:
        cp3 = int(checkpoints[2])
        for ax in axes:
            ax.axvline(x=cp3, color="red", linestyle="-", linewidth=2, alpha=0.9, zorder=5)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Hanging Mug trajectory analysis (5 phases)")
    parser.add_argument("hdf5_path", type=str, help="Path to episode_*/episode_*.hdf5")
    parser.add_argument("--save", type=str, default=None, help="Path to save png")
    args = parser.parse_args()
    analyze_episode(Path(args.hdf5_path), save_path=Path(args.save) if args.save else None)


if __name__ == "__main__":
    main()
