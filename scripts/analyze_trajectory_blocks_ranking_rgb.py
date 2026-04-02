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
import json

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

    # 可选：从 instructions.json 读取 recovery 标注（error_attempt + subtasks_per_frame）
    instr_data = None
    error_attempt_range = None
    instr_path = hdf5_path.parent / "instructions.json"
    if instr_path.exists():
        try:
            with open(instr_path, "r", encoding="utf-8") as fp:
                instr_data = json.load(fp)
            rng = (instr_data.get("phase_info") or {}).get("error_attempt_range")
            if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                error_attempt_range = (int(rng[0]), int(rng[1]))
        except Exception:
            instr_data = None

    use_annotated_phases = False
    annotated_segments = []
    if (
        instr_data
        and isinstance(instr_data.get("subtasks_per_frame"), list)
        and len(instr_data["subtasks_per_frame"]) == total_steps
    ):
        spf = instr_data["subtasks_per_frame"]
        base_descs = task_processor.get_subtask_descriptions()

        def _phase_idx_from_subtask(s):
            s = (s or "").strip()
            if not s or "[MASKED]" in s:
                return -1
            if " [Subtask] " in s:
                s = s.split(" [Subtask] ")[-1].strip()
            for i, d in enumerate(base_descs):
                if d in s or s == d:
                    return i
            return -1

        phase_indices = [_phase_idx_from_subtask(x) for x in spf]
        if phase_indices:
            start = 0
            cur = phase_indices[0]
            for i in range(1, total_steps + 1):
                if i == total_steps or phase_indices[i] != cur:
                    annotated_segments.append((start, i, cur))
                    if i < total_steps:
                        start = i
                        cur = phase_indices[i]
            use_annotated_phases = len(annotated_segments) > 0

    # 3) 可视化 (4行)
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    time_steps = np.arange(total_steps)

    # 在所有子图上给 error_attempt 打淡红底色
    if error_attempt_range is not None:
        err_s, err_e = error_attempt_range
        for ax in axes:
            ax.axvspan(err_s, err_e + 1, color="red", alpha=0.12, zorder=0)

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
    axes[3].set_title(
        "Predicted Phases (from recovery annotation)"
        if use_annotated_phases
        else f"Predicted Phases (Total: {len(checkpoints)+1})",
        fontweight="bold",
    )

    colors = ["#ffcccc", "#ccffcc", "#ccccff", "#ffffcc", "#ffccff", "#ccffff"] # 红绿蓝基调

    if use_annotated_phases:
        for seg_start, seg_end, pidx in annotated_segments:
            axes[3].axvspan(seg_start, seg_end, color=colors[pidx % len(colors)] if pidx >= 0 else "#ffcccc", alpha=0.5)
            axes[3].axvline(x=seg_start, color="k", linestyle="-", linewidth=1)
            mid = (seg_start + seg_end) / 2
            label = f"P{pidx}" if pidx >= 0 else "Error (masked)"
            axes[3].text(mid, 0.5, label, ha="center", va="center", fontsize=10, fontweight="bold")
        for seg_start, _, pidx in annotated_segments:
            if pidx >= 0 and seg_start > 0:
                for ax in axes[:3]:
                    ax.axvline(x=seg_start, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    else:
        phases = [0] + checkpoints + [total_steps]
        descriptions = task_processor.get_subtask_descriptions_for_phases(len(phases)-1)
        for i in range(len(phases) - 1):
            start, end = phases[i], phases[i+1]
            mid = (start + end) / 2
            color = colors[i % len(colors)]
            axes[3].axvspan(start, end, color=color, alpha=0.5)
            axes[3].axvline(x=start, color="k", linestyle="-", linewidth=1)
            desc_text = descriptions[i] if i < len(descriptions) else f"Phase {i}"
            short_desc = "Unknown"
            if "Pick" in desc_text:
                short_desc = f"P{i}: Pick"
            elif "Place" in desc_text:
                short_desc = f"P{i}: Place"
            else:
                short_desc = f"P{i}"
            axes[3].text(mid, 0.5, short_desc, ha="center", va="center", fontsize=10, rotation=0, fontweight="bold")
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