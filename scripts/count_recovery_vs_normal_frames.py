#!/usr/bin/env python3
"""
统计 beat_block_hammer_recovery 数据源中，在「丢弃 [MASKED] 段」之后保留的帧里：
- 带 [Correction] 的帧数（recovery 阶段）
- 纯 normal 的帧数（无 [Correction] 的普通阶段）

与 data_loader._MaskedRecoveryFilteredTransformedDataset._should_drop_masked 逻辑一致。
用法（在 policy/pi05 下）:
  uv run python scripts/count_recovery_vs_normal_frames.py --data_dir training_data/beat_block_hammer_recovery
  或
  uv run python scripts/count_recovery_vs_normal_frames.py --data_dir training_data/beat_block_hammer_recovery_pi05
"""

import argparse
import json
import os
from pathlib import Path


def get_masked_end(subtasks: list, checkpoints: list, total_steps: int) -> int | None:
    """与 data_loader._should_drop_masked 一致：计算最早 masked 阶段结束帧。"""
    if not (isinstance(subtasks, list) and len(subtasks) > 0) or not checkpoints:
        return None
    masked_ends = []
    for variant in subtasks:
        if not (isinstance(variant, list) and len(variant) > 0):
            continue
        masked_phase_idx = None
        for i, s in enumerate(variant):
            if isinstance(s, str) and "[MASKED]" in s:
                masked_phase_idx = i
                break
        if masked_phase_idx is None:
            continue
        masked_end = (
            int(checkpoints[masked_phase_idx])
            if masked_phase_idx < len(checkpoints)
            else total_steps
        )
        masked_ends.append(masked_end)
    if not masked_ends:
        return None
    return min(masked_ends)


def phase_idx_for_frame(frame_idx: int, checkpoints: list) -> int:
    """根据 frame_idx 和 checkpoints 得到当前 phase 下标。"""
    for i, cp in enumerate(checkpoints):
        if frame_idx < cp:
            return i
    return len(checkpoints)


def main():
    parser = argparse.ArgumentParser(description="Count correction vs normal retained frames")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="training_data/beat_block_hammer_recovery",
        help="Root dir containing task subdirs (e.g. beat_block_hammer-demo_clean-recovery-66), each with episode_*/instructions.json",
    )
    args = parser.parse_args()
    root = Path(args.data_dir)
    if not root.exists():
        print(f"Data dir not found: {root}")
        return

    total_retained = 0
    correction_frames = 0
    normal_frames = 0
    episodes_processed = 0
    episodes_with_masked = 0
    total_dropped = 0

    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        for ep_dir in sorted(task_dir.iterdir(), key=lambda p: (p.name,)):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            json_path = ep_dir / "instructions.json"
            if not json_path.exists():
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            subtasks = data.get("subtasks", [])
            phase_info = data.get("phase_info", {})
            if isinstance(phase_info, str):
                phase_info = json.loads(phase_info)
            checkpoints = phase_info.get("checkpoints", [])
            total_steps = int(phase_info.get("total_steps", 0) or 0)
            if not checkpoints or total_steps <= 0:
                continue

            masked_end = get_masked_end(subtasks, checkpoints, total_steps)
            if masked_end is None:
                # 无 [MASKED] 的 episode：全部保留，按 phase 区分 correction / normal
                start_idx = 0
                episodes_processed += 1
            else:
                start_idx = masked_end
                episodes_processed += 1
                episodes_with_masked += 1
                total_dropped += start_idx

            # 使用第一条 instruction 变体判定 phase 类型（与 LoadSubtaskFromInstructions 的 phase 一致）
            if not subtasks or not isinstance(subtasks[0], list):
                continue
            variant0 = subtasks[0]

            for frame_idx in range(start_idx, total_steps):
                total_retained += 1
                pi = phase_idx_for_frame(frame_idx, checkpoints)
                if pi >= len(variant0):
                    normal_frames += 1
                    continue
                st = variant0[pi]
                if isinstance(st, str) and "[Correction]" in st:
                    correction_frames += 1
                else:
                    normal_frames += 1

    print("=" * 60)
    print("beat_block_hammer_recovery 保留帧统计（与 drop_masked 后一致）")
    print("=" * 60)
    print(f"数据目录: {root.resolve()}")
    print(f"处理 episode 数: {episodes_processed}")
    print(f"其中含 [MASKED] 的 episode 数: {episodes_with_masked}")
    print(f"被丢弃的帧数（frame_idx < masked_end）: {total_dropped}")
    print("-" * 60)
    print(f"保留帧总数: {total_retained}")
    print(f"  带 [Correction] 的帧数: {correction_frames}")
    print(f"  纯 normal 的帧数:       {normal_frames}")
    if total_retained > 0:
        print(f"  占比: [Correction] {100.0 * correction_frames / total_retained:.1f}%  |  normal {100.0 * normal_frames / total_retained:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
