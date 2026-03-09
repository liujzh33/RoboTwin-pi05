#!/usr/bin/env python3
"""
通用的多阶段数据处理脚本 v1

- 在原版基础上，除了提供 Raw Z（高度）信息外，
  还从原始 episode*.hdf5 中读取末端执行器的三维位置 (x, y, z)，
  以便任务处理器可以基于运动基元 / 速度谷值进行更精细的阶段划分。
"""

import os
import sys
import h5py
import json
import argparse
import importlib
import re
from typing import Optional

import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task_definitions.base_task import BaseTaskProcessor


# 任务名 -> 原始数据根目录（含 data 的上一级）
RAW_DATA_BASE_BY_TASK = {
    "beat_block_hammer": "/mnt/data1/liujingzhi/dataset/beat_block_hammer/aloha-agilex_randomized_500/data",
    "click_alarmclock": "/mnt/data1/liujingzhi/dataset/click_alarmclock/aloha-agilex_randomized_500/data",
    "click_bell": "/mnt/data1/liujingzhi/dataset/click_bell/aloha-agilex_randomized_500/data",
    "blocks_ranking_rgb": "/mnt/data1/liujingzhi/dataset/blocks_ranking_rgb/aloha-agilex_randomized_500/data",
    "blocks_ranking_size": "/mnt/data1/liujingzhi/dataset/blocks_ranking_size/aloha-agilex_randomized_500/data",
    "blocks_ranking_rgb_v1": "/mnt/data1/liujingzhi/dataset/blocks_ranking_rgb/aloha-agilex_randomized_500/data",
    "blocks_ranking_size_v1": "/mnt/data1/liujingzhi/dataset/blocks_ranking_size/aloha-agilex_randomized_500/data",
}
DEFAULT_RAW_BASE = "/mnt/data1/liujingzhi/dataset/beat_block_hammer/aloha-agilex_randomized_500/data"


def get_raw_eef_xyz_from_reference(
    episode_dir: Path, active_side: str, task_name: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    从原始数据集中读取末端执行器的三维位置 (x, y, z)。
    task_name 用于选择数据集路径（如 click_alarmclock）。
    """
    episode_num = None
    match = re.search(r"episode_(\d+)", str(episode_dir.name))
    if match:
        episode_num = match.group(1)

    if episode_num is None:
        return None

    raw_base = RAW_DATA_BASE_BY_TASK.get(task_name, DEFAULT_RAW_BASE)
    raw_episode_path = Path(raw_base) / f"episode{episode_num}.hdf5"

    if not raw_episode_path.exists():
        return None

    try:
        with h5py.File(raw_episode_path, "r") as raw_f:
            if "endpose/left_endpose" not in raw_f or "endpose/right_endpose" not in raw_f:
                return None

            if active_side == "right":
                endpose = raw_f["endpose/right_endpose"][()]
            else:
                endpose = raw_f["endpose/left_endpose"][()]

            if endpose.ndim != 2 or endpose.shape[1] < 3:
                return None

            # 只取前三维，保证是 (x, y, z)
            eef_xyz = endpose[:, :3]
            return np.asarray(eef_xyz)
    except Exception as e:
        print(f"  [Error] Failed to read raw EEF from {raw_episode_path}: {e}")
        return None


def get_raw_dual_arm_eef_from_reference(
    episode_dir: Path, task_name: Optional[str] = None, total_steps: Optional[int] = None
) -> tuple:
    """
    从原始数据集中读取双臂末端 (x,y,z)，用于需要真实双臂速度+Z 的任务（如 beat_block_hammer）。
    返回 (eef_xyz_left, eef_xyz_right)，均为 (T, 3)；若缺失则对应为 None。
    """
    episode_num = None
    match = re.search(r"episode_(\d+)", str(episode_dir.name))
    if match:
        episode_num = match.group(1)
    if episode_num is None:
        return None, None

    raw_base = RAW_DATA_BASE_BY_TASK.get(task_name, DEFAULT_RAW_BASE)
    raw_episode_path = Path(raw_base) / f"episode{episode_num}.hdf5"
    if not raw_episode_path.exists():
        return None, None

    try:
        with h5py.File(raw_episode_path, "r") as raw_f:
            if "endpose/left_endpose" not in raw_f or "endpose/right_endpose" not in raw_f:
                return None, None
            left_ep = np.asarray(raw_f["endpose/left_endpose"][()])
            right_ep = np.asarray(raw_f["endpose/right_endpose"][()])
            if left_ep.ndim != 2 or left_ep.shape[1] < 3 or right_ep.ndim != 2 or right_ep.shape[1] < 3:
                return None, None
            eef_left = left_ep[:, :3]
            eef_right = right_ep[:, :3]
            if total_steps is not None:
                eef_left = eef_left[:total_steps]
                eef_right = eef_right[:total_steps]
            return eef_left, eef_right
    except Exception as e:
        print(f"  [Error] Failed to read raw dual-arm EEF from {raw_episode_path}: {e}")
        return None, None


def process_episode_generic(
    episode_dir: Path, task_processor: BaseTaskProcessor, task_name: Optional[str] = None
) -> bool:
    """
    通用的单集处理函数（v1）
    - 额外向任务处理器传入 external_eef_xyz (和 external_z) 以支持运动基元划分。
    """
    # 查找 HDF5 文件
    hdf5_files = list(episode_dir.glob("episode_*.hdf5"))
    if len(hdf5_files) == 0:
        print(f"Warning: No HDF5 file found in {episode_dir}")
        return False

    hdf5_path = hdf5_files[0]

    # 读取或创建 instructions.json
    instructions_path = episode_dir / "instructions.json"

    # 读取现有的 instructions（如果存在）
    if instructions_path.exists():
        with open(instructions_path, "r") as f:
            data = json.load(f)
    else:
        # 创建默认结构
        data = {
            "instructions": ["Default instruction"],
            "subtasks": [],
        }

    # 从 HDF5 提取切分点
    external_eef_xyz = None
    external_z = None

    try:
        with h5py.File(hdf5_path, "r") as f:
            total_steps = (
                f["action"].shape[0]
                if "action" in f
                else f["observations/qpos"].shape[0]
            )

            # ======================================================
            # [步骤 1] 确定 Active Side（用于选择正确的末端轨迹）
            # ======================================================
            active_side = "left"  # 默认兜底
            if hasattr(task_processor, "analyzer") and hasattr(
                task_processor.analyzer, "extract_gripper_states"
            ):
                left_gripper, right_gripper = task_processor.analyzer.extract_gripper_states(
                    f
                )

                left_delta = float(left_gripper.max() - left_gripper.min())
                right_delta = float(right_gripper.max() - right_gripper.min())
                active_side = "left" if left_delta >= right_delta else "right"

            # ======================================================
            # [步骤 2] 从原始数据中读取末端 (x, y, z)
            # ======================================================
            external_eef_xyz = get_raw_eef_xyz_from_reference(
                episode_dir, active_side, task_name=task_name
            )
            if external_eef_xyz is not None and external_eef_xyz.shape[0] > 0:
                external_z = external_eef_xyz[:, 2]

            # 真实双臂 Z + 双臂 EEF（用于 beat_block_hammer 等需要双臂速度/高度划分的任务）
            external_z_left = None
            external_z_right = None
            external_eef_xyz_left = None
            external_eef_xyz_right = None
            if task_name == "beat_block_hammer":
                eef_left, eef_right = get_raw_dual_arm_eef_from_reference(
                    episode_dir, task_name=task_name, total_steps=total_steps
                )
                if eef_left is not None and eef_right is not None:
                    external_z_left = eef_left[:, 2]
                    external_z_right = eef_right[:, 2]
                    external_eef_xyz_left = eef_left
                    external_eef_xyz_right = eef_right

            # ======================================================
            # [步骤 3] 计算 Checkpoints
            # ======================================================
            try:
                checkpoints = task_processor.get_phase_checkpoints(
                    f,
                    active_side=active_side,
                    external_z=external_z,
                    external_eef_xyz=external_eef_xyz,
                    external_z_left=external_z_left,
                    external_z_right=external_z_right,
                    external_eef_xyz_left=external_eef_xyz_left,
                    external_eef_xyz_right=external_eef_xyz_right,
                )
            except TypeError:
                try:
                    checkpoints = task_processor.get_phase_checkpoints(
                        f,
                        active_side=active_side,
                        external_z=external_z,
                        external_eef_xyz=external_eef_xyz,
                    )
                except TypeError:
                    try:
                        checkpoints = task_processor.get_phase_checkpoints(
                            f, active_side=active_side, external_z=external_z
                        )
                    except TypeError:
                        checkpoints = task_processor.get_phase_checkpoints(f)

    except Exception as e:
        print(f"Error processing {hdf5_path}: {e}")
        return False

    # 验证切分点
    checkpoints = task_processor.validate_checkpoints(checkpoints, total_steps)
    num_phases = len(checkpoints) + 1

    # 获取子任务描述
    if hasattr(task_processor, "get_subtask_descriptions_for_phases"):
        subtask_templates = task_processor.get_subtask_descriptions_for_phases(num_phases)
    else:
        all_descriptions = task_processor.get_subtask_descriptions()
        subtask_templates = all_descriptions[:num_phases]

    # 验证描述数量
    if len(subtask_templates) != num_phases:
        if len(subtask_templates) < num_phases:
            subtask_templates.extend(
                [f"Phase {i}" for i in range(len(subtask_templates), num_phases)]
            )
        else:
            subtask_templates = subtask_templates[:num_phases]

    # 获取高级指令（如果已存在）
    instructions = data.get("instructions", ["Default instruction"])
    if not isinstance(instructions, list) or len(instructions) == 0:
        instructions = ["Default instruction"]

    # 为每个指令生成对应的子任务列表
    subtasks_list = []
    for _ in instructions:
        subtasks_list.append(subtask_templates.copy())

    # 更新数据
    data["subtasks"] = subtasks_list
    data["phase_info"] = {
        "checkpoints": [int(cp) for cp in checkpoints],
        "total_steps": int(total_steps),
        "num_phases": num_phases,
    }

    # 保存更新后的 JSON
    with open(instructions_path, "w") as f:
        json.dump(data, f, indent=2)

    # 日志：标记是否成功使用 Raw EEF
    if external_eef_xyz_left is not None and external_eef_xyz_right is not None:
        eef_status = "Raw dual-arm EEF (Z+velocity)"
    elif external_eef_xyz is not None:
        eef_status = "Raw EEF XYZ"
    elif external_z is not None:
        eef_status = "Raw Z only"
    else:
        eef_status = "No external EEF (fallback to processed data only)"

    print(
        f"✓ [v1] Processed {episode_dir.name} ({active_side}, {eef_status}): "
        f"{num_phases} phases, checkpoints={checkpoints}"
    )
    return True


def load_task_processor(task_name: str) -> BaseTaskProcessor:
    """
    动态加载任务处理器
    """
    try:
        module = importlib.import_module(f"task_definitions.{task_name}")

        processor_class_name = f"{task_name.title().replace('_', '')}Processor"
        if hasattr(module, processor_class_name):
            processor_class = getattr(module, processor_class_name)
        else:
            classes = [
                cls
                for cls in dir(module)
                if isinstance(getattr(module, cls), type)
                and issubclass(getattr(module, cls), BaseTaskProcessor)
                and cls != "BaseTaskProcessor"
            ]
            if len(classes) > 0:
                processor_class = getattr(module, classes[0])
            else:
                raise ValueError(f"No processor class found in {task_name}")

        processor = processor_class()
        return processor

    except ImportError:
        print(f"Error: Cannot import task definition for '{task_name}'")
        print(f"Make sure task_definitions/{task_name}.py exists")
        raise
    except Exception as e:
        print(f"Error loading task processor: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="通用多阶段数据处理脚本 v1（支持末端 EEF 运动基元）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python process_data_generic_v1.py --task_name blocks_ranking_size_v1 --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_randomized_500-200
        """,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="任务名称（对应 task_definitions 下的模块名）",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="数据目录路径（包含多个 episode_XX 文件夹）",
    )
    parser.add_argument(
        "--episode_range",
        type=str,
        default=None,
        help="处理的episode范围，例如 '0-10' 或 '0,5,10'",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return 1

    # 加载任务处理器
    print(f"Loading task processor for: {args.task_name}")
    try:
        task_processor = load_task_processor(args.task_name)
    except Exception as e:
        print(f"Failed to load task processor: {e}")
        return 1

    # 查找所有 episode 目录
    episode_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")]
    )

    if len(episode_dirs) == 0:
        print(f"Warning: No episode directories found in {data_dir}")
        return 1

    # 解析 episode 范围
    if args.episode_range:
        if "-" in args.episode_range:
            start, end = map(int, args.episode_range.split("-"))
            episode_indices = list(range(start, end + 1))
        else:
            episode_indices = [int(x) for x in args.episode_range.split(",")]
        episode_dirs = [
            episode_dirs[i] for i in episode_indices if 0 <= i < len(episode_dirs)
        ]

    print(f"Found {len(episode_dirs)} episodes to process")

    # 处理每个 episode
    success_count = 0
    for episode_dir in episode_dirs:
        if process_episode_generic(episode_dir, task_processor, task_name=args.task_name):
            success_count += 1

    print(f"\n✓ Successfully processed {success_count}/{len(episode_dirs)} episodes (v1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

