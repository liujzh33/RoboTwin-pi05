#!/usr/bin/env python3
"""
从「正确执行」的 beat_block_hammer 数据中抽取首帧 + checkpoint 帧，生成 Judge 模型的“成功”样本，
与 extract_judge_dataset_beat_block_hammer.py 的错误样本合并，避免模型只会输出错误。

- 数据源：training_data/.../beat_block_hammer-aloha-agilex_clean_50-50 与 .../beat_block_hammer-aloha-agilex_randomized_500-500
  （每条仅 instructions.json，无 HDF5；图像从对应 raw dataset 读取）
- Raw 图像：dataset/beat_block_hammer/aloha-agilex_clean_50/data、aloha-agilex_randomized_500/data
  （HDF5 键：/observation/head_camera/rgb, left_camera/rgb, right_camera/rgb，每帧为 jpeg 字节）
- 规则（三种 reflection 类型）：
  1. 首帧一律为 episode 第一帧（0）。
  2. 综合判定正常（下标 0–5）、强调「没有发生特定错误」（下标 10–13）：结束帧从 checkpoints 中随机选，优先 >= total_steps//2。
  3. 强调抓取/物理接触的稳固（下标 6–9）：结束帧固定为 checkpoints[1]（抓取结束），与 reflection 语义对齐。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2  # noqa: F401
    import h5py  # noqa: F401
    from PIL import Image  # noqa: F401
except ImportError as e:
    raise ImportError("This script requires h5py, Pillow, and opencv-python. Install with: pip install h5py Pillow opencv-python") from e

IMAGE_KEYS_PI05 = ("cam_high", "cam_left_wrist", "cam_right_wrist")
# Raw Aloha HDF5 的 key（与 process_data.py / process_data_recovery.py 一致）
RAW_CAM_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}

SUCCESS_VARIANTS = {
    "error_type": "None",
    "reflection": [
        # 综合判定正常 (0-5)
        "[Reflection] The execution was successful. The action was performed correctly without any errors.",
        "[Reflection] No errors detected. The gripper successfully completed the intended task.",
        "[Reflection] The movement was smooth and accurate. No anomaly was observed.",
        "[Reflection] Task executed flawlessly. The physical interaction matched the expected goal.",
        "[Reflection] Everything is nominal. The gripper state and object position match the task requirements.",
        "[Reflection] Zero anomalies detected. The robotic arm followed the correct trajectory to completion.",
        # 强调抓取/物理接触的稳固 (6-9)：此类样本使用 start=0, end=checkpoints[1]
        "[Reflection] Successful grasp and manipulation. The object is securely held and moved as planned.",
        "[Reflection] The visual states confirm a successful operation. The grip is firm and well-positioned.",
        "[Reflection] No physical errors occurred. The interaction was precise and stable throughout.",
        "[Reflection] All movements were executed properly. The robot successfully achieved the desired physical state.",
        # 强调“没有发生特定的错误”(10-13)
        "[Reflection] The subtask was completed perfectly. No slipping, collision, or offset was found.",
        "[Reflection] Excellent execution. The arm moved accurately to the target without any premature actions.",
        "[Reflection] The current state shows perfect alignment and execution. No corrective action is needed.",
        "[Reflection] The operation proceeded normally. The end-effector reached the target state safely.",
    ],
    "correction": "[Correction] None",
}

# 使用 checkpoints[1] 的 reflection 下标（强调抓取稳固）
GRASP_STABLE_REFLEX_INDICES = (6, 7, 8, 9)


def _read_frame_from_raw_aloha(hdf5_path: Path, frame_idx: int) -> Optional[dict[str, np.ndarray]]:
    """从 raw Aloha HDF5 读一帧：/observation/<cam>/rgb 为 jpeg 字节，解码为 (H,W,3) uint8。"""
    result = {}
    try:
        with h5py.File(hdf5_path, "r") as f:
            obs = f.get("/observation")
            if obs is None:
                return None
            for pi05_key, raw_cam in RAW_CAM_MAP.items():
                g = obs.get(raw_cam)
                if g is None:
                    return None
                rgb = g.get("rgb")
                if rgb is None:
                    return None
                frame_data = rgb[frame_idx]
                if hasattr(frame_data, "tobytes"):
                    frame_data = frame_data.tobytes()
                arr = np.frombuffer(frame_data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result[pi05_key] = img
    except (KeyError, IndexError, OSError):
        return None
    return result


def save_success_frames(
    raw_data_dir: Path,
    episode_id: int,
    start_frame: int,
    end_frame: int,
    out_image_dir: Path,
    source_tag: str = "",
) -> Optional[tuple[list[str], list[str]]]:
    """从 raw Aloha HDF5 取 start/end 两帧三视角，保存 JPG，返回 (start_paths, end_paths)。source_tag 避免多数据源重名。"""
    prefix = f"success_{source_tag}_" if source_tag else "success_"
    h5path = raw_data_dir / f"episode{episode_id}.hdf5"
    if not h5path.exists():
        return None
    start_frames = _read_frame_from_raw_aloha(h5path, start_frame)
    end_frames = _read_frame_from_raw_aloha(h5path, end_frame)
    if not start_frames or not end_frames:
        return None
    out_image_dir.mkdir(parents=True, exist_ok=True)
    start_paths = []
    end_paths = []
    for key in IMAGE_KEYS_PI05:
        for label, frames in (("start", start_frames), ("end", end_frames)):
            sub = "start" if label == "start" else "end"
            fname = f"{prefix}ep{episode_id}_{sub}_{key}.jpg"
            path = out_image_dir / fname
            Image.fromarray(frames[key]).save(path, quality=95)
            abs_path = str(path.resolve())
            if label == "start":
                start_paths.append(abs_path)
            else:
                end_paths.append(abs_path)
    return (start_paths, end_paths)


def load_episode_instructions(instructions_path: Path) -> Optional[dict]:
    """读取 instructions.json，返回 checkpoints, total_steps, high_instruction。"""
    if not instructions_path.exists():
        return None
    with open(instructions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    phase_info = data.get("phase_info") or {}
    checkpoints = phase_info.get("checkpoints")
    total_steps = phase_info.get("total_steps")
    if not checkpoints or total_steps is None:
        return None
    instructions = data.get("instructions") or []
    high_instruction = instructions[0] if instructions else "Beat the block with the hammer."
    return {
        "checkpoints": checkpoints,
        "total_steps": int(total_steps),
        "high_instruction": high_instruction,
    }


def choose_end_frame(
    checkpoints: list[int],
    total_steps: int,
    reflection_idx: int,
) -> int:
    """按三种 reflection 类型选结束帧：抓取稳固(6–9)用 checkpoints[1]，其余从 checkpoints 随机且优先 >= total_steps//2。"""
    if reflection_idx in GRASP_STABLE_REFLEX_INDICES and len(checkpoints) >= 2:
        return checkpoints[1]
    half = max(1, total_steps // 2)
    eligible = [c for c in checkpoints if c >= half]
    if not eligible:
        eligible = list(checkpoints)
    return random.choice(eligible)


def build_success_sharegpt_item(
    high_instruction: str,
    start_paths: list[str],
    end_paths: list[str],
    reflection: str,
) -> dict:
    """一条成功样本：6 图 + 高层指令 + 三段式（error_type: None, reflection, correction: None）。"""
    all_paths = start_paths + end_paths
    n = len(all_paths)
    placeholders = " ".join(["<image>"] * n)
    user_text = (
        f"{placeholders}\n\n"
        f"Task: {high_instruction}\n\n"
        "Look at the start state (first 3 images) and the current state (last 3 images). "
        "What went wrong? Output exactly in this format:\n"
        "[error_type]: <one of: grasp_position_offset, grasp_orientation_mismatch, premature_close, grasp_slip, None>\n"
        "[reflection]: <short reflection>\n"
        "[correction]: <short correction>"
    )
    assistant_text = (
        f"[error_type]: {SUCCESS_VARIANTS['error_type']}\n"
        f"[reflection]: {reflection}\n"
        f"[correction]: {SUCCESS_VARIANTS['correction']}"
    )
    return {
        "conversations": [
            {"from": "user", "value": user_text},
            {"from": "assistant", "value": assistant_text},
        ],
        "images": all_paths,
    }


def collect_episodes_from_training_dir(training_dir: Path) -> list[int]:
    """列出所有 episode_N 的 N。"""
    episodes = []
    for d in training_dir.iterdir():
        if d.is_dir() and d.name.startswith("episode_"):
            try:
                n = int(d.name.split("_")[1])
                episodes.append(n)
            except (IndexError, ValueError):
                pass
    return sorted(episodes)


def main():
    parser = argparse.ArgumentParser(
        description="Extract success (correct) samples from normal beat_block_hammer data for Judge dataset."
    )
    parser.add_argument(
        "--training_data_dirs",
        type=str,
        nargs="+",
        default=[
            "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/training_data/pi05_multi_task_5_v1.0/beat_block_hammer-aloha-agilex_clean_50-50",
            "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/training_data/pi05_multi_task_5_v1.0/beat_block_hammer-aloha-agilex_randomized_500-500",
        ],
        help="Training data dirs (each has episode_N/instructions.json).",
    )
    parser.add_argument(
        "--raw_data_dirs",
        type=str,
        nargs="+",
        default=[
            "/mnt/data1/liujingzhi/dataset/beat_block_hammer/aloha-agilex_clean_50/data",
            "/mnt/data1/liujingzhi/dataset/beat_block_hammer/aloha-agilex_randomized_500/data",
        ],
        help="Raw HDF5 dirs (episode{N}.hdf5 with /observation/head_camera/rgb etc.), order matches --training_data_dirs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/mnt/data1/liujingzhi/RoboTwin/policy/pi05/judge_dataset/recovery_judge_beat_block_hammer",
        help="Output dir: images/ and recovery_judge_success_dataset.json (merge with error JSON if needed).",
    )
    parser.add_argument(
        "--max_per_source",
        type=int,
        default=50,
        help="Max success samples per training_data source (default: 50, so clean+rand total ≤100).",
    )
    parser.add_argument(
        "--merge_with_errors",
        type=str,
        default=None,
        help="Path to existing recovery_judge_dataset.json (errors); append success and write merged JSON to --out_dir.",
    )
    args = parser.parse_args()

    if len(args.training_data_dirs) != len(args.raw_data_dirs):
        raise SystemExit("--training_data_dirs and --raw_data_dirs must have the same length.")

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    success_items = []

    for training_dir, raw_dir in zip(
        [Path(p) for p in args.training_data_dirs],
        [Path(p) for p in args.raw_data_dirs],
    ):
        if not training_dir.exists():
            print(f"Skip (not found): {training_dir}")
            continue
        if not raw_dir.exists():
            print(f"Skip raw (not found): {raw_dir}")
            continue
        episode_ids = collect_episodes_from_training_dir(training_dir)
        if args.max_per_source is not None:
            episode_ids = episode_ids[: args.max_per_source]
        # 多数据源时 success 图片加前缀，避免 clean 与 randomized 的 episode 重名覆盖
        source_tag = "clean" if "clean_50" in training_dir.name else "rand"
        count_before = len(success_items)
        for ep_id in episode_ids:
            instr_path = training_dir / f"episode_{ep_id}" / "instructions.json"
            meta = load_episode_instructions(instr_path)
            if not meta:
                continue
            checkpoints = meta["checkpoints"]
            total_steps = meta["total_steps"]
            high_instruction = meta["high_instruction"]
            # 先随机选 reflection 类型，再按类型定 end_frame（抓取稳固类固定 checkpoints[1]）
            reflex_idx = random.randint(0, len(SUCCESS_VARIANTS["reflection"]) - 1)
            reflection = SUCCESS_VARIANTS["reflection"][reflex_idx]
            end_frame = choose_end_frame(checkpoints, total_steps, reflex_idx)
            paths = save_success_frames(raw_dir, ep_id, 0, end_frame, image_dir, source_tag=source_tag)
            if not paths:
                continue
            start_paths, end_paths = paths
            item = build_success_sharegpt_item(high_instruction, start_paths, end_paths, reflection)
            success_items.append(item)
        added = len(success_items) - count_before
        print(f"  From {training_dir.name}: added {added} success samples (total {len(success_items)}).")

    out_success = out_dir / "recovery_judge_success_dataset.json"
    with open(out_success, "w", encoding="utf-8") as f:
        json.dump(success_items, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(success_items)} success samples to {out_success}")

    if args.merge_with_errors:
        merge_path = Path(args.merge_with_errors)
        if merge_path.exists():
            with open(merge_path, "r", encoding="utf-8") as f:
                error_items = json.load(f)
            merged = error_items + success_items
            random.shuffle(merged)
            merged_path = out_dir / "recovery_judge_dataset_merged.json"
            with open(merged_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            print(f"Merged {len(error_items)} error + {len(success_items)} success -> {merged_path}")
        else:
            print(f"Merge path not found: {merge_path}")


if __name__ == "__main__":
    main()
