#!/usr/bin/env python3
"""
从 beat_block_hammer recovery 数据中提取 error_attempt 开始帧与结束帧（三视角），
生成 LLaMA-Factory ShareGPT 格式的 Judge Model 训练集。

- 输入：processed_data/beat_block_hammer-demo_clean-recovery-66（instructions.json + phase_info）
- 图像来源：dataset_recovery/beat_block_hammer/demo_clean/data/episode{N}.hdf5（三视角与 pi05 对齐）
- 输出：recovery_judge_dataset.json + 图片目录，可直接用于 LLaMA-Factory 微调 Qwen2-VL-7B-Instruct。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2  # noqa: F401  (raw Aloha jpeg decode)
    import h5py  # noqa: F401
    from PIL import Image  # noqa: F401
except ImportError as e:
    raise ImportError("This script requires h5py, Pillow, opencv-python. Install with: pip install h5py Pillow opencv-python") from e

# 与 pi05 对齐的三视角 key（LeRobot / 常见 Aloha 转换后）
IMAGE_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")

# 扩充版 [Reflection] 词库（每类 12 种），随机抽取以提升 Judge 模型对“原因/结果”的泛化
REFLECTION_VARIANTS = {
    "grasp_position_offset": [
        "[Reflection] Grasp missed due to position offset. The gripper is empty.",
        "[Reflection] Failed to grasp because of a positional error. The gripper closed on empty air.",
        "[Reflection] The gripper missed the object due to an XY offset and grabbed nothing.",
        "[Reflection] Gripper closed without the object inside due to inaccurate positioning.",
        "[Reflection] Position mismatch caused the grasp to fail. The gripper is holding nothing.",
        "[Reflection] The end-effector was off-center, resulting in an empty grasp.",
        "[Reflection] Missed the target because the gripper was not aligned properly in position.",
        "[Reflection] Grasp attempt failed; the gripper is empty due to a spatial offset.",
        "[Reflection] The gripper closed next to the object due to a position error.",
        "[Reflection] Inaccurate XY placement led to a failed, empty grasp.",
        "[Reflection] The robotic hand missed the item completely due to a translation error.",
        "[Reflection] Empty gripper detected; the grasp missed because of improper positioning.",
    ],
    "grasp_orientation_mismatch": [
        "[Reflection] Grasp failed due to incorrect orientation. The gripper collided with the object.",
        "[Reflection] The gripper hit the object instead of grasping it because of a rotation mismatch.",
        "[Reflection] Wrong wrist angle caused a collision, failing the grasp.",
        "[Reflection] Orientation error led to the gripper bumping into the object rather than holding it.",
        "[Reflection] Failed to secure the object due to an unaligned gripper posture resulting in a collision.",
        "[Reflection] The grasp was unsuccessful because the gripper's rotation did not match the object.",
        "[Reflection] Collision detected; the gripper angle was incorrect for a successful grasp.",
        "[Reflection] The end-effector collided with the item due to a mismatched orientation angle.",
        "[Reflection] Incorrect rotation caused the gripper fingers to hit the object, failing to grasp.",
        "[Reflection] Grasp failed. The gripper's posture was misaligned, causing a physical collision.",
        "[Reflection] The angle of approach was wrong, leading to a collision instead of a grip.",
        "[Reflection] Mismatched wrist orientation prevented grasping and caused a collision.",
    ],
    "premature_close": [
        "[Reflection] The gripper closed prematurely in the air before reaching the target.",
        "[Reflection] The robotic hand shut too early while still high above the object.",
        "[Reflection] Gripper closed in mid-air before descending to the correct height.",
        "[Reflection] Premature grasping action detected; the gripper closed before touching the target.",
        "[Reflection] The fingers closed too soon, leaving the gripper empty in the air.",
        "[Reflection] Failed grasp: the gripper activated its closing mechanism before reaching the item.",
        "[Reflection] The end-effector closed at the wrong height, missing the object entirely.",
        "[Reflection] Gripper shut prematurely during the descent phase.",
        "[Reflection] The closing action was triggered too early in the air, missing the target.",
        "[Reflection] Grasp failed because the gripper closed before it was lowered enough.",
        "[Reflection] The gripper clenched empty air due to closing way before reaching the object.",
        "[Reflection] An early close command caused the gripper to shut while still hovering above.",
    ],
    "grasp_slip": [
        "[Reflection] The object slipped and dropped from the gripper during transport.",
        "[Reflection] The item fell out of the gripper while being moved.",
        "[Reflection] Grasp instability caused the object to slip and drop mid-air.",
        "[Reflection] The gripper lost its hold, and the object dropped during transport.",
        "[Reflection] The grasped item slipped from the fingers and fell back down.",
        "[Reflection] Failed to maintain grip; the object dropped while the arm was moving.",
        "[Reflection] The object slipped loose from the end-effector and fell.",
        "[Reflection] Grip was lost during the movement phase, causing the item to drop.",
        "[Reflection] The robotic hand failed to hold the object securely, leading to a slip.",
        "[Reflection] The object slid out of the gripper's grasp and dropped.",
        "[Reflection] Mid-transport failure: the item slipped from the closed gripper.",
        "[Reflection] The gripper could not retain the object, which slipped and fell during transit.",
    ],
}

# 扩充版 [Correction] 词库（每类 12 种），与 Pi0.5 动作专家语义对齐，随机抽取以增强泛化
CORRECTION_VARIANTS = {
    "grasp_position_offset": [
        "[Correction] Open the gripper, adjust XY position, and retry grasping.",
        "[Correction] Release, correct XY alignment, and try to grasp again.",
        "[Correction] Open the fingers, shift XY position, and regrasp.",
        "[Correction] Unclasp, realign in the XY plane, and attempt grasp.",
        "[Correction] Open gripper, fix XY offset, and grasp once more.",
        "[Correction] Open the gripper, correct the XY error, and grab it.",
        "[Correction] Release grip, adjust the horizontal position, and retry.",
        "[Correction] Open up, align the XY coordinates, and grasp again.",
        "[Correction] Unclasp the gripper, fix planar alignment, and regrasp.",
        "[Correction] Release the object, tweak the XY position, and retry.",
        "[Correction] Open fingers, reposition along XY axes, and grab again.",
        "[Correction] Let go, shift the gripper horizontally, and regrasp.",
    ],
    "grasp_orientation_mismatch": [
        "[Correction] Open the gripper, adjust wrist rotation, and retry grasping.",
        "[Correction] Release, fix wrist angle, and try to grasp again.",
        "[Correction] Open the fingers, correct orientation, and regrasp.",
        "[Correction] Unclasp, rotate the wrist to align, and attempt grasp.",
        "[Correction] Open gripper, tune the rotation, and grasp once more.",
        "[Correction] Release grip, correct the orientation angle, and retry.",
        "[Correction] Open the gripper, align the rotation, and grab again.",
        "[Correction] Unclasp, adjust the wrist posture, and regrasp the object.",
        "[Correction] Open fingers, fix the rotation mismatch, and retry grasp.",
        "[Correction] Let go, rotate the end-effector to fit, and regrasp.",
        "[Correction] Release the object, adjust angular alignment, and grab.",
        "[Correction] Open up, correct the wrist orientation, and try again.",
    ],
    "premature_close": [
        "[Correction] Open the gripper, move down to the target, and close it.",
        "[Correction] Release, descend to the correct height, and grasp.",
        "[Correction] Open the fingers, lower to the object, and close.",
        "[Correction] Unclasp, move further down, and secure the grip.",
        "[Correction] Open gripper, drop to the target level, and shut it.",
        "[Correction] Open up, go down to the object, and close again.",
        "[Correction] Release the grip, lower the arm, and retry grasping.",
        "[Correction] Open fingers, descend to the correct Z height, and close.",
        "[Correction] Unclasp the gripper, move lower, and grab the object.",
        "[Correction] Let go, lower the end-effector to the target, and close.",
        "[Correction] Open the hand, drop down to the item, and secure grip.",
        "[Correction] Release, move the gripper downward, and shut fingers.",
    ],
    "grasp_slip": [
        "[Correction] Track the dropped object, return to it, and regrasp.",
        "[Correction] Locate the fallen object, move back, and grab it.",
        "[Correction] Find the dropped item, go to it, and secure grip.",
        "[Correction] Trace the slipped object, approach it, and regrasp.",
        "[Correction] Track the item, return to its position, and grasp again.",
        "[Correction] Detect the dropped item, move to its location, and regrasp.",
        "[Correction] Locate the slipped object, reposition above it, and grab.",
        "[Correction] Track the fallen object, go back to it, and try again.",
        "[Correction] Find where it dropped, move the gripper there, and regrasp.",
        "[Correction] Trace the object's new location, approach, and grab it.",
        "[Correction] Track down the slipped item, return to it, and grasp.",
        "[Correction] Identify the dropped item's position, move there, and regrasp.",
    ],
}


def _infer_error_type_from_subtasks(subtasks_per_frame: list[str]) -> str:
    """从 subtasks_per_frame 中第一个含 [Correction] 的句子推断 error_type。"""
    for s in subtasks_per_frame:
        if "[Correction]" not in s:
            continue
        # 取 [Correction] 后的一句，与 CORRECTION_VARIANTS 匹配
        for et, variants in CORRECTION_VARIANTS.items():
            for v in variants:
                if v.strip() in s or s.strip().startswith(v.strip()[:40]):
                    return et
        # 模糊匹配：包含关键词
        s_lower = s.lower()
        if "move down" in s_lower or "descend" in s_lower or "lower to the object" in s_lower:
            return "premature_close"
        if "xy" in s_lower or "position" in s_lower and "align" in s_lower:
            return "grasp_position_offset"
        if "wrist" in s_lower or "rotation" in s_lower or "orientation" in s_lower:
            return "grasp_orientation_mismatch"
        if "slipped" in s_lower or "dropped" in s_lower or "track" in s_lower and "object" in s_lower:
            return "grasp_slip"
    return "premature_close"


def load_episode_meta(processed_dir: Path, episode_id: int) -> Optional[dict]:
    """从 processed_data 的 episode_N/instructions.json 和可选的 metadata 读取 meta。"""
    ep_dir = processed_dir / f"episode_{episode_id}"
    instr_path = ep_dir / "instructions.json"
    if not instr_path.exists():
        return None
    with open(instr_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    phase_info = data.get("phase_info") or {}
    err_range = phase_info.get("error_attempt_range")
    if not err_range or len(err_range) < 2:
        return None
    instructions = data.get("instructions") or []
    high_instruction = instructions[0] if instructions else "Beat the block with the hammer."
    subtasks_per_frame = data.get("subtasks_per_frame") or []

    error_type = None
    meta_dir = processed_dir / "metadata"
    if meta_dir.exists():
        meta_path = meta_dir / f"episode{episode_id}_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                summary = meta.get("episode_summary") or {}
                etypes = summary.get("error_types") or []
                if etypes:
                    error_type = etypes[0]
            except Exception:
                pass
    if not error_type:
        error_type = _infer_error_type_from_subtasks(subtasks_per_frame)

    return {
        "episode_id": episode_id,
        "err_start": int(err_range[0]),
        "err_end": int(err_range[1]),
        "high_instruction": high_instruction,
        "error_type": error_type,
    }


def frame_to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """(H,W,3) 转为 uint8 [0,255]。"""
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


# Raw Aloha / dataset_recovery 的相机名 → pi05 三视角 key
_RAW_ALOHA_CAM = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}


def _read_frame_from_h5(f, frame_idx: int, image_keys: tuple[str, ...]) -> Optional[dict[str, np.ndarray]]:
    """从已打开的 h5 文件中读一帧三视角。
    支持：observations/images/<key>、observation/images/<key>；
    若不存在则尝试 raw Aloha 格式 /observation/head_camera/rgb 等（jpeg 字节解码）。
    """
    result = {}
    for key in image_keys:
        arr = None
        for key_path in (
            ("observations", "images", key),
            ("observation", "images", key),
        ):
            try:
                g = f
                for p in key_path:
                    g = g[p]
                arr = np.asarray(g[frame_idx])
                break
            except (KeyError, IndexError, TypeError):
                pass
        if arr is None:
            # Raw Aloha: /observation/head_camera/rgb 等，每帧为 jpeg 字节
            raw_cam = _RAW_ALOHA_CAM.get(key)
            if raw_cam is not None:
                try:
                    obs = f.get("observation") or f.get("/observation")
                    if obs is not None:
                        cam_g = obs.get(raw_cam)
                        if cam_g is not None:
                            rgb = cam_g.get("rgb")
                            if rgb is not None:
                                frame_data = rgb[frame_idx]
                                if hasattr(frame_data, "tobytes"):
                                    frame_data = frame_data.tobytes()
                                buf = np.frombuffer(frame_data, dtype=np.uint8)
                                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                                if img is not None:
                                    arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except (KeyError, IndexError, TypeError, OSError):
                    pass
        if arr is None:
            return None
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        result[key] = frame_to_uint8_rgb(arr)
    return result


def save_images_for_episode(
    raw_data_dir: Path,
    episode_id: int,
    err_start: int,
    err_end: int,
    image_keys: tuple[str, ...],
    out_image_dir: Path,
    source_tag: str = "",
) -> Optional[tuple[list[str], list[str]]]:
    """
    从 raw HDF5 取出 start 与 end 两帧的三视角图，保存到 out_image_dir，返回 (start_paths, end_paths)。
    顺序与 pi05 一致：cam_high, cam_left_wrist, cam_right_wrist。
    source_tag 用于多数据源时避免重名（如 clean66、rand44）。
    """
    prefix = f"{source_tag}_" if source_tag else ""
    hdf5_path = raw_data_dir / f"episode{episode_id}.hdf5"
    if not hdf5_path.exists():
        return None

    with h5py.File(hdf5_path, "r") as f:
        start_frames = _read_frame_from_h5(f, err_start, image_keys)
        end_frames = _read_frame_from_h5(f, err_end, image_keys)
    if not start_frames or not end_frames:
        return None

    out_image_dir.mkdir(parents=True, exist_ok=True)
    start_paths = []
    end_paths = []
    for key in image_keys:
        for label, frames in (("start", start_frames), ("end", end_frames)):
            sub = "start" if label == "start" else "end"
            fname = f"{prefix}ep{episode_id}_{sub}_{key}.jpg"
            path = out_image_dir / fname
            img = frames[key]
            Image.fromarray(img).save(path, quality=95)
            abs_path = str(path.resolve())
            if label == "start":
                start_paths.append(abs_path)
            else:
                end_paths.append(abs_path)
    return (start_paths, end_paths)


def build_sharegpt_item(
    high_instruction: str,
    start_image_paths: list[str],
    end_image_paths: list[str],
    error_type: str,
    use_variants: bool = True,
) -> dict:
    """一条 ShareGPT 样本：6 张图（3 start + 3 end）+ 高层指令 + 三段式回答。"""
    # 6 张图顺序：start_cam_high, start_left_wrist, start_right_wrist, end_cam_high, end_left_wrist, end_right_wrist
    all_paths = start_image_paths + end_image_paths
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
    # 随机抽取一句反思、一句纠错（扩充词库 12 种/类），提升 Judge 对原因/结果与动作意图的泛化
    default_reflections = REFLECTION_VARIANTS["premature_close"]
    default_corrections = CORRECTION_VARIANTS["premature_close"]
    reflection_list = REFLECTION_VARIANTS.get(error_type, default_reflections)
    correction_list = CORRECTION_VARIANTS.get(error_type, default_corrections)
    reflection = random.choice(reflection_list) if use_variants else reflection_list[0]
    correction = random.choice(correction_list) if use_variants else correction_list[0]
    assistant_text = (
        f"[error_type]: {error_type}\n"
        f"[reflection]: {reflection}\n"
        f"[correction]: {correction}"
    )
    return {
        "conversations": [
            {"from": "user", "value": user_text},
            {"from": "assistant", "value": assistant_text},
        ],
        "images": all_paths,
    }


def _source_tag(processed_dir: Path) -> str:
    """从 processed_data 目录名生成短标签，避免多数据源时图片重名。"""
    name = processed_dir.name
    if "demo_clean" in name:
        return "clean"
    if "demo_randomized" in name:
        return "rand"
    # 取最后一截数字或整名简写
    for part in name.split("-"):
        if part.isdigit():
            return part
    return name[:8].replace("-", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Extract error_attempt start/end frames from recovery data and build Judge Model dataset (ShareGPT)."
    )
    parser.add_argument(
        "--processed_data",
        type=str,
        nargs="+",
        default=[
            "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/beat_block_hammer-demo_clean-recovery-66",
        ],
        help="One or more processed recovery data dirs (episode_N/instructions.json + phase_info).",
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        nargs="+",
        default=[
            "/mnt/data1/liujingzhi/dataset_recovery/beat_block_hammer/demo_clean/data",
        ],
        help="Raw HDF5 dir(s), same length as --processed_data (observations/images/cam_* or /observation/.../rgb).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/mnt/data1/liujingzhi/RoboTwin/policy/pi05/judge_dataset/recovery_judge_beat_block_hammer",
        help="Output dir: images/ and recovery_judge_dataset.json.",
    )
    parser.add_argument(
        "--episode_range",
        type=str,
        default=None,
        help="e.g. 0-65 or 0,1,2 to limit episodes (applies to each source).",
    )
    parser.add_argument(
        "--no_variants",
        action="store_true",
        help="Use single correction per error_type instead of sampling from CORRECTION_VARIANTS.",
    )
    args = parser.parse_args()

    processed_dirs = [Path(p) for p in args.processed_data]
    raw_dirs = [Path(p) for p in args.raw_data_dir]
    if len(processed_dirs) != len(raw_dirs):
        raise SystemExit("--processed_data and --raw_data_dir must have the same number of arguments.")
    out_dir = Path(args.out_dir)
    for p in processed_dirs:
        if not p.exists():
            raise SystemExit(f"Processed data dir not found: {p}")
    for r in raw_dirs:
        if not r.exists():
            raise SystemExit(f"Raw data dir not found: {r}")

    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    dataset = []

    for processed_dir, raw_data_dir in zip(processed_dirs, raw_dirs):
        tag = _source_tag(processed_dir)
        print(f"\n--- Source: {processed_dir.name} (tag={tag}) ---")
        episode_dirs = sorted(
            [d for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")],
            key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
        )
        if args.episode_range:
            if "-" in args.episode_range:
                a, b = map(int, args.episode_range.split("-"))
                episode_dirs = [d for d in episode_dirs if a <= int(d.name.split("_")[1]) <= b]
            else:
                idx_set = {int(x) for x in args.episode_range.split(",")}
                episode_dirs = [d for d in episode_dirs if int(d.name.split("_")[1]) in idx_set]

        for ep_dir in episode_dirs:
            ep_id = int(ep_dir.name.split("_")[1])
            meta = load_episode_meta(processed_dir, ep_id)
            if not meta:
                print(f"  Skip episode_{ep_id}: no phase_info.error_attempt_range or instructions")
                continue
            paths = save_images_for_episode(
                raw_data_dir,
                ep_id,
                meta["err_start"],
                meta["err_end"],
                IMAGE_KEYS,
                image_dir,
                source_tag=tag,
            )
            if not paths:
                print(f"  Skip episode_{ep_id}: failed to read/save frames from {raw_data_dir}/episode{ep_id}.hdf5")
                continue
            start_paths, end_paths = paths
            item = build_sharegpt_item(
                meta["high_instruction"],
                start_paths,
                end_paths,
                meta["error_type"],
                use_variants=not args.no_variants,
            )
            dataset.append(item)
            print(f"  ✓ {tag} episode_{ep_id} error_type={meta['error_type']}")

    out_json = out_dir / "recovery_judge_dataset.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(dataset)} error samples to {out_json}")
    print("Register in LLaMA-Factory data/dataset_info.json with:")
    print(f'  "recovery_judge_beat_block_hammer": {{')
    print(f'    "file_name": "{out_json.resolve()}",')
    print('    "formatting": "sharegpt",')
    print('    "columns": { "messages": "conversations", "images": "images" }')
    print("  }")


if __name__ == "__main__":
    main()
