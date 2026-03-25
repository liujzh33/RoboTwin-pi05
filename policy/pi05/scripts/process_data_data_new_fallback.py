"""
Process raw RoboTwin data_new tasks into the pi0/pi05 processed_data format.
This variant keeps the older fallback: if episode_instructions are empty,
it will use task_prompt.full_description as a single instruction; if that is
also missing, it will skip the episode.

Usage (from policy/pi05):
    python scripts/process_data_data_new_fallback.py <task_dir_name> [--data-root ../../data/data_new]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np


CAMERA_RESIZE = (640, 480)


@dataclass
class EpisodePaths:
    hdf5_path: str
    metadata_path: str


def find_episode_count(task_path: str, max_episodes: int | None) -> int:
    stats_path = os.path.join(task_path, "collection_stats.json")
    data_dir = os.path.join(task_path, "data")

    if os.path.isfile(stats_path):
        with open(stats_path, "r") as f:
            total = int(json.load(f).get("total_episodes", 0))
            if total > 0:
                return total if max_episodes is None else min(total, max_episodes)

    h5_files = [
        name
        for name in os.listdir(data_dir)
        if name.startswith("episode") and name.endswith(".hdf5")
    ]
    count = len(h5_files)
    return count if max_episodes is None else min(count, max_episodes)


def build_episode_paths(task_path: str, episode_idx: int) -> EpisodePaths:
    return EpisodePaths(
        hdf5_path=os.path.join(task_path, "data", f"episode{episode_idx}.hdf5"),
        metadata_path=os.path.join(
            task_path, "metadata", f"episode{episode_idx}_metadata.json"
        ),
    )


def load_instructions_with_fallback(metadata_path: str, desc_type: str) -> Tuple[List[str], bool]:
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    instr_dict = meta.get("episode_instructions", {}) or {}

    # Prefer requested desc_type if non-empty
    instructions = instr_dict.get(desc_type) or []
    if instructions:
        return instructions, False

    # Fallback to any non-empty list
    for val in instr_dict.values():
        if val:
            return val, False

    # Fallback to task_prompt.full_description
    prompt = meta.get("task_prompt", {}).get("full_description")
    if prompt:
        return [prompt], False

    # Nothing found
    return [], True


def detect_cameras(root: h5py.File) -> List[str]:
    if "/observation" in root:
        return list(root["/observation"].keys())
    return []


def load_hdf5(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        if "/joint_action" not in root or "/observation" not in root:
            raise KeyError(f"Unexpected hdf5 layout in {dataset_path}")

        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )

        image_dict: Dict[str, np.ndarray] = {}
        for cam_name in detect_cameras(root):
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs: List[np.ndarray]) -> Tuple[List[bytes], int]:
    encoded = []
    max_len = 0
    for img in imgs:
        success, enc = cv2.imencode(".jpg", img)
        if not success:
            raise RuntimeError("cv2.imencode failed")
        jpeg = enc.tobytes()
        encoded.append(jpeg)
        max_len = max(max_len, len(jpeg))
    return encoded, max_len


def process_episode(
    paths: EpisodePaths,
    instructions: List[str],
    save_root: str,
    episode_idx: int,
) -> None:
    os.makedirs(os.path.join(save_root, f"episode_{episode_idx}"), exist_ok=True)
    with open(
        os.path.join(save_root, f"episode_{episode_idx}", "instructions.json"), "w"
    ) as f:
        json.dump({"instructions": instructions}, f, indent=2)

    left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = load_hdf5(
        paths.hdf5_path
    )

    qpos = []
    actions = []
    cam_high = []
    cam_right_wrist = []
    cam_left_wrist = []
    left_arm_dim = []
    right_arm_dim = []

    for j in range(left_gripper_all.shape[0]):
        left_gripper, left_arm, right_gripper, right_arm = (
            left_gripper_all[j],
            left_arm_all[j],
            right_gripper_all[j],
            right_arm_all[j],
        )
        state = np.array(
            left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper],
            dtype=np.float32,
        )

        if j != left_gripper_all.shape[0] - 1:
            qpos.append(state)
            camera_high_bits = image_dict["head_camera"][j]
            camera_high = cv2.imdecode(
                np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR
            )
            camera_high_resized = cv2.resize(camera_high, CAMERA_RESIZE)
            cam_high.append(camera_high_resized)

            camera_right_wrist_bits = image_dict["right_camera"][j]
            camera_right_wrist = cv2.imdecode(
                np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR
            )
            camera_right_wrist_resized = cv2.resize(camera_right_wrist, CAMERA_RESIZE)
            cam_right_wrist.append(camera_right_wrist_resized)

            camera_left_wrist_bits = image_dict["left_camera"][j]
            camera_left_wrist = cv2.imdecode(
                np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR
            )
            camera_left_wrist_resized = cv2.resize(camera_left_wrist, CAMERA_RESIZE)
            cam_left_wrist.append(camera_left_wrist_resized)

        if j != 0:
            actions.append(state)
            left_arm_dim.append(left_arm.shape[0])
            right_arm_dim.append(right_arm.shape[0])

    hdf5path = os.path.join(save_root, f"episode_{episode_idx}", f"episode_{episode_idx}.hdf5")
    with h5py.File(hdf5path, "w") as f:
        f.create_dataset("action", data=np.array(actions))
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=np.array(qpos))
        obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
        obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
        image = obs.create_group("images")
        cam_high_enc, len_high = images_encoding(cam_high)
        cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
        cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
        image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
        image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
        image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")


def main():
    parser = argparse.ArgumentParser(description="Process data_new task with instruction fallback.")
    parser.add_argument(
        "task_dir",
        type=str,
        help="Task directory name under data_root (e.g., beat_block_hammer-premature_close:0.8:0.3)",
    )
    parser.add_argument("--data-root", type=str, default="../../data/data_new")
    parser.add_argument("--output-root", type=str, default="processed_data")
    parser.add_argument("--desc-type", type=str, default="seen", help="Instruction type: seen or unseen")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap on episodes to process")
    args = parser.parse_args()

    task_path = os.path.join(args.data_root, args.task_dir)
    if not os.path.isdir(task_path):
        raise FileNotFoundError(f"Task path not found: {task_path}")

    episode_num = find_episode_count(task_path, args.max_episodes)
    save_path = os.path.join(args.output_root, f"{args.task_dir}-fallback-{episode_num}")
    os.makedirs(save_path, exist_ok=True)

    missing_episodes: List[int] = []

    for i in range(episode_num):
        paths = build_episode_paths(task_path, i)
        if not os.path.isfile(paths.hdf5_path):
            print(f"[warn] skip episode {i}: missing hdf5 {paths.hdf5_path}")
            continue
        if not os.path.isfile(paths.metadata_path):
            print(f"[warn] skip episode {i}: missing metadata {paths.metadata_path}")
            continue

        instructions, missing = load_instructions_with_fallback(paths.metadata_path, args.desc_type)
        if missing:
            print(f"[warn] skip episode {i}: no instructions in {paths.metadata_path}")
            missing_episodes.append(i)
            continue

        process_episode(paths, instructions, save_path, i)
        print(f"processed episode {i}")

    print(f"done. saved to {save_path}")
    if missing_episodes:
        print(f"[summary] missing instructions even after fallback: {missing_episodes}")


if __name__ == "__main__":
    main()
