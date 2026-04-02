#!/usr/bin/env python3
"""
将 dataset_recovery 下的错误恢复数据转为 processed_data 格式，并保留 metadata。

- 数据源：/mnt/data1/liujingzhi/dataset_recovery/{task_name}/{setting}/
  需包含 data/episode{i}.hdf5、metadata/episode{i}_metadata.json（无 instructions 目录时从 metadata 取）
- 输出：processed_data/{task_name}-{setting}-recovery-{expert_data_num}/
  含 episode_0, episode_1, ... 及完整 metadata 目录副本
- instructions：与原有逻辑一致，从 metadata 的 episode_instructions.seen 生成 instructions.json，
  保证后续子任务添加（process_data_generic_v1）使用相同结构。
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

import h5py
import numpy as np
import cv2

# 与 process_data.py 相同的 HDF5 读写逻辑
def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        left_gripper = root["/joint_action/left_gripper"][()]
        left_arm = root["/joint_action/left_arm"][()]
        right_gripper = root["/joint_action/right_gripper"][()]
        right_arm = root["/joint_action/right_arm"][()]
        image_dict = {}
        for cam_name in root["/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    encode_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    padded_data = [enc.ljust(max_len, b"\0") for enc in encode_data]
    return padded_data, max_len


def get_instructions_from_metadata(metadata_path: Path) -> list:
    """
    从 metadata JSON 中取出 instructions，与原有「instructions 列表」逻辑一致。
    优先使用 episode_instructions.seen；若无则用 task_prompt.full_description 单条。
    """
    if not metadata_path.exists():
        return ["Default instruction"]
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # 与 process_data 的 desc_type="seen" 一致：使用 seen 列表
    ep_instr = meta.get("episode_instructions") or {}
    seen = ep_instr.get("seen")
    if isinstance(seen, list) and len(seen) > 0:
        return seen
    # 兜底：任务描述
    task_prompt = meta.get("task_prompt") or {}
    desc = task_prompt.get("full_description")
    if desc:
        return [desc]
    return ["Default instruction"]


def data_transform_recovery(load_dir: Path, metadata_dir: Path, episode_num: int, save_path: Path) -> int:
    """
    将 load_dir 下的 data/episode{i}.hdf5 转为 processed 的 episode_{i}/episode_{i}.hdf5，
    instructions 从 metadata_dir/episode{i}_metadata.json 的 episode_instructions.seen 生成。
    """
    data_dir = load_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    save_path.mkdir(parents=True, exist_ok=True)
    processed = 0

    for i in range(episode_num):
        hdf5_src = data_dir / f"episode{i}.hdf5"
        if not hdf5_src.exists():
            print(f"Skip episode {i}: {hdf5_src} not found")
            continue

        metadata_src = metadata_dir / f"episode{i}_metadata.json"
        instructions = get_instructions_from_metadata(metadata_src)
        save_instructions_json = {"instructions": instructions}

        ep_save = save_path / f"episode_{i}"
        ep_save.mkdir(parents=True, exist_ok=True)
        with open(ep_save / "instructions.json", "w", encoding="utf-8") as f:
            json.dump(save_instructions_json, f, indent=2)

        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = load_hdf5(str(hdf5_src))

        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        T = left_gripper_all.shape[0]
        for j in range(T):
            left_gripper = left_gripper_all[j]
            left_arm = left_arm_all[j]
            right_gripper = right_gripper_all[j]
            right_arm = right_arm_all[j]
            state = np.array(
                left_arm.tolist() + [float(left_gripper)] + right_arm.tolist() + [float(right_gripper)],
                dtype=np.float32,
            )

            if j != T - 1:
                qpos.append(state)
                for cam_key, out_list, target_size in [
                    ("head_camera", cam_high, (640, 480)),
                    ("right_camera", cam_right_wrist, (640, 480)),
                    ("left_camera", cam_left_wrist, (640, 480)),
                ]:
                    bits = image_dict[cam_key][j]
                    img = cv2.imdecode(np.frombuffer(bits, np.uint8), cv2.IMREAD_COLOR)
                    out_list.append(cv2.resize(img, target_size))

            if j != 0:
                actions.append(state)
                left_arm_dim.append(len(left_arm))
                right_arm_dim.append(len(right_arm))

        cam_high_enc, len_high = images_encoding(cam_high)
        cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
        cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)

        hdf5path = ep_save / f"episode_{i}.hdf5"
        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            image = obs.create_group("images")
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        processed += 1
        print(f"Process episode {i} success!")

    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset_recovery to processed_data and preserve metadata."
    )
    parser.add_argument("task_name", type=str, default="beat_block_hammer", help="Task name (e.g. beat_block_hammer)")
    parser.add_argument("setting", type=str, help="Setting (e.g. demo_clean)")
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=0,
        help="Number of episodes to process; 0 = auto-detect from data folder",
    )
    parser.add_argument(
        "--dataset_recovery_root",
        type=str,
        default="/mnt/data1/liujingzhi/dataset_recovery",
        help="Root of dataset_recovery",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    recovery_root = Path(args.dataset_recovery_root)

    load_dir = recovery_root / task_name / setting
    metadata_dir = load_dir / "metadata"
    data_dir = load_dir / "data"
    if not load_dir.exists():
        print(f"Error: Load dir not found: {load_dir}")
        return 1
    if not data_dir.exists():
        print(f"Error: Data dir not found: {data_dir}")
        return 1
    if not metadata_dir.exists():
        print(f"Warning: Metadata dir not found: {metadata_dir} (instructions will use fallback)")

    # 0 表示自动检测：按 data 目录下 episode*.hdf5 数量
    expert_data_num = args.expert_data_num
    if expert_data_num <= 0:
        expert_data_num = len(sorted(data_dir.glob("episode*.hdf5")))
        print(f"Auto-detected {expert_data_num} episodes in {data_dir}")

    # 输出目录固定在项目根下 processed_data，命名含 recovery 避免覆盖
    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / "processed_data" / f"{task_name}-{setting}-recovery-{expert_data_num}"
    if not target_dir.parent.exists():
        target_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Read data from: {load_dir}")
    print(f"Output to: {target_dir}")

    n = data_transform_recovery(load_dir, metadata_dir, expert_data_num, target_dir)
    print(f"Processed {n} episodes.")

    # 保留 metadata：整目录拷贝到 processed 下
    if metadata_dir.exists():
        dest_metadata = target_dir / "metadata"
        if dest_metadata.exists():
            shutil.rmtree(dest_metadata)
        shutil.copytree(metadata_dir, dest_metadata)
        print(f"Copied metadata to {dest_metadata}")
    else:
        print("No metadata to copy.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
