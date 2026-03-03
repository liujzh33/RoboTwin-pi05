#!/usr/bin/env python3
"""
处理数据并自动添加子任务信息
这是 process_data.py 的增强版本，会在处理数据时自动检测夹爪状态并添加子任务
"""

import sys
import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import yaml
import json
from pathlib import Path

# 导入夹爪检测函数
from detect_gripper_phases import extract_gripper_values, detect_grasp_phase


def load_hdf5(dataset_path):
    """加载原始 HDF5 数据"""
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, image_dict


def images_encoding(imgs):
    """编码图像为 JPEG"""
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def generate_subtask_descriptions(high_level_instruction):
    """
    根据高级任务描述生成子任务
    对于 beat_block_hammer 任务，固定为两个子任务
    """
    # 可以根据高级任务描述进行更智能的生成
    # 这里使用简单的固定描述
    subtask1 = "Grab the hammer"
    subtask2 = "Strike the block with the hammer"
    
    return [subtask1, subtask2]


def data_transform(path, episode_num, save_path, gripper_threshold=0.5):
    """
    转换数据并添加子任务信息
    
    Args:
        path: 原始数据路径
        episode_num: episode 数量
        save_path: 保存路径
        gripper_threshold: 夹爪闭合阈值
    """
    begin = 0
    floders = os.listdir(path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):
        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        
        # 读取原始指令
        with open(instruction_data_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict[desc_type]
        
        # 加载 HDF5 数据以检测阶段
        hdf5_path = os.path.join(path, "data", f"episode{i}.hdf5")
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = load_hdf5(hdf5_path)
        
        # 处理数据并检测阶段
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []

        last_state = None
        for j in range(0, left_gripper_all.shape[0]):
            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            state = np.array(left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper])
            state = state.astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)
                left_arm_dim.append(left_arm.shape[0])
                right_arm_dim.append(right_arm.shape[0])

        # 检测夹爪阶段
        qpos_array = np.array(qpos)
        left_arm_dim_val = left_arm_dim[0] if left_arm_dim else 7
        right_arm_dim_val = right_arm_dim[0] if right_arm_dim else 7
        
        left_gripper_vals, right_gripper_vals = extract_gripper_values(
            qpos_array, left_arm_dim_val, right_arm_dim_val
        )
        grasp_end_idx = detect_grasp_phase(left_gripper_vals, right_gripper_vals, gripper_threshold)
        
        # 生成子任务描述
        subtasks_list = []
        for high_level in instructions:
            subtasks = generate_subtask_descriptions(high_level)
            subtasks_list.append(subtasks)
        
        # 创建包含子任务信息的 instructions.json
        save_instructions_json = {
            "instructions": instructions,
            "subtasks": subtasks_list,
            "phase_info": {
                "grasp_end_idx": int(grasp_end_idx),
                "total_steps": int(len(qpos)),
                "phase1_steps": int(grasp_end_idx),
                "phase2_steps": int(len(qpos) - grasp_end_idx),
                "threshold": gripper_threshold
            }
        }

        os.makedirs(os.path.join(save_path, f"episode_{i}"), exist_ok=True)

        with open(
                os.path.join(os.path.join(save_path, f"episode_{i}"), "instructions.json"),
                "w",
        ) as f:
            json.dump(save_instructions_json, f, indent=2)

        # 保存 HDF5 数据
        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

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

        begin += 1
        print(f"Processed episode {i}: Phase 1 (Grasp) = {grasp_end_idx} steps, Phase 2 (Strike) = {len(qpos)-grasp_end_idx} steps")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process episodes with subtask detection.")
    parser.add_argument(
        "task_name",
        type=str,
        default="beat_block_hammer",
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=50,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Gripper closure threshold (default: 0.5)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    dataset_root = "/mnt/data1/liujingzhi/dataset"
    load_dir = os.path.join(dataset_root, str(task_name), str(setting))
    begin = 0
    print(f'Reading data from: {load_dir}')

    target_dir = f"processed_data/{task_name}-{setting}-{expert_data_num}"
    begin = data_transform(
        load_dir,
        expert_data_num,
        target_dir,
        gripper_threshold=args.gripper_threshold,
    )
    print(f"\nProcessing complete! Data saved to: {target_dir}")

