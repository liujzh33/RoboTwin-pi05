#!/usr/bin/env python3
"""
从HDF5文件回放episode，严格按照动作帧执行，并生成视频
参考 eval_policy_initial.py
"""
import sys
import os
import subprocess
import numpy as np
import h5py
import cv2
from pathlib import Path
from datetime import datetime

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import yaml
import importlib
import argparse
import traceback

from envs.utils.parse_hdf5 import read_hdf5, parse_img_array
from envs.utils.images_to_video import images_to_video

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")
    assert os.path.isfile(camera_config_path), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_hdf5_data(hdf5_path):
    """加载HDF5文件中的所有数据"""
    print(f"\n📂 加载HDF5文件: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        data = {}
        
        # 加载joint_action
        if 'joint_action' in f:
            joint_action = {}
            for key in ['left_arm', 'left_gripper', 'right_arm', 'right_gripper', 'vector']:
                if key in f['joint_action']:
                    joint_action[key] = f[f'joint_action/{key}'][:]
            data['joint_action'] = joint_action
        
        # 加载observation中的图像
        if 'observation' in f:
            obs_data = {}
            for cam_name in f['observation'].keys():
                if 'rgb' in f[f'observation/{cam_name}']:
                    # 解码JPEG图像
                    rgb_data = f[f'observation/{cam_name}/rgb'][:]
                    obs_data[cam_name] = {
                        'rgb': parse_img_array(rgb_data)
                    }
            data['observation'] = obs_data
        
        # 加载其他数据
        for key in f.keys():
            if key not in ['joint_action', 'observation']:
                if isinstance(f[key], h5py.Group):
                    data[key] = {}
                    for subkey in f[key].keys():
                        data[key][subkey] = f[f'{key}/{subkey}'][:]
                else:
                    data[key] = f[key][:]
        
        # 获取文件属性
        if f.attrs:
            data['_attrs'] = dict(f.attrs)
    
    print(f"✅ 加载完成，共 {len(data.get('joint_action', {}).get('left_arm', []))} 帧")
    return data


def convert_to_control_seq(hdf5_data, frame_idx, num_control_steps=10):
    """
    将HDF5中的关节位置转换为控制序列格式
    使用插值生成平滑轨迹
    """
    joint_action = hdf5_data['joint_action']
    
    # 获取当前帧和目标帧的关节位置
    if frame_idx == 0:
        # 第一帧，使用当前位置
        left_arm_start = joint_action['left_arm'][0]
        right_arm_start = joint_action['right_arm'][0]
        left_arm_target = joint_action['left_arm'][0]
        right_arm_target = joint_action['right_arm'][0]
    else:
        # 从前一帧插值到当前帧
        left_arm_start = joint_action['left_arm'][frame_idx-1]
        right_arm_start = joint_action['right_arm'][frame_idx-1]
        left_arm_target = joint_action['left_arm'][frame_idx]
        right_arm_target = joint_action['right_arm'][frame_idx]
    
    # 线性插值生成轨迹
    t = np.linspace(0, 1, num_control_steps)
    left_arm_traj = left_arm_start[None, :] + (left_arm_target - left_arm_start)[None, :] * t[:, None]
    right_arm_traj = right_arm_start[None, :] + (right_arm_target - right_arm_start)[None, :] * t[:, None]
    
    # 计算速度（差分）
    left_arm_vel = np.diff(left_arm_traj, axis=0, prepend=left_arm_traj[0:1]) * 100  # 假设100Hz
    right_arm_vel = np.diff(right_arm_traj, axis=0, prepend=right_arm_traj[0:1]) * 100
    
    # 获取gripper位置
    if frame_idx == 0:
        left_gripper_pos = joint_action['left_gripper'][0]
        right_gripper_pos = joint_action['right_gripper'][0]
    else:
        # gripper也进行插值
        left_gripper_start = joint_action['left_gripper'][frame_idx-1]
        left_gripper_target = joint_action['left_gripper'][frame_idx]
        right_gripper_start = joint_action['right_gripper'][frame_idx-1]
        right_gripper_target = joint_action['right_gripper'][frame_idx]
        left_gripper_traj = left_gripper_start + (left_gripper_target - left_gripper_start) * t
        right_gripper_traj = right_gripper_start + (right_gripper_target - right_gripper_start) * t
    
    # 构建控制序列
    control_seq = {
        "left_arm": {
            "position": left_arm_traj,  # [num_control_steps, 6]
            "velocity": left_arm_vel,  # [num_control_steps, 6]
        },
        "right_arm": {
            "position": right_arm_traj,  # [num_control_steps, 6]
            "velocity": right_arm_vel,  # [num_control_steps, 6]
        },
        "left_gripper": {
            "result": left_gripper_traj if frame_idx > 0 else np.array([left_gripper_pos] * num_control_steps),
            "num_step": num_control_steps,
            "per_step": 1,
        },
        "right_gripper": {
            "result": right_gripper_traj if frame_idx > 0 else np.array([right_gripper_pos] * num_control_steps),
            "num_step": num_control_steps,
            "per_step": 1,
        },
    }
    
    return control_seq


def replay_episode(hdf5_path, task_name, task_config, output_dir=None):
    """回放HDF5中的episode"""
    
    # 加载HDF5数据
    hdf5_data = load_hdf5_data(hdf5_path)
    joint_action = hdf5_data['joint_action']
    num_frames = len(joint_action['left_arm'])
    
    print(f"\n🎬 开始回放，共 {num_frames} 帧")
    
    # 读取任务配置
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['task_name'] = task_name
    args["task_config"] = task_config
    args["dual_arm_embodied"] = True
    
    # 设置embodiment
    embodiment_type = args.get("embodiment", ["aloha-agilex"])
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file
    
    args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    
    # 设置相机
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]
    
    # 创建输出目录
    if output_dir is None:
        output_dir = Path(hdf5_path).parent.parent / "replay_videos"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化环境
    TASK_ENV = class_decorator(task_name)
    
    # 从HDF5属性或文件名获取seed和episode信息
    seed = hdf5_data.get('_attrs', {}).get('seed', 0)
    ep_num = 0
    
    # 设置环境
    args["render_freq"] = 0  # 不渲染UI
    args["eval_video_log"] = False  # 我们自己保存视频
    args["save_data"] = False  # 不保存数据
    
    try:
        TASK_ENV.setup_demo(now_ep_num=ep_num, seed=seed, is_test=True, **args)
        
        # 存储所有视角的图像
        camera_images = {}
        if 'observation' in hdf5_data:
            for cam_name in hdf5_data['observation'].keys():
                camera_images[cam_name] = []
        
        # 逐帧执行动作
        for frame_idx in range(num_frames):
            print(f"\r执行帧 {frame_idx+1}/{num_frames}", end="", flush=True)
            
            # 转换为控制序列
            control_seq = convert_to_control_seq(hdf5_data, frame_idx, num_control_steps=10)
            
            # 执行动作
            TASK_ENV.take_dense_action(control_seq, save_freq=-1)
            
            # 在动作帧结束时获取观测（图像）- 与hdf5保存频率一致
            obs = TASK_ENV.get_obs()
            if 'observation' in obs:
                for cam_name in obs['observation'].keys():
                    if cam_name not in camera_images:
                        camera_images[cam_name] = []
                    if 'rgb' in obs['observation'][cam_name]:
                        camera_images[cam_name].append(obs['observation'][cam_name]['rgb'].copy())
        
        print(f"\n✅ 回放完成")
        
        # 保存视频
        print(f"\n🎥 保存视频...")
        for cam_name, images in camera_images.items():
            if len(images) > 0:
                # 转换为numpy数组 [N, H, W, C]
                images_array = np.stack(images, axis=0)
                video_path = output_dir / f"replay_{cam_name}.mp4"
                images_to_video(images_array, str(video_path), fps=10.0, is_rgb=True)
                print(f"  ✅ {cam_name}: {video_path}")
        
        # 同时从HDF5中提取原始图像并保存视频（用于对比）
        if 'observation' in hdf5_data:
            print(f"\n📹 从HDF5提取原始图像并保存视频...")
            for cam_name, cam_data in hdf5_data['observation'].items():
                if 'rgb' in cam_data:
                    images = cam_data['rgb']  # 已经是numpy数组
                    video_path = output_dir / f"original_{cam_name}.mp4"
                    images_to_video(images, str(video_path), fps=10.0, is_rgb=True)
                    print(f"  ✅ {cam_name} (原始): {video_path}")
        
        TASK_ENV.close_env()
        
        print(f"\n🎉 所有视频已保存到: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
        TASK_ENV.close_env()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回放HDF5 episode并生成视频")
    parser.add_argument("--hdf5_path", type=str, required=True, help="HDF5文件路径")
    parser.add_argument("--task_name", type=str, required=True, help="任务名称")
    parser.add_argument("--task_config", type=str, required=True, help="任务配置名称")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（可选）")
    
    args = parser.parse_args()
    
    replay_episode(
        hdf5_path=args.hdf5_path,
        task_name=args.task_name,
        task_config=args.task_config,
        output_dir=args.output_dir
    )

