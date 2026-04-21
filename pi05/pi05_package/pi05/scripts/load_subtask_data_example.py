#!/usr/bin/env python3
"""
示例：如何加载包含子任务的数据用于训练

这个脚本展示了如何从 processed_data 目录加载数据，并使用子任务信息。
"""

import os
import json
import h5py
import numpy as np
from pathlib import Path

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.transforms import LoadSubtaskFromInstructions, TokenizeHighLowPrompt
from openpi.models.model import Observation


def load_episode_data(episode_dir):
    """
    从 episode 目录加载数据
    
    Args:
        episode_dir: episode 目录路径（如 processed_data/.../episode_0/）
    
    Returns:
        dict: 包含 images, state, actions, instructions, subtasks 的字典
    """
    episode_path = Path(episode_dir)
    
    # 1. 加载 instructions.json
    instructions_path = episode_path / "instructions.json"
    with open(instructions_path, 'r') as f:
        instructions_data = json.load(f)
    
    instructions = instructions_data['instructions']
    subtasks = instructions_data.get('subtasks', [])
    phase_info = instructions_data.get('phase_info', {})
    
    # 2. 加载 HDF5 数据
    hdf5_files = list(episode_path.glob("episode_*.hdf5"))
    if len(hdf5_files) == 0:
        raise ValueError(f"No HDF5 file found in {episode_dir}")
    
    hdf5_path = hdf5_files[0]
    with h5py.File(hdf5_path, 'r') as f:
        actions = f['action'][()]
        qpos = f['observations/qpos'][()]
        # 加载图像（需要解码 JPEG）
        images = {}
        for key in ['cam_high', 'cam_right_wrist', 'cam_left_wrist']:
            if key in f['observations/images']:
                # 图像是 JPEG 编码的，需要解码
                image_data = f[f'observations/images/{key}'][()]
                # 这里简化处理，实际需要解码 JPEG
                images[key] = image_data
    
    # 3. 构建 state（从 qpos）
    state = qpos  # qpos 已经包含了 state 信息
    
    return {
        'instructions': instructions,
        'subtasks': subtasks,
        'phase_info': phase_info,
        'images': images,
        'state': state,
        'actions': actions,
    }


def prepare_training_sample(episode_dir, tokenizer):
    """
    准备一个训练样本，包含子任务信息
    
    Args:
        episode_dir: episode 目录路径
        tokenizer: PaligemmaTokenizer 实例
    
    Returns:
        Observation: 准备好的 Observation 对象
    """
    # 1. 加载原始数据
    data = load_episode_data(episode_dir)
    
    # 2. 加载子任务（随机选择一个 instruction 和对应的 subtask）
    load_subtask = LoadSubtaskFromInstructions(use_first_subtask=False)
    data = load_subtask(data)
    # 现在 data 包含: high_prompt, low_prompt
    
    # 3. Tokenize 高级任务和子任务
    tokenize_transform = TokenizeHighLowPrompt(tokenizer=tokenizer)
    data = tokenize_transform(data)
    # 现在 data 包含: tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask
    
    # 4. 构建 Observation
    # 注意：这里简化了图像处理，实际需要解码 JPEG 并转换为正确的格式
    observation = Observation.from_dict({
        'image': {
            'base_0_rgb': data['images'].get('cam_high', np.zeros((1, 224, 224, 3))),
            'left_wrist_0_rgb': data['images'].get('cam_left_wrist', np.zeros((1, 224, 224, 3))),
            'right_wrist_0_rgb': data['images'].get('cam_right_wrist', np.zeros((1, 224, 224, 3))),
        },
        'image_mask': {
            'base_0_rgb': np.ones(1, dtype=bool),
            'left_wrist_0_rgb': np.ones(1, dtype=bool),
            'right_wrist_0_rgb': np.ones(1, dtype=bool),
        },
        'state': data['state'][:1],  # 取第一个时间步
        'tokenized_prompt': data['tokenized_prompt'][np.newaxis, :],
        'tokenized_prompt_mask': data['tokenized_prompt_mask'][np.newaxis, :],
        'token_ar_mask': data['token_ar_mask'][np.newaxis, :],
        'token_loss_mask': data['token_loss_mask'][np.newaxis, :],
    })
    
    return observation, data['actions'][:1]  # 返回第一个 action


def example_usage():
    """使用示例"""
    # 初始化 tokenizer
    tokenizer = PaligemmaTokenizer(max_len=200)
    
    # 加载一个 episode
    episode_dir = "processed_data/beat_block_hammer-aloha-agilex_randomized_500-200/episode_0"
    
    try:
        observation, actions = prepare_training_sample(episode_dir, tokenizer)
        
        print("Successfully loaded training sample!")
        print(f"Observation keys: {list(observation.to_dict().keys())}")
        print(f"Tokenized prompt shape: {observation.tokenized_prompt.shape}")
        print(f"Token loss mask shape: {observation.token_loss_mask.shape}")
        print(f"Token loss mask (first 50): {observation.token_loss_mask[0, :50]}")
        print(f"Actions shape: {actions.shape}")
        
        # 验证 loss_mask
        loss_mask_sum = np.sum(observation.token_loss_mask)
        total_tokens = np.sum(observation.tokenized_prompt_mask)
        print(f"\nLoss mask statistics:")
        print(f"  Total valid tokens: {total_tokens}")
        print(f"  Tokens with loss: {loss_mask_sum}")
        print(f"  Loss ratio: {loss_mask_sum / total_tokens * 100:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    example_usage()

