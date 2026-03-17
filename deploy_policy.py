import numpy as np
import torch
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (
        usr_args["train_config_name"],
        usr_args["model_name"],
        usr_args["checkpoint_id"],
        usr_args["pi0_step"],
    )
    # When `use_torch_checkpoint` is True, PI0 will load weights from
    # `policy/pi05/checkpoint_torch` (PyTorch checkpoint with model.safetensors) by default;
    # if `torch_checkpoint_dir` is provided (e.g. set by eval_policy_ppo.py), it will load from there.
    # otherwise it uses the default JAX-style checkpoint layout.
    use_torch_checkpoint = usr_args.get("use_torch_checkpoint", False)
    torch_checkpoint_dir = usr_args.get("torch_checkpoint_dir", None)
    norm_stats_path = usr_args.get("norm_stats_path", None)
    return PI0(
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        use_torch_checkpoint=use_torch_checkpoint,
        torch_checkpoint_dir=torch_checkpoint_dir,
        norm_stats_path=norm_stats_path,
    )


def eval(TASK_ENV, model, observation, open_gripper=False, left_arm_dim=6, right_arm_dim=6):
    """
    Evaluate policy with optional gripper recovery test.
    
    Args:
        TASK_ENV: Task environment
        model: Policy model
        observation: Current observation
        open_gripper: If True, force open gripper when closed to test recovery
        left_arm_dim: Dimension of left arm joints (default 6)
        right_arm_dim: Dimension of right arm joints (default 6)
    """
    # Initialize episode-level flag for gripper intervention (only once per episode)
    if not hasattr(eval, '_gripper_intervened'):
        eval._gripper_intervened = False
        eval._gripper_hold_frames = 0
        eval._gripper_hold_target = 0

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]
    
    # Convert to numpy array and make writable if needed
    if not isinstance(actions, np.ndarray):
        actions = np.array(actions)
    actions = np.array(actions, copy=True)  # Ensure writable copy
    
    # Calculate gripper indices
    left_gripper_idx = left_arm_dim  # 左夹爪索引
    right_gripper_idx = left_arm_dim + 1 + right_arm_dim  # 右夹爪索引
    
    # Check if we need to force open gripper (only once per episode)
    if open_gripper and not eval._gripper_intervened:
        # Check first action in the batch for closed gripper
        first_action = actions[0] if len(actions) > 0 else None
        if first_action is not None:
            left_gripper_val = first_action[left_gripper_idx]
            right_gripper_val = first_action[right_gripper_idx]
            
            # Check if any gripper is closed (value < 0.2)
            if left_gripper_val < 0.2 or right_gripper_val < 0.2:
                eval._gripper_intervened = True
                # Hold gripper open for 30 frames
                eval._gripper_hold_target = 30
                eval._gripper_hold_frames = 0
                print(f"\033[93m[Gripper Recovery Test] Detected closed gripper! "
                      f"Left: {left_gripper_val:.3f}, Right: {right_gripper_val:.3f}. "
                      f"Force opening for {eval._gripper_hold_target} frames.\033[0m")
    
    # Apply gripper intervention if active
    if open_gripper and eval._gripper_intervened and eval._gripper_hold_frames < eval._gripper_hold_target:
        # Calculate how many frames we still need to intervene
        remaining_frames = eval._gripper_hold_target - eval._gripper_hold_frames
        # Only modify the actions we need (not the entire batch)
        num_actions_to_modify = min(len(actions), remaining_frames)
        
        if num_actions_to_modify > 0:
            # Force open grippers only in the actions we need to modify
            actions[:num_actions_to_modify, left_gripper_idx] = 1.0  # Force open left gripper
            actions[:num_actions_to_modify, right_gripper_idx] = 1.0  # Force open right gripper
            eval._gripper_hold_frames += num_actions_to_modify
            
            if eval._gripper_hold_frames >= eval._gripper_hold_target:
                print(f"\033[93m[Gripper Recovery Test] Gripper intervention completed after exactly {eval._gripper_hold_frames} frames.\033[0m")

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
    # Reset gripper intervention flags when model is reset (new episode)
    if hasattr(eval, '_gripper_intervened'):
        eval._gripper_intervened = False
        eval._gripper_hold_frames = 0
        eval._gripper_hold_target = 0
