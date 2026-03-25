import sys
import os
import subprocess
import argparse
import importlib
import re

from pathlib import Path
from datetime import datetime
from collections import deque
import traceback

import yaml
import numpy as np

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import *


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except Exception:
        raise SystemExit(f"No Task env: {task_name}")
    return env_instance


def eval_function_decorator(policy_name: str, model_name: str):
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)


def _get_camera_config(camera_type: str):
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")
    assert os.path.isfile(camera_config_path), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def _get_embodiment_config(robot_file: str):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def _prepare_pi05_checkpoint_from_policy_checkpoint_dir(usr_args: dict) -> None:
    """
    Force PI05 to load from `policy/checkpoint` by ensuring:
      - policy/checkpoint/model.safetensors exists (convert from ckpt_step_0030000.pt)
      - norm_stats.json is loaded from the specified path
    Then set:
      - usr_args["use_torch_checkpoint"] = True
      - usr_args["torch_checkpoint_dir"] = <abs path policy/checkpoint>
      - usr_args["norm_stats_path"] = <abs path to norm_stats.json>
    """
    ckpt_dir = os.path.abspath(os.path.join("policy", "checkpoint"))
    os.makedirs(ckpt_dir, exist_ok=True)

    usr_args["use_torch_checkpoint"] = True
    usr_args["torch_checkpoint_dir"] = ckpt_dir

    # Force use specific norm_stats.json path
    norm_stats_path = os.path.abspath(
        os.path.join("policy", "checkpoint", "pi0.5_robotwin2", "assets", "pi0.5_clean_randomize_joint_training", "norm_stats.json")
    )
    if not os.path.isfile(norm_stats_path):
        raise FileNotFoundError(
            f"Required norm_stats.json not found at: {norm_stats_path}\n"
            "Please ensure the file exists at the specified path."
        )
    usr_args["norm_stats_path"] = norm_stats_path
    print(f"\033[94m[PI05] Using norm_stats: {norm_stats_path}\033[0m")

    # Force use ckpt_step_0030000.pt for weights
    weight_path = os.path.join(ckpt_dir, "model.safetensors")
    src_pt = os.path.join(ckpt_dir, "ckpt_step_0030000.pt")
    
    if not os.path.isfile(src_pt):
        raise FileNotFoundError(
            f"Required checkpoint file not found: {src_pt}\n"
            "Please ensure ckpt_step_0030000.pt exists in policy/checkpoint/"
        )

    print(f"\033[94m[PI05] Converting torch checkpoint to safetensors: {src_pt} -> {weight_path}\033[0m")

    import torch
    from safetensors.torch import save_file

    obj = torch.load(src_pt, map_location="cpu")
    state_dict = None
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "policy", "module", "params", "weights"):
            if k in obj and isinstance(obj[k], dict):
                state_dict = obj[k]
                break
        if state_dict is None and all(hasattr(v, "shape") for v in obj.values()):
            state_dict = obj

    if state_dict is None:
        raise RuntimeError(
            f"Cannot extract state_dict from {src_pt}. "
            "If this ckpt is not a plain model state_dict, you need to export PI05 to safetensors."
        )

    cleaned = {}
    for k, v in state_dict.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        cleaned[nk] = v

    # safetensors refuses tensors that share underlying storage (tied weights).
    # Break shared storage by cloning duplicates (values stay identical).
    seen = {}
    for k, v in list(cleaned.items()):
        try:
            # Prefer untyped_storage() for new torch; fall back to storage()
            ptr = v.untyped_storage().data_ptr()  # type: ignore[attr-defined]
        except Exception:
            ptr = v.storage().data_ptr()  # type: ignore[union-attr]
        if ptr in seen:
            cleaned[k] = v.clone()
        else:
            seen[ptr] = k

    save_file(cleaned, weight_path)
    print(f"\033[92m[PI05] Saved {weight_path}\033[0m")


def main(usr_args: dict):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args.get("ckpt_setting", "default")
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args.get("instruction_type", "unseen")

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    # embodiment
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(emb_type):
        robot_file = _embodiment_types[emb_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])

    # output dir
    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    video_size = None
    if args.get("eval_video_log", False):
        camera_config = _get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        args["eval_video_save_dir"] = save_dir

    # print config summary
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))
    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    # env + model args
    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    # PI05: if user explicitly enables torch checkpoint mode, load weights from policy/checkpoint.
    # Otherwise (use_torch_checkpoint=False), PI05 will load the original JAX checkpoint from:
    #   policy/pi05/checkpoints/<train_config>/<model_name>/<checkpoint_id>
    if policy_name == "pi05" and usr_args.get("use_torch_checkpoint", False):
        _prepare_pi05_checkpoint_from_policy_checkpoint_dir(usr_args)

    seed = usr_args["seed"]
    st_seed = 100000 * (1 + seed)

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(
        task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
        test_num=usr_args.get("test_num", 30),
                                   video_size=video_size,
        instruction_type=instruction_type,
    )

    file_path = os.path.join(save_dir, "_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(str(suc_num) + "\n")
    print(f"Data has been saved to {file_path}")


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    """
    Pure rollout evaluation (NO recovery logic).
    """
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        # Enable HDF5 caching during rollout when eval video log is enabled
        original_save_data_rollout = getattr(TASK_ENV, "save_data", False)
        original_save_dir_rollout = getattr(TASK_ENV, "save_dir", None)
        original_save_freq_rollout = getattr(TASK_ENV, "save_freq", None)
        eval_video_log_enabled = (TASK_ENV.eval_video_path is not None) or args.get("eval_video_log", False)
        if eval_video_log_enabled:
            TASK_ENV.save_data = True
            rollout_data_dir = os.path.join(TASK_ENV.eval_video_path, "data")
            os.makedirs(rollout_data_dir, exist_ok=True)
            TASK_ENV.save_dir = TASK_ENV.eval_video_path
            TASK_ENV.save_freq = 4
            if not hasattr(TASK_ENV, "folder_path") or TASK_ENV.folder_path is None:
                TASK_ENV.folder_path = {"cache": f"{TASK_ENV.eval_video_path}/.cache/episode{TASK_ENV.test_num}/"}
            TASK_ENV.FRAME_IDX = 0
        
        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()
            
        # Save rollout HDF5/video cache
        if eval_video_log_enabled:
            try:
                rollout_episode_name = f"episode{TASK_ENV.test_num}"
                TASK_ENV.merge_pkl_to_hdf5_video(custom_video_name=rollout_episode_name)
                try:
                    TASK_ENV.remove_data_cache()
                except Exception:
                    pass
                print(f"\033[94m[Rollout] Saved rollout HDF5 data: {rollout_episode_name}\033[0m")
            except Exception as e:
                print(f"\033[93m[Rollout] Warning: Failed to save rollout HDF5 data: {e}\033[0m")
            finally:
                TASK_ENV.save_data = original_save_data_rollout
                TASK_ENV.save_dir = original_save_dir_rollout
                TASK_ENV.save_freq = original_save_freq_rollout

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()
        TASK_ENV.test_num += 1
        now_seed += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except Exception:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST

    Sapien_TEST()
    usr_args = parse_args_and_config()
    main(usr_args)


