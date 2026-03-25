# import sys
# import os
# import subprocess

# sys.path.append("./")
# sys.path.append(f"./policy")
# sys.path.append("./description/utils")
# from envs import CONFIGS_PATH
# from envs.utils.create_actor import UnStableError

# import numpy as np
# from pathlib import Path
# from collections import deque
# import traceback

# import yaml
# from datetime import datetime
# import importlib
# import argparse
# import pdb

# from generate_episode_instructions import *

# current_file_path = os.path.abspath(__file__)
# parent_directory = os.path.dirname(current_file_path)


# # 用于导入任务环境，env_class 是任务环境的类，env_instance 是任务环境的实例
# def class_decorator(task_name):
#     envs_module = importlib.import_module(f"envs.{task_name}")
#     try:
#         env_class = getattr(envs_module, task_name)
#         env_instance = env_class()
#     except:
#         raise SystemExit("No Task")
#     return env_instance

# # 导入模型，policy_name和model_name的区别是model_name是模型的具体函数，而policy_name是模型的模块，返回的是模型的具体函数
# def eval_function_decorator(policy_name, model_name):
#     try:
#         policy_model = importlib.import_module(policy_name)
#         return getattr(policy_model, model_name)
#     except ImportError as e:
#         raise e


# def get_camera_config(camera_type):
#     camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

#     assert os.path.isfile(camera_config_path), "task config file is missing"

#     with open(camera_config_path, "r", encoding="utf-8") as f:
#         args = yaml.load(f.read(), Loader=yaml.FullLoader)

#     assert camera_type in args, f"camera {camera_type} is not defined"
#     return args[camera_type]


# def get_embodiment_config(robot_file):
#     robot_config_file = os.path.join(robot_file, "config.yml")
#     with open(robot_config_file, "r", encoding="utf-8") as f:
#         embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
#     return embodiment_args


# def main(usr_args):
#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     task_name = usr_args["task_name"]
#     task_config = usr_args["task_config"]
#     ckpt_setting = usr_args["ckpt_setting"]
#     # checkpoint_num = usr_args['checkpoint_num']
#     policy_name = usr_args["policy_name"]
#     instruction_type = usr_args["instruction_type"]
#     save_dir = None
#     video_save_dir = None
#     video_size = None

#     get_model = eval_function_decorator(policy_name, "get_model")

#     with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
#         args = yaml.load(f.read(), Loader=yaml.FullLoader)

#     args['task_name'] = task_name
#     args["task_config"] = task_config
#     args["ckpt_setting"] = ckpt_setting

#     embodiment_type = args.get("embodiment")
#     embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

#     with open(embodiment_config_path, "r", encoding="utf-8") as f:
#         _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

#     def get_embodiment_file(embodiment_type):
#         robot_file = _embodiment_types[embodiment_type]["file_path"]
#         if robot_file is None:
#             raise "No embodiment files"
#         return robot_file

#     with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
#         _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

#     head_camera_type = args["camera"]["head_camera_type"]
#     args["head_camera_h"] = _camera_config[head_camera_type]["h"]
#     args["head_camera_w"] = _camera_config[head_camera_type]["w"]

#     if len(embodiment_type) == 1:
#         args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
#         args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
#         args["dual_arm_embodied"] = True
#     elif len(embodiment_type) == 3:
#         args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
#         args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
#         args["embodiment_dis"] = embodiment_type[2]
#         args["dual_arm_embodied"] = False
#     else:
#         raise "embodiment items should be 1 or 3"

#     args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
#     args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

#     if len(embodiment_type) == 1:
#         embodiment_name = str(embodiment_type[0])
#     else:
#         embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

#     save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
#     save_dir.mkdir(parents=True, exist_ok=True)

#     if args["eval_video_log"]:
#         video_save_dir = save_dir
#         camera_config = get_camera_config(args["camera"]["head_camera_type"])
#         video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
#         video_save_dir.mkdir(parents=True, exist_ok=True)
#         args["eval_video_save_dir"] = video_save_dir

#     # output camera config
#     print("============= Config =============\n")
#     print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
#     print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
#     if args["domain_randomization"]["random_background"]:
#         print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
#     print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
#     if args["domain_randomization"]["random_light"]:
#         print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
#     print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
#     print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

#     print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
#           str(args["camera"]["collect_head_camera"]))
#     print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
#           str(args["camera"]["collect_wrist_camera"]))
#     print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
#     print("\n==================================")

#     TASK_ENV = class_decorator(args["task_name"])
#     args["policy_name"] = policy_name
#     usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
#     usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

#     seed = usr_args["seed"]

#     st_seed = 100000 * (1 + seed)
#     suc_nums = []
#     test_num = 100
#     topk = 1

#     model = get_model(usr_args)
#     st_seed, suc_num = eval_policy(task_name,
#                                    TASK_ENV,
#                                    args,
#                                    model,
#                                    st_seed,
#                                    test_num=test_num,
#                                    video_size=video_size,
#                                    instruction_type=instruction_type)
#     suc_nums.append(suc_num)

#     topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

#     file_path = os.path.join(save_dir, f"_result.txt")
#     with open(file_path, "w") as file:
#         file.write(f"Timestamp: {current_time}\n\n")
#         file.write(f"Instruction Type: {instruction_type}\n\n")
#         # file.write(str(task_reward) + '\n')
#         file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

#     print(f"Data has been saved to {file_path}")
#     # return task_reward


# def eval_policy(task_name,
#                 TASK_ENV,
#                 args,
#                 model,
#                 st_seed,
#                 test_num=100,
#                 video_size=None,
#                 instruction_type=None):
#     print(f"\033[34mTask Name: {args['task_name']}\033[0m")
#     print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

#     expert_check = True
#     TASK_ENV.suc = 0
#     TASK_ENV.test_num = 0

#     now_id = 0
#     succ_seed = 0
#     suc_test_seed_list = []

#     policy_name = args["policy_name"]
#     eval_func = eval_function_decorator(policy_name, "eval")
#     reset_func = eval_function_decorator(policy_name, "reset_model")

#     now_seed = st_seed
#     task_total_reward = 0
#     clear_cache_freq = args["clear_cache_freq"]

#     args["eval_mode"] = True

#     # Per-variant recovery statistics
#     variant_success_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
#     variant_total_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}

#     # ---------------- Recovery helpers for beat_block_hammer ----------------
#     # We only enable expert-based recovery for the beat_block_hammer task.
#     def _snapshot_env(env):
#         """Lightweight snapshot of the current env state for recovery branches."""
#         snap = {
#             "take_action_cnt": env.take_action_cnt,
#             "left_jointstate": env.robot.get_left_arm_jointState(),
#             "right_jointstate": env.robot.get_right_arm_jointState(),
#             "hammer_pose": env.hammer.get_pose() if hasattr(env, "hammer") else None,
#             "block_pose": env.block.get_pose() if hasattr(env, "block") else None,
#         }
#         return snap

#     def _restore_env(env, snapshot):
#         """Restore env to the snapshot state (same scene, rolled back configuration)."""
#         # Restore robot joints and grippers
#         left = snapshot["left_jointstate"]
#         right = snapshot["right_jointstate"]
#         if left is not None and right is not None:
#             # left arm
#             env.robot.set_arm_joints(left[:-1], np.zeros_like(left[:-1]), "left")
#             env.robot.set_gripper(left[-1], "left")
#             # right arm
#             env.robot.set_arm_joints(right[:-1], np.zeros_like(right[:-1]), "right")
#             env.robot.set_gripper(right[-1], "right")

#         # Restore object poses if available
#         # Note: Actor objects use actor.actor.set_pose() where actor.actor is the SAPIEN Entity
#         if snapshot["hammer_pose"] is not None and hasattr(env, "hammer"):
#             env.hammer.actor.set_pose(snapshot["hammer_pose"])
#         if snapshot["block_pose"] is not None and hasattr(env, "block"):
#             env.block.actor.set_pose(snapshot["block_pose"])

#         # Reset planning / success flags
#         env.plan_success = True
#         env.eval_success = False

#         # One simulation step to flush the new state
#         env.scene.step()
#         env._update_render()

#     def _try_expert_recovery(
#         env,
#         recovery_step=200,
#         episode_index: int = 0,
#         variant_success_counts: dict[int, int] | None = None,
#         variant_total_counts: dict[int, int] | None = None,
#     ):
#         """
#         At a given step in the rollout, fork three expert branches:
#           1) grasp + lift + place
#           2) lift + place
#           3) place only
#         and check whether any branch can succeed from the current failed state.
#         """
#         if task_name != "beat_block_hammer":
#             return False  # only support this task for now

#         if not hasattr(env, "expert_recovery_sequence"):
#             return False

#         # Take a snapshot at the current ACT rollout state
#         snapshot = _snapshot_env(env)

#         print(f"\n\033[93m[Recovery] Trying expert recovery branches at step {env.take_action_cnt}\033[0m")
#         for variant in (1, 2, 3):
#             print(f"\033[93m[Recovery] Variant {variant}: "
#                   f"{'grasp+lift+place' if variant == 1 else 'lift+place' if variant == 2 else 'place only'}\033[0m")

#             # Count how many times each variant is attempted
#             if variant_total_counts is not None:
#                 variant_total_counts[variant] = variant_total_counts.get(variant, 0) + 1

#             _restore_env(env, snapshot)
#             # Configure per-variant data/video saving
#             original_save_data = getattr(env, "save_data", False)
#             original_save_dir = getattr(env, "save_dir", None)
#             original_save_freq = getattr(env, "save_freq", None)
#             # Reset folder_path so that _take_picture will create a new cache directory
#             if hasattr(env, "folder_path"):
#                 env.folder_path = None

#             # Save recovery branches under a dedicated directory
#             env.save_data = True
#             env.save_dir = os.path.join(
#                 "eval_result",
#                 task_name,
#                 args["policy_name"],
#                 args["task_config"],
#                 args["ckpt_setting"],
#                 "recovery",
#                 f"episode{episode_index}_variant{variant}",
#             )
#             # Reset frame index so that each variant starts its own cache/episode
#             env.FRAME_IDX = 0
#             # Use a relatively sparse save frequency during recovery so that
#             # Use the same frame saving frequency as the main evaluation video,
#             # so that the expert recovery clip plays at a similar perceived speed.
#             # This means we save one frame every control step.
#             env.save_freq = 1

#             success = env.expert_recovery_sequence(variant)

#             # Merge cached pkl files into an hdf5 + mp4 video for this variant
#             variant_video_path = None
#             try:
#                 custom_name = f"episode{episode_index}_variant{variant}"
#                 env.merge_pkl_to_hdf5_video(custom_video_name=custom_name)
#                 env.remove_data_cache()
#                 # The merged variant video is saved under:
#                 #   {env.save_dir}/video/{custom_name}.mp4
#                 variant_video_path = os.path.join(env.save_dir, "video", f"{custom_name}.mp4")
#             except Exception as e:
#                 print(f"\033[91m[Recovery] Warning: failed to save recovery data for variant {variant}: {e}\033[0m")

#             # Restore original save flags/dirs
#             env.save_data = original_save_data
#             env.save_dir = original_save_dir
#             env.save_freq = original_save_freq

#             if success:
#                 if variant_success_counts is not None:
#                     variant_success_counts[variant] = variant_success_counts.get(variant, 0) + 1
#                 print(f"\033[92m[Recovery] Success with variant {variant}\033[0m")

#                 # Store concatenation info for later (after main video is fully written)
#                 # We'll do the actual concatenation after _del_eval_video_ffmpeg() is called
#                 if env.eval_video_path is not None and variant_video_path is not None:
#                     # Store in a way that can be accessed later
#                     if not hasattr(env, '_recovery_concat_info'):
#                         env._recovery_concat_info = []
#                     env._recovery_concat_info.append({
#                         'episode_index': episode_index,
#                         'variant': variant,
#                         'variant_video_path': variant_video_path,
#                     })
#                 return True

#         print("\033[91m[Recovery] All expert variants failed\033[0m")
#         return False
#     # -----------------------------------------------------------------------

#     while succ_seed < test_num:
#         render_freq = args["render_freq"]
#         args["render_freq"] = 0

#         if expert_check:
#             try:
#                 TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
#                 episode_info = TASK_ENV.play_once()
#                 TASK_ENV.close_env()
#             except UnStableError as e:
#                 # print(" -------------")
#                 # print("Error: ", e)
#                 # print(" -------------")
#                 TASK_ENV.close_env()
#                 now_seed += 1
#                 args["render_freq"] = render_freq
#                 continue
#             except Exception as e:
#                 # stack_trace = traceback.format_exc()
#                 # print(" -------------")
#                 # print("Error: ", e)
#                 # print(" -------------")
#                 TASK_ENV.close_env()
#                 now_seed += 1
#                 args["render_freq"] = render_freq
#                 print("error occurs !")
#                 continue

#         if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
#             succ_seed += 1
#             suc_test_seed_list.append(now_seed)
#         else:
#             now_seed += 1
#             args["render_freq"] = render_freq
#             continue

#         args["render_freq"] = render_freq

#         TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
#         episode_info_list = [episode_info["info"]]
#         results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
#         instruction = np.random.choice(results[0][instruction_type])
#         TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

#         if TASK_ENV.eval_video_path is not None:
#             ffmpeg = subprocess.Popen(
#                 [
#                     "ffmpeg",
#                     "-y",
#                     "-loglevel",
#                     "error",
#                     "-f",
#                     "rawvideo",
#                     "-pixel_format",
#                     "rgb24",
#                     "-video_size",
#                     video_size,
#                     "-framerate",
#                     "10",
#                     "-i",
#                     "-",
#                     "-pix_fmt",
#                     "yuv420p",
#                     "-vcodec",
#                     "libx264",
#                     "-crf",
#                     "23",
#                     f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
#                 ],
#                 stdin=subprocess.PIPE,
#             )
#             TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

#         succ = False
#         reset_func(model)
#         recovery_attempted = False
#         recovery_step = 200  # step at which we trigger recovery branches
#         while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
#             observation = TASK_ENV.get_obs()
#             eval_func(TASK_ENV, model, observation)

#             # ---------------- Expert-based error recovery ----------------
#             if (not recovery_attempted
#                     and TASK_ENV.take_action_cnt >= recovery_step
#                     and (not TASK_ENV.eval_success)):
#                 recovery_attempted = True
#                 recovered = _try_expert_recovery(
#                     TASK_ENV,
#                     recovery_step=recovery_step,
#                     episode_index=TASK_ENV.test_num,
#                     variant_success_counts=variant_success_counts,
#                     variant_total_counts=variant_total_counts,
#                 )
#                 if recovered:
#                     succ = True
#                     break
#             # -------------------------------------------------------------

#             if TASK_ENV.eval_success:
#                 succ = True
#                 break
#         # task_total_reward += TASK_ENV.episode_score
#         if TASK_ENV.eval_video_path is not None:
#             TASK_ENV._del_eval_video_ffmpeg()

#         # Now that the main video is fully written, concatenate with successful recovery videos
#         if succ and hasattr(TASK_ENV, '_recovery_concat_info') and TASK_ENV._recovery_concat_info:
#             for concat_info in TASK_ENV._recovery_concat_info:
#                 episode_idx = concat_info['episode_index']
#                 variant = concat_info['variant']
#                 variant_video_path = concat_info['variant_video_path']
                
#                 try:
#                     main_video = os.path.join(TASK_ENV.eval_video_path, f"episode{episode_idx}.mp4")
                    
#                     # Wait a bit to ensure file is fully flushed
#                     import time
#                     max_wait = 5
#                     waited = 0
#                     while waited < max_wait and (not os.path.exists(main_video) or os.path.getsize(main_video) == 0):
#                         time.sleep(0.1)
#                         waited += 0.1
                    
#                     if os.path.exists(main_video) and os.path.exists(variant_video_path):
#                         # Verify main video is valid (has moov atom)
#                         try:
#                             check_result = subprocess.run(
#                                 ["ffprobe", "-v", "error", main_video],
#                                 capture_output=True,
#                                 timeout=2,
#                             )
#                             if check_result.returncode != 0:
#                                 print(f"\033[93m[Recovery] Main video {main_video} appears invalid, skipping concatenation\033[0m")
#                                 continue
#                         except Exception:
#                             pass  # ffprobe not available or failed, try anyway
                        
#                         full_video = os.path.join(
#                             TASK_ENV.eval_video_path,
#                             f"episode{episode_idx}_variant{variant}_full.mp4",
#                         )
#                         concat_list_path = os.path.join(
#                             TASK_ENV.eval_video_path,
#                             f"episode{episode_idx}_variant{variant}_concat.txt",
#                         )
                        
#                         # Use absolute paths in concat file
#                         with open(concat_list_path, "w", encoding="utf-8") as f:
#                             f.write(f"file '{os.path.abspath(main_video)}'\n")
#                             f.write(f"file '{os.path.abspath(variant_video_path)}'\n")
                        
#                         # Run ffmpeg concatenation.
#                         # NOTE:
#                         #   - We re-encode instead of using `-c copy` to avoid timestamp
#                         #     issues when the two input clips have different FPS.
#                         #   - We force a fixed FPS (10) so that the concatenated video
#                         #     has a uniform playback speed without long "frozen" segments.
#                         result = subprocess.run(
#                             [
#                                 "ffmpeg",
#                                 "-y",
#                                 "-loglevel",
#                                 "error",
#                                 "-f",
#                                 "concat",
#                                 "-safe",
#                                 "0",
#                                 "-i",
#                                 concat_list_path,
#                                 "-r",
#                                 "10",
#                                 "-pix_fmt",
#                                 "yuv420p",
#                                 "-vcodec",
#                                 "libx264",
#                                 "-crf",
#                                 "23",
#                                 full_video,
#                             ],
#                             capture_output=True,
#                             timeout=60,
#                         )
                        
#                         if result.returncode == 0 and os.path.exists(full_video) and os.path.getsize(full_video) > 0:
#                             print(f"\033[92m[Recovery] Full episode video saved to {full_video}\033[0m")
#                             # Clean up concat list file
#                             try:
#                                 os.remove(concat_list_path)
#                             except:
#                                 pass
#                         else:
#                             error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
#                             print(f"\033[91m[Recovery] Failed to concatenate videos: {error_msg}\033[0m")
#                     else:
#                         print(
#                             f"\033[93m[Recovery] Skip concatenation: main or variant video not found "
#                             f"(main={os.path.exists(main_video)}, variant={os.path.exists(variant_video_path)})\033[0m"
#                         )
#                 except Exception as e:
#                     print(f"\033[91m[Recovery] Warning: Failed to concatenate videos: {e}\033[0m")
            
#             # Clear the concat info after processing
#             TASK_ENV._recovery_concat_info = []

#         if succ:
#             TASK_ENV.suc += 1
#             print("\033[92mSuccess!\033[0m")
#         else:
#             print("\033[91mFail!\033[0m")

#         now_id += 1
#         TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

#         if TASK_ENV.render_freq:
#             TASK_ENV.viewer.close()

#         TASK_ENV.test_num += 1

#         print(
#             f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
#             f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
#         )
#         # TASK_ENV._take_picture()
#         now_seed += 1

#     # Print per-variant recovery statistics
#     print("\n\033[94m[Recovery Summary per variant]\033[0m")
#     for v in (1, 2, 3):
#         total = variant_total_counts.get(v, 0)
#         succ_v = variant_success_counts.get(v, 0)
#         if total == 0:
#             rate = 0.0
#         else:
#             rate = round(succ_v / total * 100.0, 1)
#         label = "grasp+lift+place" if v == 1 else "lift+place" if v == 2 else "place only"
#         print(f"  Variant {v} ({label}): {succ_v}/{total} => {rate}%")

#     return now_seed, TASK_ENV.suc


# def parse_args_and_config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True)
#     parser.add_argument("--overrides", nargs=argparse.REMAINDER)
#     args = parser.parse_args()

#     with open(args.config, "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)

#     # Parse overrides
#     def parse_override_pairs(pairs):
#         override_dict = {}
#         for i in range(0, len(pairs), 2):
#             key = pairs[i].lstrip("--")
#             value = pairs[i + 1]
#             try:
#                 value = eval(value)
#             except:
#                 pass
#             override_dict[key] = value
#         return override_dict

#     if args.overrides:
#         overrides = parse_override_pairs(args.overrides)
#         config.update(overrides)

#     return config


# if __name__ == "__main__":
#     from test_render import Sapien_TEST
#     Sapien_TEST()

#     usr_args = parse_args_and_config()

#     main(usr_args)
import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *
from script.recovery_utils import (
    load_task_frame_threshold,
    count_play_once_stages,
    get_all_actor_poses,
    restore_all_actor_poses,
)

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


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

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


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
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
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = 300
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    # ================ Universal Recovery System ================
    # Load frame threshold for multi-task support
    threshold_type = usr_args.get("recovery_threshold_type", "max")  # max, p99, p95, p90, mean
    frame_threshold = load_task_frame_threshold(task_name, threshold_type)
    if frame_threshold is not None:
        print(f"\033[94m[Recovery] Loaded frame threshold for {task_name}: {frame_threshold} ({threshold_type})\033[0m")
    else:
        print(f"\033[93m[Recovery] Warning: No frame threshold found for {task_name}, using step-based trigger\033[0m")
    
    # Option to force open grippers at recovery start
    # This is useful when rollout ends with closed grippers, which would prevent proper grasping
    recovery_force_open_grippers = usr_args.get("recovery_force_open_grippers", True)  # Default: True
    if recovery_force_open_grippers:
        print(f"\033[94m[Recovery] Will force open both grippers at recovery start\033[0m")
    
    # Auto-detect number of stages in play_once
    num_stages = count_play_once_stages(TASK_ENV.__class__)
    # If task implements get_recovery_steps, use it (backward compatibility)
    if hasattr(TASK_ENV, 'get_recovery_steps'):
        try:
            num_stages = TASK_ENV.get_recovery_steps()
            print(f"\033[94m[Recovery] Task implements get_recovery_steps(): {num_stages} stages\033[0m")
        except:
            pass
    print(f"\033[94m[Recovery] Detected {num_stages} stages for {task_name}\033[0m")
    
    # Per-variant recovery statistics (dynamically sized based on task)
    variant_success_counts: dict[int, int] = {}
    variant_total_counts: dict[int, int] = {}

    def _snapshot_env(env):
        """Universal snapshot of the current env state for recovery branches."""
        snap = {
            "take_action_cnt": env.take_action_cnt,
            "left_jointstate": env.robot.get_left_arm_jointState(),
            "right_jointstate": env.robot.get_right_arm_jointState(),
        }
        
        # Automatically detect and save all actor poses (multi-task support)
        actor_poses = get_all_actor_poses(env)
        snap.update(actor_poses)
        
        return snap

    def _restore_env(env, snapshot):
        """Universal restore env to the snapshot state."""
        # Restore robot joints and grippers
        left = snapshot["left_jointstate"]
        right = snapshot["right_jointstate"]
        if left is not None and right is not None:
            env.robot.set_arm_joints(left[:-1], np.zeros_like(left[:-1]), "left")
            env.robot.set_gripper(left[-1], "left")
            env.robot.set_arm_joints(right[:-1], np.zeros_like(right[:-1]), "right")
            env.robot.set_gripper(right[-1], "right")

        # Automatically restore all actor poses (multi-task support)
        actor_poses = {k: v for k, v in snapshot.items() 
                      if k.endswith('_pose') and k not in ['left_jointstate', 'right_jointstate']}
        restore_all_actor_poses(env, actor_poses)

        # Reset planning / success flags
        env.plan_success = True
        env.eval_success = False

        # One simulation step to flush the new state
        env.scene.step()
        env._update_render()

    def _should_trigger_recovery(env, task_name, frame_threshold=None, min_step=50, check_window=20):
        """
        Intelligently determine when to trigger recovery based on:
        1. Minimum step count (avoid too early)
        2. Frame count threshold (NEW - multi-task support) - PRIORITY
        3. Progress stagnation (end-effector position not changing)
        4. Step limit threshold (95% of max steps) - FALLBACK ONLY
        """
        # Condition 1: At least min_step steps
        if env.take_action_cnt < min_step:
            return False
        
        # Condition 2: Frame count threshold check (NEW - multi-task support) - HIGHEST PRIORITY
        # Convert steps to video frames and compare with threshold
        if frame_threshold is not None:
            eval_video_freq = 4  # 每4步保存一帧（与eval_policy.py中的eval_video_freq一致）
            current_frames = env.take_action_cnt // eval_video_freq
            if current_frames > frame_threshold:
                print(f"\n\033[93m[Recovery Trigger] Frame count {current_frames} > threshold {frame_threshold}, triggering recovery\033[0m")
                return True
        
        # Condition 3: Progress stagnation check
        if not hasattr(env, '_recent_ee_positions'):
            env._recent_ee_positions = deque(maxlen=check_window)
        
        # Record current end-effector positions
        left_ee_pos = env.robot.get_left_tcp_pose()[:3]
        right_ee_pos = env.robot.get_right_tcp_pose()[:3]
        env._recent_ee_positions.append((left_ee_pos, right_ee_pos))
        
        # Check if positions are stagnating
        if len(env._recent_ee_positions) >= check_window:
            recent_left = [pos[0] for pos in env._recent_ee_positions]
            recent_right = [pos[1] for pos in env._recent_ee_positions]
            
            # Check Z-axis variance (movement in vertical direction)
            left_z_variance = np.var([pos[2] for pos in recent_left])
            right_z_variance = np.var([pos[2] for pos in recent_right])
            
            if left_z_variance < 0.001 and right_z_variance < 0.001:
                # Both arms are stuck
                return True
        
        # Condition 4: Over 95% of max steps without success (fallback only - only if no frame threshold)
        # This is a safety net for tasks without frame threshold data
        if frame_threshold is None and env.take_action_cnt >= env.step_lim * 0.95 and not env.eval_success:
            print(f"\n\033[93m[Recovery Trigger] Step count {env.take_action_cnt} >= {env.step_lim * 0.95:.0f} (95% of max), triggering recovery (fallback)\033[0m")
            return True
        
        return False

    def _try_multi_env_recovery(
        task_name,
        original_env,
        snapshot,
        episode_index: int,
        num_stages: int,
        args: dict,
        recovery_seed: int,
        recovery_force_open_grippers: bool = True,
        variant_success_counts: dict[int, int] | None = None,
        variant_total_counts: dict[int, int] | None = None,
    ):
        """
        Multi-environment recovery system for multi-task support.
        
        Creates multiple environment instances, each skipping different stages.
        This ensures that no matter where rollout stops, one environment will
        start from the correct point.
        """
        print(f"\n\033[93m[Recovery] Starting multi-environment recovery with {num_stages} stages\033[0m")
        
        recovery_results = []
        
        # Execute recovery for each variant (create and execute serially to save GPU memory)
        # We create environments one at a time, execute, then close to free GPU resources
        # This prevents "cannot create buffer" errors when creating multiple environments
        for variant in range(1, num_stages + 1):
            skip_stages = variant - 1
            print(f"\n\033[93m[Recovery] Variant {variant}: Skipping first {skip_stages} stage(s)\033[0m")
            
            # Create recovery environment for this variant (serially, not in parallel)
            env = None
            try:
                print(f"[Recovery] Creating recovery environment {variant}/{num_stages}")
                envs_module = importlib.import_module(f"envs.{task_name}")
                env_class = getattr(envs_module, task_name)
                env = env_class()
                
                # IMPORTANT: Disable video saving during preparation phase
                # These are background operations and should NOT be saved to video frames
                original_save_data_prep = getattr(env, "save_data", False)
                original_save_freq_prep = getattr(env, "save_freq", None)
                env.save_data = False  # Disable saving during preparation
                env.save_freq = None   # Disable saving during preparation
                
                # ========== PREPARATION PHASE (NOT saved to video) ==========
                # CRITICAL: For recovery, we must NOT do any robot initialization movements
                # We use skip_robot_init=True to completely skip:
                #   - robot.move_to_homestate() - would move robot to initial position (SKIPPED)
                #   - together_open_gripper() - would open grippers (SKIPPED, we restore from snapshot)
                #   - robot.set_origin_endpose() - would set origin pose (SKIPPED, not needed)
                # 
                # Only basic setup is done: table, robot model loading, camera, actors
                # Robot and object poses are completely determined by _restore_env(snapshot) below
                
                # Call _init_task_env_() for basic setup (table, robot model, camera, actors)
                # IMPORTANT: skip_robot_init=True ensures NO robot initialization movements
                #   - NO robot.move_to_homestate()
                #   - NO together_open_gripper()
                #   - NO robot.set_origin_endpose()
                # Robot and object poses are ONLY set by _restore_env(snapshot) below
                env._init_task_env_(
                    table_xy_bias=args.get("table_xy_bias", [0, 0]),
                    table_height_bias=args.get("table_height_bias", 0),
                    skip_robot_init=True,
                    seed=recovery_seed,
                    task_name=args.get("task_name"),
                    task_config=args.get("task_config"),
                    save_path=args.get("save_path", "data"),
                    now_ep_num=episode_index,
                    render_freq=args.get("render_freq", 10),
                    data_type=args.get("data_type", None),
                    save_data=False,  # Keep disabled during prep
                    dual_arm=args.get("dual_arm", True),
                    eval_mode=True,
                    domain_randomization=args.get("domain_randomization", {}),
                    need_plan=args.get("need_plan", True),
                    left_joint_path=args.get("left_joint_path", []),
                    right_joint_path=args.get("right_joint_path", []),
                    eval_video_save_dir=args.get("eval_video_save_dir", None),
                    save_freq=args.get("save_freq"),
                    **{k: v for k, v in args.items() if k not in [
                        'table_xy_bias', 'table_height_bias', 'seed', 'task_name', 'task_config',
                        'save_path', 'now_ep_num', 'render_freq', 'data_type', 'save_data',
                        'dual_arm', 'eval_mode', 'domain_randomization', 'need_plan',
                        'left_joint_path', 'right_joint_path', 'eval_video_save_dir', 'save_freq'
                    ]}
                )
                
                # CRITICAL: Restore robot and object states from snapshot
                # IMPORTANT: set_arm_joints() uses set_drive_target(), which requires multiple physics steps
                # to move from initial position (URDF default) to target position (snapshot).
                # We must execute enough steps with save_data=False to ensure robot fully reaches target
                # before enabling video saving, otherwise we'll see "return to initial pose" frames.
                print(f"\033[94m[Recovery] [Variant {variant}] Restoring robot and object states from snapshot\033[0m")
                _restore_env(env, snapshot)
                
                # Ensure all actor poses are correctly applied (load_actors() reset them)
                actor_poses = {k: v for k, v in snapshot.items() 
                              if k.endswith('_pose') and k not in ['left_jointstate', 'right_jointstate']}
                if actor_poses:
                    print(f"\033[94m[Recovery] [Variant {variant}] Re-applying {len(actor_poses)} actor pose(s) after restore\033[0m")
                    restore_all_actor_poses(env, actor_poses)
                
                # CRITICAL: Execute many physics steps to ensure robot fully reaches target position
                # set_arm_joints() sets drive targets, requiring multiple steps to converge
                # All these steps happen with save_data=False, so no frames are saved
                # This prevents "return to initial pose" frames from appearing in recovery videos
                print(f"\033[94m[Recovery] [Variant {variant}] Stabilizing robot position (save_data=False, no frames saved)\033[0m")
                for _ in range(100):  # Increased from 10 to 100 to ensure full convergence
                    env.scene.step()
                    if env.render_freq:
                        env._update_render()
                
                # Note: Grippers will be opened just before play_once() if recovery_force_open_grippers is True
                # This ensures we maintain rollout's arm state but have open grippers for grasping
                # ========== END PREPARATION PHASE ==========
            except Exception as e:
                print(f"\033[91m[Recovery] Failed to create recovery environment {variant}: {e}\033[0m")
                recovery_results.append({
                    'variant': variant,
                    'success': False,
                    'variant_video_path': None,
                })
                continue
            
            # Count attempts
            if variant_total_counts is not None:
                variant_total_counts[variant] = variant_total_counts.get(variant, 0) + 1
            
            # Calculate skip_stages: variant 1 skips 0, variant 2 skips 1, etc.
            # This ensures:
            #   variant 1: executes all stages (skip_stages=0)
            #   variant 2: skips first stage, executes remaining (skip_stages=1)
            #   variant 3: skips first 2 stages, executes remaining (skip_stages=2)
            #   etc.
            skip_stages = variant - 1
            print(f"\033[94m[Recovery] [Variant {variant}] Will skip first {skip_stages} stage(s), execute remaining {num_stages - skip_stages} stage(s)\033[0m")
            
            # Open grippers BEFORE executing play_once (still in preparation phase, NOT saved)
            # This ensures we maintain rollout's arm state but have open grippers for grasping
            # Note: This is still part of preparation, so save_data is still False
            if recovery_force_open_grippers:
                try:
                    # Get current gripper states before opening
                    left_gripper_before = env.robot.get_left_gripper_val()
                    right_gripper_before = env.robot.get_right_gripper_val()
                    
                    # Force open both grippers
                    env.robot.set_gripper(1.0, "left")   # 1.0 = fully open
                    env.robot.set_gripper(1.0, "right")  # 1.0 = fully open
                    
                    # Step simulation multiple times to ensure grippers fully open
                    for _ in range(20):  # Step 20 times to ensure grippers open
                        env.scene.step()
                        if env.render_freq:
                            env._update_render()
                    
                    # Verify grippers are actually open
                    left_gripper_after = env.robot.get_left_gripper_val()
                    right_gripper_after = env.robot.get_right_gripper_val()
                    
                    print(f"\033[94m[Recovery] [Variant {variant}] Opened both grippers before play_once:")
                    print(f"  Left:  {left_gripper_before:.3f} -> {left_gripper_after:.3f} {'✓' if left_gripper_after > 0.8 else '✗ (may not be fully open)'}")
                    print(f"  Right: {right_gripper_before:.3f} -> {right_gripper_after:.3f} {'✓' if right_gripper_after > 0.8 else '✗ (may not be fully open)'}\033[0m")
                    
                    # If grippers didn't open fully, try again
                    if left_gripper_after < 0.8 or right_gripper_after < 0.8:
                        print(f"\033[93m[Recovery] [Variant {variant}] Warning: Grippers not fully open, retrying...\033[0m")
                        env.robot.set_gripper(1.0, "left")
                        env.robot.set_gripper(1.0, "right")
                        for _ in range(20):
                            env.scene.step()
                            if env.render_freq:
                                env._update_render()
                        left_gripper_final = env.robot.get_left_gripper_val()
                        right_gripper_final = env.robot.get_right_gripper_val()
                        print(f"\033[94m[Recovery] [Variant {variant}] After retry - Left: {left_gripper_final:.3f}, Right: {right_gripper_final:.3f}\033[0m")
                except Exception as e:
                    print(f"\033[93m[Recovery] [Variant {variant}] Warning: Failed to open grippers: {e}\033[0m")
            
            # ========== NOW ENABLE VIDEO SAVING (before executing play_once) ==========
            # Configure data saving - ONLY NOW enable saving, after all preparation is done
            original_save_data = getattr(env, "save_data", False)
            original_save_dir = getattr(env, "save_dir", None)
            original_save_freq = getattr(env, "save_freq", None)
            if hasattr(env, "folder_path"):
                env.folder_path = None
            
            # CRITICAL: Disable back_to_origin() in recovery environment
            # Recovery should start from rollout's failed state, not from initial position
            # back_to_origin() calls should be silently ignored (no-op) to prevent
            # "return to initial position" frames from appearing in recovery videos
            original_back_to_origin = env.back_to_origin
            def back_to_origin_noop(arm_tag):
                """No-op version of back_to_origin for recovery - prevents return to initial position"""
                print(f"\033[93m[Recovery] [Variant {variant}] Skipping back_to_origin({arm_tag}) - recovery starts from rollout state\033[0m")
                # Return empty action sequence (no-op)
                return arm_tag, []
            env.back_to_origin = back_to_origin_noop
            
            env.save_data = True  # Enable saving ONLY for play_once() execution
            env.save_dir = os.path.join(
                "eval_result",
                task_name,
                args["policy_name"],
                args["task_config"],
                args["ckpt_setting"],
                "recovery",
                f"episode{episode_index}_variant{variant}",
            )
            env.FRAME_IDX = 0  # Reset frame index - video starts from play_once()
            env.save_freq = 8
            
            # Execute recovery
            success = False
            variant_video_path = None
            custom_name = f"episode{episode_index}_variant{variant}"
            
            # Wrap grasp_actor to ensure gripper is open before grasping
            # This is critical because grasp_actor() doesn't open gripper itself,
            # and if gripper is closed, it will push the object away
            original_grasp_actor = env.grasp_actor
            def grasp_actor_wrapper(actor, arm_tag, **kwargs):
                # Check current gripper state
                # Convert arm_tag to string safely (handle both ArmTag and string)
                arm_tag_str = str(arm_tag) if hasattr(arm_tag, '__str__') else arm_tag
                if arm_tag_str == "left":
                    current_gripper_val = env.robot.get_left_gripper_val()
                else:
                    current_gripper_val = env.robot.get_right_gripper_val()
                
                # Call original grasp_actor to get action sequence
                result = original_grasp_actor(actor, arm_tag, **kwargs)
                if result is None or result[0] is None:
                    return result
                
                returned_arm_tag, action_list = result
                
                # Import Action class
                from envs.utils.action import Action
                
                # CRITICAL FIX: Always ensure gripper is open before and during move actions
                # Even if gripper is currently open, we need to maintain it open during moves
                # because physical simulation may cause gripper to close passively
                new_action_list = []
                last_action_was_move = False
                
                for i, action in enumerate(action_list):
                    # If this is a move action and gripper is not open, or if previous action was move
                    # we need to ensure gripper stays open
                    if action.action == "move":
                        # Before first move action, ensure gripper is open
                        if i == 0 or (i > 0 and action_list[i-1].action != "gripper"):
                            if current_gripper_val < 0.8:
                                # Insert open action before this move
                                open_action = Action(arm_tag, "open", target_gripper_pos=1.0)
                                new_action_list.append(open_action)
                                print(f"\033[94m[Recovery] [Variant {variant}] Inserted open gripper action before move action {i+1}\033[0m")
                            else:
                                # Gripper is open, but insert a maintain-open action to prevent passive closing
                                # during physical simulation
                                maintain_open_action = Action(arm_tag, "open", target_gripper_pos=1.0)
                                new_action_list.append(maintain_open_action)
                                print(f"\033[94m[Recovery] [Variant {variant}] Inserted maintain-open gripper action before move action {i+1} (prevent passive closing)\033[0m")
                        new_action_list.append(action)
                        last_action_was_move = True
                    elif action.action == "gripper":
                        # This is a gripper action (open/close), just add it
                        new_action_list.append(action)
                        last_action_was_move = False
                        # Update current_gripper_val for next iteration
                        if action.target_gripper_pos is not None:
                            current_gripper_val = action.target_gripper_pos
                    else:
                        new_action_list.append(action)
                        last_action_was_move = False
                
                # If no actions were added (shouldn't happen), return original
                if not new_action_list:
                    return result
                
                print(f"\033[94m[Recovery] [Variant {variant}] Modified action sequence: {len(action_list)} -> {len(new_action_list)} actions\033[0m")
                return returned_arm_tag, new_action_list
            
            # Replace grasp_actor with wrapper
            env.grasp_actor = grasp_actor_wrapper
            
            try:
                # Try to use expert_recovery_sequence if available (backward compatibility)
                # If it raises NotImplementedError, fall back to play_once with skip_stages
                if hasattr(env, "expert_recovery_sequence"):
                    try:
                        success = env.expert_recovery_sequence(variant)
                    except NotImplementedError:
                        # If expert_recovery_sequence is not implemented, fall back to play_once with skip_stages
                        print(f"\033[93m[Recovery] expert_recovery_sequence not implemented, using play_once with skip_stages={skip_stages}\033[0m")
                        # Apply skip_stages logic (same as below)
                        import inspect
                        if hasattr(env, 'play_once'):
                            sig = inspect.signature(env.play_once)
                            if 'skip_stages' in sig.parameters:
                                env.play_once(skip_stages=skip_stages)
                            else:
                                # Apply generic stage skipping
                                if skip_stages > 0:
                                    original_move = env.move
                                    move_call_count = [0]
                                    
                                    def move_wrapper(*args, **kwargs):
                                        move_call_count[0] += 1
                                        if move_call_count[0] <= skip_stages:
                                            print(f"\033[93m[Recovery] [Variant {variant}] Skipping stage {move_call_count[0]}/{skip_stages} (move call {move_call_count[0]})\033[0m")
                                            return True
                                        else:
                                            executed_stage = move_call_count[0] - skip_stages
                                            print(f"\033[92m[Recovery] [Variant {variant}] Executing stage {executed_stage} (move call {move_call_count[0]})\033[0m")
                                            return original_move(*args, **kwargs)
                                    
                                    env.move = move_wrapper
                                    try:
                                        print(f"\033[94m[Recovery] Executing play_once with {skip_stages} stages skipped\033[0m")
                                        env.play_once()
                                    finally:
                                        env.move = original_move
                                        print(f"\033[94m[Recovery] Completed play_once, executed {move_call_count[0] - skip_stages} move calls\033[0m")
                                else:
                                    print(f"\033[94m[Recovery] Executing play_once with all stages (variant 1)\033[0m")
                                    env.play_once()
                        success = env.check_success() if hasattr(env, 'check_success') else False
                else:
                    # Use play_once directly
                    import inspect
                    if hasattr(env, 'play_once'):
                        sig = inspect.signature(env.play_once)
                        if 'skip_stages' in sig.parameters:
                            # Task supports skip_stages parameter
                            env.play_once(skip_stages=skip_stages)
                        else:
                            # Task doesn't support skip_stages, use generic stage skipping
                            # We intercept self.move() calls to skip the first skip_stages calls
                            if skip_stages > 0:
                                # Create a wrapper to intercept self.move() calls
                                original_move = env.move
                                move_call_count = [0]  # Use list to allow modification in nested function
                                
                                def move_wrapper(*args, **kwargs):
                                    move_call_count[0] += 1
                                    if move_call_count[0] <= skip_stages:
                                        # Skip this move call - return True to avoid breaking play_once logic
                                        print(f"\033[93m[Recovery] [Variant {variant}] Skipping stage {move_call_count[0]}/{skip_stages} (move call {move_call_count[0]})\033[0m")
                                        return True  # Return True instead of None to match move() return type
                                    else:
                                        # Execute the move call
                                        executed_stage = move_call_count[0] - skip_stages
                                        print(f"\033[92m[Recovery] [Variant {variant}] Executing stage {executed_stage} (move call {move_call_count[0]})\033[0m")
                                        return original_move(*args, **kwargs)
                                
                                # Replace move method temporarily
                                env.move = move_wrapper
                                try:
                                    print(f"\033[94m[Recovery] Executing play_once with {skip_stages} stages skipped\033[0m")
                                    env.play_once()
                                finally:
                                    # Restore original move method
                                    env.move = original_move
                                    print(f"\033[94m[Recovery] Completed play_once, executed {move_call_count[0] - skip_stages} move calls\033[0m")
                            else:
                                # variant 1: execute all stages
                                print(f"\033[94m[Recovery] Executing play_once with all stages (variant 1)\033[0m")
                                env.play_once()
                    
                    # Check success
                    success = env.check_success() if hasattr(env, 'check_success') else False
                
                # Always save video (whether success or failure)
                try:
                    env.merge_pkl_to_hdf5_video(custom_video_name=custom_name)
                    env.remove_data_cache()
                    variant_video_path = os.path.join(env.save_dir, "video", f"{custom_name}.mp4")
                    if success:
                        print(f"\033[92m[Recovery] Variant {variant} succeeded, video saved to {variant_video_path}\033[0m")
                    else:
                        print(f"\033[93m[Recovery] Variant {variant} failed, video saved to {variant_video_path}\033[0m")
                except Exception as e:
                    print(f"\033[91m[Recovery] Warning: failed to save recovery video for variant {variant}: {e}\033[0m")
                
            except Exception as e:
                print(f"\033[91m[Recovery] Variant {variant} failed with error: {e}\033[0m")
                import traceback
                traceback.print_exc()
                success = False
                
                # Still try to save video even if execution failed
                try:
                    env.merge_pkl_to_hdf5_video(custom_video_name=custom_name)
                    env.remove_data_cache()
                    variant_video_path = os.path.join(env.save_dir, "video", f"{custom_name}.mp4")
                    print(f"\033[93m[Recovery] Variant {variant} error video saved to {variant_video_path}\033[0m")
                except:
                    pass
            finally:
                # Always restore original methods, even if exception occurred
                if 'original_grasp_actor' in locals():
                    env.grasp_actor = original_grasp_actor
                if 'original_back_to_origin' in locals():
                    env.back_to_origin = original_back_to_origin
            
            # Restore original save flags
            env.save_data = original_save_data
            env.save_dir = original_save_dir
            env.save_freq = original_save_freq
            
            # Store result
            recovery_results.append({
                'variant': variant,
                'success': success,
                'variant_video_path': variant_video_path,
                'env': None,  # Don't store env reference, we'll close it
            })
            
            # IMPORTANT: Close environment immediately to free GPU resources
            # This prevents "cannot create buffer" errors when creating multiple environments
            try:
                env.close_env()
            except:
                pass
            
            if success:
                if variant_success_counts is not None:
                    variant_success_counts[variant] = variant_success_counts.get(variant, 0) + 1
                print(f"\033[92m[Recovery] Success with variant {variant}\033[0m")
                
                # Store concatenation info
                if original_env.eval_video_path is not None and variant_video_path is not None:
                    if not hasattr(original_env, '_recovery_concat_info'):
                        original_env._recovery_concat_info = []
                    original_env._recovery_concat_info.append({
                        'episode_index': episode_index,
                        'variant': variant,
                        'variant_video_path': variant_video_path,
                    })
        
        # Return True if any variant succeeded
        any_success = any(r['success'] for r in recovery_results)
        if not any_success:
            print("\033[91m[Recovery] All recovery variants failed\033[0m")
        return any_success

    def _try_expert_recovery(
        env,
        episode_index: int = 0,
        variant_success_counts: dict[int, int] | None = None,
        variant_total_counts: dict[int, int] | None = None,
    ):
        """
        Legacy expert recovery system (backward compatibility).
        For new multi-task support, use _try_multi_env_recovery instead.
        """
        if not hasattr(env, "expert_recovery_sequence") or not hasattr(env, "get_recovery_steps"):
            return False  # Task doesn't support recovery

        try:
            num_steps = env.get_recovery_steps()
        except (NotImplementedError, AttributeError):
            return False  # Task hasn't implemented get_recovery_steps()

        # Take a snapshot at the current rollout state
        snapshot = _snapshot_env(env)

        print(f"\n\033[93m[Recovery] Trying expert recovery branches at step {env.take_action_cnt}\033[0m")
        
        # Try to detect which step failed (smart variant selection)
        # If task has detect_failed_step method, use it to determine starting variant
        variants_to_try = None
        if hasattr(env, "detect_failed_step"):
            try:
                detected_variant = env.detect_failed_step()
                if detected_variant is not None:
                    variants_to_try = [detected_variant]
                    print(f"\033[93m[Recovery] Detected failed step, will try variant {detected_variant} only\033[0m")
                else:
                    # All steps completed, no recovery needed
                    print(f"\033[92m[Recovery] All steps already completed, no recovery needed\033[0m")
                    return False
            except Exception as e:
                print(f"\033[93m[Recovery] Failed to detect failed step: {e}, will try all variants\033[0m")
        
        # If detection failed or not implemented, try all variants
        if variants_to_try is None:
            variants_to_try = range(1, num_steps + 1)
            print(f"\033[93m[Recovery] Task has {num_steps} steps, generating {num_steps} variants\033[0m")
        
        for variant in variants_to_try:
            variant_label = f"variant {variant}" if variant == 1 else f"variant {variant} (skip first {variant-1} step(s))"
            print(f"\033[93m[Recovery] {variant_label}\033[0m")

            # Count how many times each variant is attempted
            if variant_total_counts is not None:
                variant_total_counts[variant] = variant_total_counts.get(variant, 0) + 1

            _restore_env(env, snapshot)
            
            # Configure per-variant data/video saving
            original_save_data = getattr(env, "save_data", False)
            original_save_dir = getattr(env, "save_dir", None)
            original_save_freq = getattr(env, "save_freq", None)
            if hasattr(env, "folder_path"):
                env.folder_path = None

            # Save recovery branches under a dedicated directory
            env.save_data = True
            env.save_dir = os.path.join(
                "eval_result",
                task_name,
                args["policy_name"],
                args["task_config"],
                args["ckpt_setting"],
                "recovery",
                f"episode{episode_index}_variant{variant}",
            )
            env.FRAME_IDX = 0
            env.save_freq = 8  # Record every 8 steps during recovery to speed up video significantly

            success = env.expert_recovery_sequence(variant)

            # Merge cached pkl files into an hdf5 + mp4 video for this variant
            variant_video_path = None
            try:
                custom_name = f"episode{episode_index}_variant{variant}"
                env.merge_pkl_to_hdf5_video(custom_video_name=custom_name)
                env.remove_data_cache()
                variant_video_path = os.path.join(env.save_dir, "video", f"{custom_name}.mp4")
            except Exception as e:
                print(f"\033[91m[Recovery] Warning: failed to save recovery data for variant {variant}: {e}\033[0m")

            # Restore original save flags/dirs
            env.save_data = original_save_data
            env.save_dir = original_save_dir
            env.save_freq = original_save_freq

            # Store info to create full video later (after main video is fully written)
            # The main video might still be writing when recovery completes
            if variant_video_path is not None and os.path.exists(variant_video_path):
                # Store recovery video info for later concatenation
                if not hasattr(env, '_recovery_video_info'):
                    env._recovery_video_info = []
                env._recovery_video_info.append({
                    'episode_index': episode_index,
                    'variant': variant,
                    'variant_video_path': variant_video_path,
                    'recovery_save_dir': env.save_dir,  # Save before restoring
                    'custom_name': custom_name,
                })

            if success:
                if variant_success_counts is not None:
                    variant_success_counts[variant] = variant_success_counts.get(variant, 0) + 1
                print(f"\033[92m[Recovery] Success with variant {variant}\033[0m")

                # Store concatenation info for later (to create full video in main directory)
                if env.eval_video_path is not None and variant_video_path is not None:
                    if not hasattr(env, '_recovery_concat_info'):
                        env._recovery_concat_info = []
                    env._recovery_concat_info.append({
                        'episode_index': episode_index,
                        'variant': variant,
                        'variant_video_path': variant_video_path,
                    })
                return True

        print("\033[91m[Recovery] All expert variants failed\033[0m")
        return False
    # ===========================================================

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

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

        succ = False
        reset_func(model)
        recovery_attempted = False
        
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            
            # 执行策略动作（rollout）
            eval_func(TASK_ENV, model, observation)

            # ---------------- Universal Expert-based Error Recovery ----------------
            if (not recovery_attempted
                    and _should_trigger_recovery(TASK_ENV, task_name, frame_threshold)
                    and (not TASK_ENV.eval_success)):
                recovery_attempted = True
                
                # Take snapshot before recovery
                snapshot = _snapshot_env(TASK_ENV)
                
                # Use multi-environment recovery (new multi-task support)
                recovered = _try_multi_env_recovery(
                    task_name,
                    TASK_ENV,
                    snapshot,
                    episode_index=TASK_ENV.test_num,
                    num_stages=num_stages,
                    args=args,
                    recovery_seed=now_seed,
                    recovery_force_open_grippers=recovery_force_open_grippers,
                    variant_success_counts=variant_success_counts,
                    variant_total_counts=variant_total_counts,
                )
                
                if recovered:
                    succ = True
                    # IMPORTANT: Stop rollout after recovery (don't continue)
                    break
            # -----------------------------------------------------------------------

            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        # Create full videos (rollout + recovery) in recovery directory
        # This is done AFTER main video is fully written and closed
        if hasattr(TASK_ENV, '_recovery_video_info') and TASK_ENV._recovery_video_info:
            import time
            for recovery_info in TASK_ENV._recovery_video_info:
                episode_idx = recovery_info['episode_index']
                variant = recovery_info['variant']
                variant_video_path = recovery_info['variant_video_path']
                recovery_save_dir = recovery_info['recovery_save_dir']
                custom_name = recovery_info['custom_name']
                
                try:
                    # Get main video path (rollout part)
                    main_video = None
                    if TASK_ENV.eval_video_path is not None:
                        main_video = os.path.join(TASK_ENV.eval_video_path, f"episode{episode_idx}.mp4")
                    
                    if main_video:
                        # Wait for main video to be fully written
                        max_wait = 10
                        waited = 0
                        while waited < max_wait:
                            if os.path.exists(main_video) and os.path.getsize(main_video) > 0:
                                # Verify video is valid (has moov atom) by checking with ffprobe
                                try:
                                    check_result = subprocess.run(
                                        ["ffprobe", "-v", "error", main_video],
                                        capture_output=True,
                                        timeout=2,
                                    )
                                    if check_result.returncode == 0:
                                        break  # Video is valid, proceed
                                except:
                                    pass  # ffprobe failed, try anyway
                            time.sleep(0.2)
                            waited += 0.2
                        
                        if os.path.exists(main_video) and os.path.exists(variant_video_path):
                            # Ensure video directory exists
                            recovery_video_dir = os.path.join(recovery_save_dir, "video")
                            os.makedirs(recovery_video_dir, exist_ok=True)
                            
                            # Create full video path in recovery directory
                            recovery_full_video = os.path.join(
                                recovery_video_dir,
                                f"{custom_name}_full.mp4",
                            )
                            concat_list_path = os.path.join(
                                recovery_video_dir,
                                f"{custom_name}_full_concat.txt",
                            )
                            
                            # Create concat file with absolute paths
                            with open(concat_list_path, "w", encoding="utf-8") as f:
                                f.write(f"file '{os.path.abspath(main_video)}'\n")
                                f.write(f"file '{os.path.abspath(variant_video_path)}'\n")
                            
                            # Concatenate with proper FPS (20 FPS for normal playback speed)
                            result = subprocess.run(
                                [
                                    "ffmpeg",
                                    "-y",
                                    "-loglevel",
                                    "error",
                                    "-f",
                                    "concat",
                                    "-safe",
                                    "0",
                                    "-i",
                                    concat_list_path,
                                    "-r",
                                    "20",  # 20 FPS for consistent playback speed
                                    "-pix_fmt",
                                    "yuv420p",
                                    "-vcodec",
                                    "libx264",
                                    "-crf",
                                    "23",
                                    recovery_full_video,
                                ],
                                capture_output=True,
                                timeout=60,
                            )
                            
                            if result.returncode == 0 and os.path.exists(recovery_full_video):
                                print(f"\033[92m[Recovery] Full video (rollout+recovery) saved to {recovery_full_video}\033[0m")
                                try:
                                    os.remove(concat_list_path)
                                except:
                                    pass
                            else:
                                error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
                                print(f"\033[93m[Recovery] Warning: Failed to create full video in recovery dir: {error_msg}\033[0m")
                        else:
                            print(f"\033[93m[Recovery] Warning: Main or recovery video not found (main={os.path.exists(main_video) if main_video else False}, recovery={os.path.exists(variant_video_path)})\033[0m")
                except Exception as e:
                    print(f"\033[93m[Recovery] Warning: Failed to create full video in recovery directory: {e}\033[0m")
            
            # Clear the recovery video info after processing
            TASK_ENV._recovery_video_info = []

        # Concatenate main video with successful recovery videos (for main directory)
        if succ and hasattr(TASK_ENV, '_recovery_concat_info') and TASK_ENV._recovery_concat_info:
            for concat_info in TASK_ENV._recovery_concat_info:
                episode_idx = concat_info['episode_index']
                variant = concat_info['variant']
                variant_video_path = concat_info['variant_video_path']
                
                try:
                    main_video = os.path.join(TASK_ENV.eval_video_path, f"episode{episode_idx}.mp4")
                    
                    # Wait a bit to ensure file is fully flushed
                    import time
                    max_wait = 5
                    waited = 0
                    while waited < max_wait and (not os.path.exists(main_video) or os.path.getsize(main_video) == 0):
                        time.sleep(0.1)
                        waited += 0.1
                    
                    if os.path.exists(main_video) and os.path.exists(variant_video_path):
                        full_video = os.path.join(
                            TASK_ENV.eval_video_path,
                            f"episode{episode_idx}_variant{variant}_full.mp4",
                        )
                        concat_list_path = os.path.join(
                            TASK_ENV.eval_video_path,
                            f"episode{episode_idx}_variant{variant}_concat.txt",
                        )
                        
                        # Use absolute paths in concat file
                        with open(concat_list_path, "w", encoding="utf-8") as f:
                            f.write(f"file '{os.path.abspath(main_video)}'\n")
                            f.write(f"file '{os.path.abspath(variant_video_path)}'\n")
                        
                        # Run ffmpeg concatenation with re-encoding
                        # Use 20 FPS for concatenated video to speed it up
                        # This matches main video's perceived speed better
                        result = subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-loglevel",
                                "error",
                                "-f",
                                "concat",
                                "-safe",
                                "0",
                                "-i",
                                concat_list_path,
                                "-r",
                                "20",  # Increased from 10 to 20 FPS to speed up concatenated video
                                "-pix_fmt",
                                "yuv420p",
                                "-vcodec",
                                "libx264",
                                "-crf",
                                "23",
                                full_video,
                            ],
                            capture_output=True,
                            timeout=60,
                        )
                        
                        if result.returncode == 0 and os.path.exists(full_video) and os.path.getsize(full_video) > 0:
                            print(f"\033[92m[Recovery] Full episode video saved to {full_video}\033[0m")
                            try:
                                os.remove(concat_list_path)
                            except:
                                pass
                        else:
                            error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
                            print(f"\033[91m[Recovery] Failed to concatenate videos: {error_msg}\033[0m")
                    else:
                        print(
                            f"\033[93m[Recovery] Skip concatenation: main or variant video not found "
                            f"(main={os.path.exists(main_video)}, variant={os.path.exists(variant_video_path)})\033[0m"
                        )
                except Exception as e:
                    print(f"\033[91m[Recovery] Warning: Failed to concatenate videos: {e}\033[0m")
            
            # Clear the concat info after processing
            TASK_ENV._recovery_concat_info = []

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

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    # Print per-variant recovery statistics
    if variant_total_counts:
        print("\n\033[94m[Recovery Summary per variant]\033[0m")
        for v in sorted(variant_total_counts.keys()):
            total = variant_total_counts.get(v, 0)
            succ_v = variant_success_counts.get(v, 0)
            if total == 0:
                rate = 0.0
            else:
                rate = round(succ_v / total * 100.0, 1)
            print(f"  Variant {v}: {succ_v}/{total} => {rate}%")

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
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
