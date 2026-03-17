"""
blocks_ranking_rgb_v1 子任务描述丰富化（CycleVLA 风格，动词开头）
13 阶段：红→左, 绿→中, 蓝→右（每色：接近→抓取→搬运→释放）+ 复位。
"""

import os
import json
import glob
import random
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/blocks_ranking_rgb-aloha-agilex_randomized_500-200"

# === 13 阶段多样化词库 (红->左, 绿->中, 蓝->右) ===
PHASE_VARIANTS = {
    0: [
        "Move the gripper above the red block.",
        "Position the gripper over the red cube.",
        "Navigate the arm above the red block.",
        "Bring the end-effector over the red object.",
        "Hover the gripper above the red block.",
        "Direct the gripper to the top of the red cube.",
        "Align the gripper above the red block.",
        "Reach the space directly above the red cube.",
        "Move towards the top of the red block.",
        "Approach the red block from above.",
    ],
    1: [
        "Close the gripper to grasp the red block.",
        "Grasp the red block by closing the fingers.",
        "Secure the red cube with the gripper.",
        "Pick up the red block.",
        "Clamp down on the red object.",
        "Take hold of the red block.",
        "Close the fingers around the red cube.",
        "Squeeze the gripper to hold the red block.",
        "Grip the red block tightly.",
        "Engage the gripper to pick the red cube.",
    ],
    2: [
        "Move the gripper to the left target position while holding the red block.",
        "Transport the red block to the left side.",
        "Carry the red cube to the left target.",
        "Shift the grasped red block to the left.",
        "Navigate to the left position with the red block.",
        "Bring the red object to the target on the left.",
        "Move the red cube towards the left placement area.",
        "Transfer the red block to the left.",
        "Relocate the red block to the left target.",
        "Guide the red cube to the left side of the workspace.",
    ],
    3: [
        "Open the gripper to release the red block.",
        "Release the red block by opening the fingers.",
        "Drop the red cube at the current position.",
        "Let go of the red block.",
        "Open the fingers to place the red object.",
        "Unclasp the gripper to release the red block.",
        "Place the red block down by opening the gripper.",
        "Release grip on the red cube.",
        "Set the red block down and open the gripper.",
        "Free the red block from the gripper.",
    ],
    4: [
        "Move the gripper above the green block.",
        "Position the gripper over the green cube.",
        "Navigate the arm above the green block.",
        "Bring the end-effector over the green object.",
        "Hover the gripper above the green block.",
        "Direct the gripper to the top of the green cube.",
        "Align the gripper above the green block.",
        "Reach the space directly above the green cube.",
        "Move towards the top of the green block.",
        "Approach the green block from above.",
    ],
    5: [
        "Close the gripper to grasp the green block.",
        "Grasp the green block by closing the fingers.",
        "Secure the green cube with the gripper.",
        "Pick up the green block.",
        "Clamp down on the green object.",
        "Take hold of the green block.",
        "Close the fingers around the green cube.",
        "Squeeze the gripper to hold the green block.",
        "Grip the green block tightly.",
        "Engage the gripper to pick the green cube.",
    ],
    6: [
        "Move the gripper to the middle target position while holding the green block.",
        "Transport the green block to the center.",
        "Carry the green cube to the middle target.",
        "Shift the grasped green block to the center.",
        "Navigate to the middle position with the green block.",
        "Bring the green object to the target in the middle.",
        "Move the green cube towards the central placement area.",
        "Transfer the green block to the middle.",
        "Relocate the green block to the center target.",
        "Guide the green cube to the middle of the workspace.",
    ],
    7: [
        "Open the gripper to release the green block.",
        "Release the green block by opening the fingers.",
        "Drop the green cube at the current position.",
        "Let go of the green block.",
        "Open the fingers to place the green object.",
        "Unclasp the gripper to release the green block.",
        "Place the green block down by opening the gripper.",
        "Release grip on the green cube.",
        "Set the green block down and open the gripper.",
        "Free the green block from the gripper.",
    ],
    8: [
        "Move the gripper above the blue block.",
        "Position the gripper over the blue cube.",
        "Navigate the arm above the blue block.",
        "Bring the end-effector over the blue object.",
        "Hover the gripper above the blue block.",
        "Direct the gripper to the top of the blue cube.",
        "Align the gripper above the blue block.",
        "Reach the space directly above the blue cube.",
        "Move towards the top of the blue block.",
        "Approach the blue block from above.",
    ],
    9: [
        "Close the gripper to grasp the blue block.",
        "Grasp the blue block by closing the fingers.",
        "Secure the blue cube with the gripper.",
        "Pick up the blue block.",
        "Clamp down on the blue object.",
        "Take hold of the blue block.",
        "Close the fingers around the blue cube.",
        "Squeeze the gripper to hold the blue block.",
        "Grip the blue block tightly.",
        "Engage the gripper to pick the blue cube.",
    ],
    10: [
        "Move the gripper to the right target position while holding the blue block.",
        "Transport the blue block to the right side.",
        "Carry the blue cube to the right target.",
        "Shift the grasped blue block to the right.",
        "Navigate to the right position with the blue block.",
        "Bring the blue object to the target on the right.",
        "Move the blue cube towards the right placement area.",
        "Transfer the blue block to the right.",
        "Relocate the blue block to the right target.",
        "Guide the blue cube to the right side of the workspace.",
    ],
    11: [
        "Open the gripper to release the blue block.",
        "Release the blue block by opening the fingers.",
        "Drop the blue cube at the current position.",
        "Let go of the blue block.",
        "Open the fingers to place the blue object.",
        "Unclasp the gripper to release the blue block.",
        "Place the blue block down by opening the gripper.",
        "Release grip on the blue cube.",
        "Set the blue block down and open the gripper.",
        "Free the blue block from the gripper.",
    ],
    12: [
        "Return to a neutral position.",
        "Move the gripper back to the starting pose.",
        "Reset the arm to a neutral state.",
        "Withdraw the arm to a safe position.",
        "Navigate the end-effector back to the rest pose.",
        "Retract the gripper to the initial position.",
        "Move back to the standby location.",
        "Finish by returning to neutral.",
        "Restore the robot arm to its home position.",
        "Clear the workspace and return to neutral.",
    ],
}

FALLBACK_VARIANTS = ["Complete the task.", "Continue with the next step."]

def main(data_root=None):
    root = (data_root or DATA_ROOT).rstrip("/")
    if not os.path.exists(root):
        print(f"Error: Data root not found: {root}")
        return

    # 查找所有 episode 文件夹
    episode_dirs = sorted(glob.glob(os.path.join(root, "episode_*")))
    
    # 按照数字顺序排序 (防止 episode_10 排在 episode_2 前面)
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

    print(f"Found {len(episode_dirs)} episodes in {root}")
    print("Enriching subtasks (13 phases: Red->Left, Green->Middle, Blue->Right)...")

    success_count = 0

    for ep_dir in tqdm(episode_dirs):
        json_path = os.path.join(ep_dir, "instructions.json")

        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            instructions = data.get("instructions", [])
            if not instructions:
                instructions = ["Sort the blocks by color: red on the left, green in the middle, blue on the right."]
                data["instructions"] = instructions

            phase_info = data.get("phase_info", {})
            num_phases = phase_info.get("num_phases")
            if num_phases is None and data.get("subtasks"):
                num_phases = len(data["subtasks"][0])
            if num_phases is None:
                num_phases = 13

            new_subtasks_list = []
            for _ in range(len(instructions)):
                variant = []
                for phase_idx in range(num_phases):
                    if phase_idx in PHASE_VARIANTS:
                        variant.append(random.choice(PHASE_VARIANTS[phase_idx]))
                    else:
                        variant.append(random.choice(FALLBACK_VARIANTS))
                new_subtasks_list.append(variant)

            data["subtasks"] = new_subtasks_list
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            success_count += 1

        except Exception as e:
            print(f"Error processing {ep_dir}: {e}")

    print(f"\nDone! Successfully updated {success_count}/{len(episode_dirs)} episodes.")
    print("Subtask descriptions are now diverse (CycleVLA style) for Red->Left, Green->Middle, Blue->Right.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="blocks_ranking_rgb_v1 子任务语言丰富化（clean/random 共用一套）")
    p.add_argument("--data_dir", type=str, default=None, help="processed_data 目录")
    args = p.parse_args()
    main(data_root=args.data_dir)