"""
beat_block_hammer 子任务描述丰富化（CycleVLA 风格，动词开头）
4 阶段：找锤子 → 抓锤子 → 移动到红块上方 → 敲击。
"""

import os
import json
import glob
import random
from tqdm import tqdm

DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/beat_block_hammer-aloha-agilex_randomized_500-200"

PHASE_VARIANTS = {
    # Phase 0: Approach Hammer
    0: [
        "Move the gripper above the hammer.",
        "Position the gripper over the hammer handle.",
        "Navigate the arm towards the hammer.",
        "Bring the end-effector above the tool.",
        "Hover the gripper over the hammer.",
        "Direct the arm to the hammer on the table.",
        "Align the gripper with the hammer.",
        "Reach the space directly above the hammer.",
        "Move towards the hammer to pick it up.",
        "Approach the hammer from above.",
    ],
    # Phase 1: Grasp Hammer
    1: [
        "Close the gripper to grasp the hammer.",
        "Grasp the hammer firmly.",
        "Secure the hammer with the gripper.",
        "Pick up the hammer from the table.",
        "Clamp down on the hammer handle.",
        "Take hold of the hammer.",
        "Close the fingers to hold the tool.",
        "Squeeze the gripper around the hammer.",
        "Grip the hammer tightly.",
        "Engage the gripper to secure the hammer.",
    ],
    # Phase 2: Move Hammer above Block
    2: [
        "Move the hammer above the red block.",
        "Transport the hammer over the red cube.",
        "Navigate the tool to the top of the red block.",
        "Position the grasped hammer above the target block.",
        "Carry the hammer over to the red cube.",
        "Bring the hammer directly above the red block.",
        "Align the hammer with the red target.",
        "Hover the hammer over the red object.",
        "Move the arm holding the hammer towards the red block.",
        "Aim the hammer at the top of the red cube.",
    ],
    # Phase 3: Hit Block
    3: [
        "Move the hammer down to hit the red block.",
        "Strike the red block with the hammer.",
        "Hit the red cube using the hammer.",
        "Swing the hammer down onto the red block.",
        "Beat the red block with the tool.",
        "Drive the hammer down to strike the target.",
        "Smash the top of the red block.",
        "Lower the hammer aggressively to hit the red cube.",
        "Pound the red block with the grasped hammer.",
        "Execute a downward strike on the red block.",
    ],
}

FALLBACK_VARIANTS = ["Complete the task.", "Continue with the next step."]


def main(data_root=None):
    root = (data_root or DATA_ROOT).rstrip("/")
    if not os.path.exists(root):
        print(f"Error: Data root not found: {root}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(root, "episode_*")))
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    print(f"Found {len(episode_dirs)} episodes. Enriching (4 phases: Approach hammer → Grasp → Move above block → Hit)...")

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
                instructions = ["Use the hammer to hit the red block."]
                data["instructions"] = instructions

            phase_info = data.get("phase_info", {})
            num_phases = phase_info.get("num_phases")
            if num_phases is None and data.get("subtasks"):
                num_phases = len(data["subtasks"][0])
            if num_phases is None:
                num_phases = 4

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

    print(f"\nDone! Updated {success_count}/{len(episode_dirs)} episodes.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="beat_block_hammer 子任务语言丰富化（clean/random 共用一套 PHASE_VARIANTS）")
    p.add_argument("--data_dir", type=str, default=None, help="processed_data 目录，如 processed_data/beat_block_hammer-aloha-agilex_clean_50-50 或 ...-randomized_500-500")
    args = p.parse_args()
    main(data_root=args.data_dir)
