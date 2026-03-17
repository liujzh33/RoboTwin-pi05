"""
click_alarmclock 子任务描述丰富化（CycleVLA 风格，动词开头）
3 阶段：悬停 → 握拳准备 → 按压并返回。
"""

import os
import json
import glob
import random
from tqdm import tqdm

DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/click_alarmclock-aloha-agilex_randomized_500-200"

PHASE_VARIANTS = {
    # Phase 0: Approach
    0: [
        "Move the gripper above the alarm clock.",
        "Position the arm over the alarm clock.",
        "Navigate the gripper to the top of the clock.",
        "Hover the end-effector above the alarm clock.",
        "Bring the gripper directly over the alarm clock.",
        "Align the arm above the clock button.",
        "Reach the space above the alarm clock.",
        "Move towards the top of the alarm clock.",
        "Direct the gripper above the target clock.",
        "Approach the alarm clock from above.",
    ],
    # Phase 1: Close to prepare (握拳)
    1: [
        "Close the gripper to prepare for clicking.",
        "Make a fist with the gripper to press the clock.",
        "Shut the fingers to prepare for pushing.",
        "Close the gripper tight to click the alarm.",
        "Prepare to click by closing the gripper.",
        "Contract the fingers to form a pressing pose.",
        "Close the hand to get ready to push the clock.",
        "Fold the gripper to strike the alarm clock.",
        "Clench the gripper in preparation for the click.",
        "Close the end-effector to press the button.",
    ],
    # Phase 2: Click and return
    2: [
        "Click the alarm clock and return.",
        "Move the gripper down to press the alarm clock.",
        "Push the alarm clock button and retreat.",
        "Press down on the alarm clock.",
        "Strike the top of the alarm clock.",
        "Push the clock button down and move back.",
        "Click the clock with the closed gripper.",
        "Perform a downward press on the alarm clock.",
        "Hit the alarm clock button.",
        "Lower the arm to click the alarm clock.",
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

    print(f"Found {len(episode_dirs)} episodes. Enriching (3 phases: Approach → Close to prepare → Click and return)...")

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
                instructions = ["Click the alarm clock."]
                data["instructions"] = instructions

            phase_info = data.get("phase_info", {})
            num_phases = phase_info.get("num_phases")
            if num_phases is None and data.get("subtasks"):
                num_phases = len(data["subtasks"][0])
            if num_phases is None:
                num_phases = 3

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
    p = argparse.ArgumentParser(description="click_alarmclock 子任务语言丰富化（clean/random 共用一套）")
    p.add_argument("--data_dir", type=str, default=None, help="processed_data 目录")
    args = p.parse_args()
    main(data_root=args.data_dir)
