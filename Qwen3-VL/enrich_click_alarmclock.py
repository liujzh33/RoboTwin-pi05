"""
click_alarmclock 子任务描述丰富化
仅 2 阶段：① 靠近物体（到达上方），夹爪闭合后 → ② click 物体。
"""

import os
import json
import glob
import random
from tqdm import tqdm

DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/click_alarmclock-aloha-agilex_randomized_500-200"

PHASE_VARIANTS = {
    # Phase 0: 靠近物体（到达目标上方），直至夹爪闭合
    0: [
        "Approach the alarm clock (reach above it)",
        "Move to the alarm clock and position above it",
        "Reach above the alarm clock",
        "Position the end-effector above the alarm clock",
        "Get close to the alarm clock from above",
        "Align the gripper above the alarm clock",
        "Move the arm above the alarm clock",
        "Reach for the alarm clock and get above it",
        "Go to the alarm clock and hover above it",
        "Approach from above until the gripper closes",
    ],
    # Phase 1: 点击物体
    1: [
        "Click the alarm clock",
        "Press the alarm clock",
        "Tap the alarm clock",
        "Trigger the alarm clock",
        "Activate the alarm clock by clicking",
        "Perform the click on the alarm clock",
        "Execute the click on the alarm clock",
        "Press down on the alarm clock",
        "Click to activate the alarm clock",
    ],
}

FALLBACK_VARIANTS = ["Complete the task.", "Continue with the next step."]


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root not found: {DATA_ROOT}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "episode_*")))
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    print(f"Found {len(episode_dirs)} episodes. Enriching (2 phases: Approach → Click)...")

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
                num_phases = 2

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
    main()
