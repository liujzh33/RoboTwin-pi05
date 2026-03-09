"""
hanging_mug 子任务描述丰富化脚本

与 task_definitions/hanging_mug.py 一致：4 个 checkpoint → 5 阶段。
  P0: 靠近马克杯  P1: 抓住马克杯  P2: 放置到桌面中间
  P3: 另一臂抓住马克杯边缘  P4: 悬挂马克杯
"""

import os
import json
import glob
import random
from tqdm import tqdm

DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/hanging_mug-aloha-agilex_randomized_500-200"

FALLBACK_VARIANTS = ["Complete the task.", "Continue with the next step."]

PHASE_VARIANTS = {
    0: [
        "Approach the mug.",
        "Move the arm close to the mug.",
        "Bring the gripper near the mug.",
        "Position the end-effector above the mug.",
        "Reach towards the mug.",
        "Drive the arm to the mug.",
        "Align the gripper with the mug.",
        "Get the arm above the mug.",
        "Move to the mug.",
        "Reach for the mug.",
    ],
    1: [
        "Grasp the mug.",
        "Close the gripper to hold the mug.",
        "Clamp the mug with the gripper.",
        "Grab the mug.",
        "Take hold of the mug.",
        "Secure the mug with the gripper.",
        "Close the gripper around the mug.",
        "Pick up the mug.",
        "Hold the mug firmly with the gripper.",
        "Grasp the mug with the gripper.",
    ],
    2: [
        "Place the mug in the middle of the table.",
        "Set the mug down in the center of the table.",
        "Put the mug in the middle of the workspace.",
        "Move the mug to the middle of the table and release.",
        "Position the mug in the center and release the gripper.",
        "Drop the mug at the middle of the table.",
        "Place the mug at the central position on the table.",
        "Set down the mug in the middle of the table.",
        "Put the mug down in the center of the workspace.",
        "Release the mug in the middle of the table.",
    ],
    3: [
        "The other arm grasps the rim of the mug.",
        "Grasp the rim of the mug with the other arm.",
        "The other gripper closes on the mug rim.",
        "Take hold of the mug rim with the other arm.",
        "The other arm grabs the edge of the mug.",
        "Grasp the mug edge with the other gripper.",
        "The other arm secures the mug by the rim.",
        "Close the other gripper on the mug rim.",
        "The other arm grasps the mug by its edge.",
        "Grab the mug rim with the other hand.",
    ],
    4: [
        "Hang the mug on the rack.",
        "Place the mug on the rack.",
        "Hang the mug on the hook.",
        "Mount the mug on the rack.",
        "Put the mug on the hanging rack.",
        "Hang the mug on the stand.",
        "Place the mug onto the rack.",
        "Set the mug on the rack.",
        "Hang the mug on the rack and release.",
        "Mount the mug on the hanging rack.",
    ],
}


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root not found: {DATA_ROOT}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "episode_*")))
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    print(f"Found {len(episode_dirs)} episodes in {DATA_ROOT}")
    print("Enriching hanging_mug subtasks (5-phase: Approach → Grasp → Place → Other grasp rim → Hang)...")

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
                instructions = ["Grasp the mug, place it in the middle, then hang it on the rack with the other arm."]
                data["instructions"] = instructions
            phase_info = data.get("phase_info", {})
            num_phases = phase_info.get("num_phases")
            if num_phases is None and data.get("subtasks"):
                num_phases = len(data["subtasks"][0])
            if num_phases is None:
                num_phases = 5
            num_phases = int(num_phases)
            new_subtasks_list = []
            for _ in range(len(instructions)):
                variant = [random.choice(PHASE_VARIANTS.get(i, FALLBACK_VARIANTS)) for i in range(num_phases)]
                new_subtasks_list.append(variant)
            data["subtasks"] = new_subtasks_list
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            success_count += 1
        except Exception as e:
            print(f"Error processing {ep_dir}: {e}")

    print(f"\nDone! Successfully updated {success_count}/{len(episode_dirs)} episodes.")
    print("Subtask descriptions are now diverse (5-phase: Approach → Grasp → Place → Other grasp rim → Hang).")


if __name__ == "__main__":
    main()
