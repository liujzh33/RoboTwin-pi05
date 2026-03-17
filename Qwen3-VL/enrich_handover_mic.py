"""
handover_mic 子任务描述丰富化脚本

与 task_definitions/handover_mic.py 一致：2 个 checkpoint → 3 阶段。
  checkpoint: [先闭合臂闭合, 另一臂闭合]
  P0: 0~cp1 靠近麦克风
  P1: cp1~cp2 抓住麦克风并移动到桌面中间
  P2: cp2~end 一臂保持不动、另一臂抓住麦克风并原臂放开

num_phases 以 instructions.json 的 phase_info.num_phases 为准（process_data_generic 写入）。
"""

import os
import json
import glob
import random
from tqdm import tqdm

DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/handover_mic-aloha-agilex_randomized_500-200"

FALLBACK_VARIANTS = ["Complete the task.", "Continue with the next step."]

PHASE_VARIANTS = {
    # Phase 0: 靠近麦克风
    0: [
        "Approach the microphone.",
        "Move the arm close to the microphone.",
        "Bring the gripper near the microphone.",
        "Position the end-effector above the microphone.",
        "Reach towards the microphone.",
        "Drive the arm to the microphone.",
        "Align the gripper with the microphone.",
        "Get the arm above the microphone.",
        "Move to the microphone.",
        "Reach for the microphone.",
    ],
    # Phase 1: 抓住麦克风并移动到桌面中间
    1: [
        "Grasp the microphone and move it to the middle of the table.",
        "Pick up the microphone and move it to the center of the table.",
        "Grab the microphone and bring it to the middle of the workspace.",
        "Close the gripper on the microphone and move it to the middle.",
        "Take hold of the microphone and move it to the center.",
        "Grasp the microphone and move it to the central area of the table.",
        "Secure the microphone and move it to the middle.",
        "Pick up the microphone and reposition it to the middle of the table.",
        "Grab the microphone and move it to the center of the table.",
        "Hold the microphone and move it to the middle of the workspace.",
    ],
    # Phase 2: 一臂保持不动，另一臂抓住麦克风并原臂放开
    2: [
        "Keep one arm still; the other arm grasps the microphone and the first arm releases.",
        "One arm holds steady while the other grasps the microphone; then the first arm releases.",
        "Hold with one arm; the other gripper grasps the microphone and the first releases.",
        "Transfer the microphone: other arm grasps it and the first arm releases.",
        "Other arm grasps the microphone while the first arm holds still, then the first releases.",
        "One arm keeps still; the other grasps the microphone and the first arm lets go.",
        "Other gripper grasps the microphone; first arm holds then releases.",
        "Take the microphone with the other arm and release with the first arm.",
        "Other arm grasps the microphone; first arm opens to release.",
        "Hand over: other arm grasps the microphone and the first arm releases it.",
    ],
}


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root not found: {DATA_ROOT}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "episode_*")))
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    print(f"Found {len(episode_dirs)} episodes in {DATA_ROOT}")
    print("Enriching handover_mic subtasks (3-phase: Approach → Grasp+Move → Handover)...")

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
                instructions = ["Grasp the microphone with one arm, then hand it over to the other arm."]
                data["instructions"] = instructions
            phase_info = data.get("phase_info", {})
            num_phases = phase_info.get("num_phases")
            if num_phases is None and data.get("subtasks"):
                num_phases = len(data["subtasks"][0])
            if num_phases is None:
                num_phases = 3
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
    print("Subtask descriptions are now diverse (3-phase: Approach → Grasp+Move → Handover).")


if __name__ == "__main__":
    main()
