"""
handover_block 子任务描述丰富化脚本

与 task_definitions/handover_block.py 一致：3 个 checkpoint → 4 阶段。
  checkpoint: [左闭, 右闭, 左开]
  P0: 0~cp1 左臂靠近红色木块
  P1: cp1~cp2 抓住红色木块并移动到中间
  P2: cp2~cp3 右臂抓住红块、左臂放开
  P3: cp3~end 右臂放置红色木块到蓝色块上

num_phases 以 instructions.json 的 phase_info.num_phases 为准（process_data_generic 写入）。
"""

import os
import json
import glob
import random
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/handover_block-aloha-agilex_randomized_500-200"

FALLBACK_VARIANTS = [
    "Complete the task.",
    "Continue with the next step.",
]

# === 4 阶段多样化词库 ===
PHASE_VARIANTS = {
    # Phase 0: 左臂靠近红色木块
    0: [
        "Approach the red block with the left arm.",
        "Move the left arm close to the red block.",
        "Bring the left gripper near the red block.",
        "Position the left end-effector above the red block.",
        "Reach towards the red block with the left arm.",
        "Drive the left arm to the red block.",
        "Align the left gripper with the red block.",
        "Get the left arm above the red block.",
        "Move the left arm to the red block.",
        "Reach for the red block with the left arm.",
    ],
    # Phase 1: 抓住红色木块并移动到中间
    1: [
        "Grasp the red block and move it to the middle.",
        "Pick up the red block with the left gripper and move it to the center.",
        "Grab the red block and bring it to the middle of the workspace.",
        "Close the left gripper on the red block and move it to the middle.",
        "Take hold of the red block and move it to the center.",
        "Grasp the red block with the left arm and move it to the middle.",
        "Secure the red block and move it to the central area.",
        "Pick up the red block and reposition it to the middle.",
        "Grab the red block and move it to the center of the table.",
        "Hold the red block and move it to the middle.",
    ],
    # Phase 2: 右臂抓住红块，左臂放开
    2: [
        "Grasp the red block with the right gripper and release with the left.",
        "Right gripper grasps the red block while the left gripper releases it.",
        "Take the red block with the right arm and release with the left.",
        "Close the right gripper on the red block and open the left gripper.",
        "Hand over: right grasps the red block, left releases it.",
        "Transfer the block to the right gripper and release the left.",
        "Right arm grasps the red block; left arm releases it.",
        "Grasp the red block with the right hand and let go with the left.",
        "Right gripper takes the red block; left gripper opens to release.",
        "Take the red block with the right arm; release with the left gripper.",
    ],
    # Phase 3: 右臂放置红色木块到蓝色块上
    3: [
        "Place the red block on the blue block with the right arm.",
        "Set the red block down on the blue block.",
        "Put the red block on top of the blue block.",
        "Position the red block on the blue block with the right arm.",
        "Drop the red block onto the blue block.",
        "Place the red block on the blue block and release.",
        "Set down the red block on the blue block with the right gripper.",
        "Move the red block to the blue block and release it.",
        "Place the red block on the blue block using the right arm.",
        "Put the red block on the blue block and open the right gripper.",
    ],
}


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root not found: {DATA_ROOT}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "episode_*")))
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    print(f"Found {len(episode_dirs)} episodes in {DATA_ROOT}")
    print("Enriching handover_block subtasks (4-phase: L-Approach → Grasp+Move → R-Grasp+L-Release → R-Place)...")

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
                instructions = ["Pick up the red block with the left arm, hand it to the right arm, and place it on the blue block."]
                data["instructions"] = instructions

            phase_info = data.get("phase_info", {})
            num_phases = phase_info.get("num_phases")
            if num_phases is None and data.get("subtasks"):
                num_phases = len(data["subtasks"][0])
            if num_phases is None:
                num_phases = 4
            num_phases = int(num_phases)

            new_subtasks_list = []
            for _ in range(len(instructions)):
                variant = []
                for phase_idx in range(num_phases):
                    candidates = PHASE_VARIANTS.get(phase_idx, FALLBACK_VARIANTS)
                    variant.append(random.choice(candidates))
                new_subtasks_list.append(variant)

            data["subtasks"] = new_subtasks_list

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            success_count += 1

        except Exception as e:
            print(f"Error processing {ep_dir}: {e}")

    print(f"\nDone! Successfully updated {success_count}/{len(episode_dirs)} episodes.")
    print("Subtask descriptions are now diverse (4-phase: L-Approach → Grasp+Move → R-Grasp+L-Release → R-Place).")


if __name__ == "__main__":
    main()
