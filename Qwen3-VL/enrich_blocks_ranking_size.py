"""
blocks_ranking_size 子任务描述丰富化
顺序：先抓最小的放到最右边 → 中等放中间 → 最大的放最左边 → 复位
与 enrich_blocks_ranking.py 同理，用多样化词库替换单一表述。
"""

import os
import json
import glob
import random
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/blocks_ranking_size-aloha-agilex_randomized_500-200"

# === 7 阶段多样化词库 (小→最右, 中→中间, 大→最左, 复位) ===
PHASE_VARIANTS = {
    # Phase 0: 抓取最小 (Pick Small)
    0: [
        "Pick up the small block",
        "Grasp the smallest block",
        "Grab the small cube",
        "Retrieve the small block from the table",
        "Secure the small cube with the gripper",
        "Take hold of the small block",
        "Reach for and grasp the small block",
        "Start by picking up the small block",
        "Locate and grab the smallest cube",
        "Lift the small block first",
    ],
    # Phase 1: 放置最小到最右边 (Place Small - Far Right)
    1: [
        "Place the small block on the far right",
        "Set the small block down at the rightmost position",
        "Position the small cube on the far right side",
        "Move the small block to the far right and release it",
        "Put the small block on the extreme right",
        "Establish the right end by placing the small block there",
        "Align the small block on the right side of the workspace",
        "Drop the small block to the far right",
        "Place the smallest block at the right end of the row",
    ],
    # Phase 2: 抓取中等 (Pick Medium)
    2: [
        "Pick up the medium block",
        "Grasp the medium-sized cube",
        "Grab the medium block",
        "Now, take the medium block",
        "Retrieve the medium cube",
        "Secure the medium block",
        "Move to grasp the medium block",
        "Select the medium block next",
        "Pick up the middle-sized one",
        "Lift the medium block",
    ],
    # Phase 3: 放置中等到中间 (Place Medium - Middle)
    3: [
        "Place the medium block in the middle",
        "Put the medium block in the center",
        "Position the medium cube between the small and large blocks",
        "Set the medium block down in the middle",
        "Align the medium block in the center of the row",
        "Drop the medium object in the middle",
        "Place the medium block between the two others",
        "Position the middle-sized block centrally",
        "Set the medium cube in the middle of the workspace",
    ],
    # Phase 4: 抓取最大 (Pick Large)
    4: [
        "Pick up the large block",
        "Grasp the largest cube",
        "Grab the large object",
        "Finally, take the large block",
        "Retrieve the large cube",
        "Secure the large block with the gripper",
        "Pick up the last block, the large one",
        "Take hold of the large cube",
        "Lift the largest block",
        "Grasp the big block",
    ],
    # Phase 5: 放置最大到最左边 (Place Large - Far Left)
    5: [
        "Place the large block on the far left",
        "Put the large block on the leftmost position",
        "Position the large cube on the far left side",
        "Set the large block down at the left end",
        "Move the large block to the far left and release it",
        "Establish the left end by placing the large block there",
        "Align the large block on the left side of the workspace",
        "Complete the row by placing the large block on the far left",
        "Place the largest block at the left end of the row",
    ],
    # Phase 6: 复位 (Return/Finish)
    6: [
        "Return to a neutral position",
        "Move the arm back to the start position",
        "Reset the arm to a safe pose",
        "Retract the gripper to neutral",
        "Task complete, return to rest position",
        "Move the end-effector away",
        "Lift and retreat to home position",
        "Return arm to standby",
        "Finish task and reset",
        "Release and return to neutral",
    ],
}

# 若某 episode 阶段数 > 7 时的兜底
FALLBACK_VARIANTS = [
    "Complete the task.",
    "Continue with the next step.",
    "Proceed to the next phase.",
]


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root not found: {DATA_ROOT}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "episode_*")))
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    print(f"Found {len(episode_dirs)} episodes in {DATA_ROOT}")
    print("Enriching subtasks (Order: small→far right, medium→middle, large→far left)...")

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
                instructions = ["Sort the blocks by size: small on the right, medium in the middle, large on the left."]
                data["instructions"] = instructions

            # 用 phase_info 或已有 subtasks 确定阶段数
            phase_info = data.get("phase_info", {})
            num_phases = phase_info.get("num_phases")
            if num_phases is None and data.get("subtasks"):
                num_phases = len(data["subtasks"][0])
            if num_phases is None:
                num_phases = 7

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
    print("Subtask descriptions are now diverse while keeping order: small→far right, medium→middle, large→far left.")


if __name__ == "__main__":
    main()
