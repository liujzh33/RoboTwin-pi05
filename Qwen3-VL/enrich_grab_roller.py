"""
grab_roller 子任务描述丰富化脚本

2 阶段：双臂靠近容器 → 双臂抓住容器。
与 enrich_blocks_ranking_size.py 同理，用多样化词库替换单一表述。
"""

import os
import json
import glob
import random
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/grab_roller-aloha-agilex_randomized_500-200"

FALLBACK_VARIANTS = [
    "Complete the task.",
    "Continue with the next step.",
]

# === 2 阶段多样化词库（双臂靠近 / 双臂抓住）===
PHASE_VARIANTS = {
    # Phase 0: 双臂靠近容器
    0: [
        "Approach the container with both arms.",
        "Move both arms close to the container.",
        "Bring both grippers near the container.",
        "Position both end-effectors near the container.",
        "Reach towards the container with both arms.",
        "Drive both arms to the container area.",
        "Align both grippers with the container.",
        "Get both arms above the container.",
        "Move both arms smoothly towards the container.",
        "Position the left and right grippers near the container.",
    ],
    # Phase 1: 双臂抓住容器
    1: [
        "Grasp the container with both grippers.",
        "Close both grippers to hold the container firmly.",
        "Clamp the container securely with both arms.",
        "Grab the container using both hands.",
        "Stably hold the container with both grippers.",
        "Take hold of the container with both arms.",
        "Secure the container with both grippers.",
        "Close both grippers around the container.",
        "Pinch the container with both grippers.",
        "Grab and hold the container firmly with both arms.",
    ],
}


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root not found: {DATA_ROOT}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "episode_*")))
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    print(f"Found {len(episode_dirs)} episodes in {DATA_ROOT}")
    print("Enriching grab_roller subtasks (2-phase: Approach → Grasp)...")

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
                instructions = ["Grab the roller with both arms."]
                data["instructions"] = instructions

            phase_info = data.get("phase_info", {})
            num_phases = phase_info.get("num_phases")
            if num_phases is None and data.get("subtasks"):
                num_phases = len(data["subtasks"][0])
            if num_phases is None:
                num_phases = 2
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
    print("Subtask descriptions are now diverse (2-phase: Approach → Grasp).")


if __name__ == "__main__":
    main()
