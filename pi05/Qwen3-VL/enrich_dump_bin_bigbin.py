"""
dump_bin_bigbin 子任务描述丰富化脚本

与 enrich_blocks_ranking_size.py 同理，用多样化词库替换单一表述。
- 3 阶段：单臂（靠近 / 抓住 / 倾倒）
- 6 阶段：双臂（右靠近 / 右抓 / 放中间 / 左靠近 / 左抓 / 倾倒）
"""

import os
import json
import glob
import random
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/dump_bin_bigbin-aloha-agilex_randomized_500-200"

# 若某 episode 阶段数不在 3/6 时的兜底
FALLBACK_VARIANTS = [
    "Complete the task.",
    "Continue with the next step.",
    "Proceed to the next phase.",
]

# === 3 阶段多样化词库（单臂：靠近 / 抓住 / 倾倒）===
PHASE_VARIANTS_3 = {
    # Phase 0: 靠近容器
    0: [
        "Move the left arm close to the container.",
        "Approach the container with the left arm.",
        "Bring the end-effector near the container.",
        "Reach towards the container with the gripper.",
        "Position the left gripper near the container.",
        "Drive the arm to the container area.",
        "Move smoothly towards the container.",
        "Get the gripper above the container.",
        "Align the end-effector with the container.",
        "Reach for the container and position above it.",
    ],
    # Phase 1: 抓住容器
    1: [
        "Grasp the container firmly with the gripper.",
        "Close the gripper to securely hold the container.",
        "Tighten the gripper to grab the container.",
        "Clamp the container with a stable grip.",
        "Lock the container inside the gripper.",
        "Securely pinch the container.",
        "Take hold of the container with the gripper.",
        "Close the gripper around the container.",
        "Secure the container in the gripper.",
        "Grab and hold the container firmly.",
    ],
    # Phase 2: 倾倒容器
    2: [
        "Tilt the container to pour out its contents.",
        "Rotate the wrist to pour the container.",
        "Pour the contents from the container into the bin.",
        "Carefully tilt the container to dump everything out.",
        "Tip the container so that its contents fall down.",
        "Lean the container over the bin to pour.",
        "Dump the container contents into the bin.",
        "Tilt the container and pour into the bin.",
        "Tip the container to empty it into the bin.",
        "Rotate the container to pour its contents out.",
    ],
}

# === 6 阶段多样化词库（双臂：R-Approach / R-Grasp / R-Place / L-Approach / L-Grasp / L-Pour）===
PHASE_VARIANTS_6 = {
    # Phase 0: 右臂靠近容器
    0: [
        "Use the right arm to move close to the container.",
        "Approach the container with the right arm.",
        "Bring the right gripper near the container.",
        "Reach towards the container using the right arm.",
        "Position the right end-effector near the container.",
        "Move the right arm to the container.",
        "Drive the right gripper to the container area.",
        "Align the right end-effector with the container.",
        "Get the right arm above the container.",
        "Reach for the container with the right arm.",
    ],
    # Phase 1: 右臂抓住容器
    1: [
        "Use the right gripper to grasp the container.",
        "Close the right gripper to hold the container firmly.",
        "Clamp the container securely with the right gripper.",
        "Grab the container using the right arm.",
        "Stably hold the container with the right gripper.",
        "Take hold of the container with the right gripper.",
        "Secure the container with the right hand.",
        "Close the right gripper around the container.",
        "Grab the container firmly with the right arm.",
        "Pinch the container with the right gripper.",
    ],
    # Phase 2: 右臂放置到中间
    2: [
        "Place the container in the middle of the workspace.",
        "Move the container to the center area and release it.",
        "Reposition the container to the middle of the table.",
        "Set the container down at the central position.",
        "Drop the container at the middle spot.",
        "Put the container in the center of the workspace.",
        "Release the container in the middle of the table.",
        "Position the container in the middle and release.",
        "Place the container at the central spot.",
        "Set down the container in the middle area.",
    ],
    # Phase 3: 左臂靠近中间的容器
    3: [
        "Move the left arm close to the container in the middle.",
        "Approach the centered container with the left arm.",
        "Bring the left gripper near the container at the center.",
        "Reach towards the container in the middle using the left arm.",
        "Position the left end-effector above the container in the middle.",
        "Get the left arm above the container in the center.",
        "Align the left gripper with the container in the middle.",
        "Move the left arm to the container at the center.",
        "Drive the left gripper to the centered container.",
        "Reach for the container in the middle with the left arm.",
    ],
    # Phase 4: 左臂抓住容器
    4: [
        "Grasp the container with the left gripper.",
        "Close the left gripper to hold the container firmly.",
        "Clamp the container securely using the left arm.",
        "Grab the container at the center with the left gripper.",
        "Stably hold the container using the left hand.",
        "Take hold of the container with the left gripper.",
        "Secure the container with the left arm.",
        "Close the left gripper around the container.",
        "Pinch the container with the left gripper.",
        "Grab the container firmly with the left hand.",
    ],
    # Phase 5: 左臂倾倒容器
    5: [
        "Tilt the container with the left arm to pour out its contents.",
        "Rotate the left wrist to pour the container.",
        "Pour the container's contents into the bin using the left arm.",
        "Carefully tilt the container with the left arm to dump everything out.",
        "Tip the container over with the left arm so that its contents fall into the bin.",
        "Dump the container into the bin with the left arm.",
        "Tilt the container and pour into the bin with the left arm.",
        "Empty the container into the bin using the left arm.",
        "Tip the container with the left arm to empty it into the bin.",
        "Rotate the container with the left arm to pour out its contents.",
    ],
}


def build_subtasks_for_num_phases(num_phases: int) -> list[str]:
    """根据 num_phases 构造一条子任务序列（每阶段从对应词库随机选一句）"""
    subtasks: list[str] = []
    if num_phases <= 0:
        return subtasks

    if num_phases == 3:
        for phase_idx in range(3):
            candidates = PHASE_VARIANTS_3.get(phase_idx, FALLBACK_VARIANTS)
            subtasks.append(random.choice(candidates))
        return subtasks

    if num_phases >= 6:
        for phase_idx in range(6):
            candidates = PHASE_VARIANTS_6.get(phase_idx, FALLBACK_VARIANTS)
            subtasks.append(random.choice(candidates))
        for phase_idx in range(6, num_phases):
            subtasks.append(random.choice(FALLBACK_VARIANTS))
        return subtasks[:num_phases]

    # 其它阶段数：用 3 阶段模板循环
    for i in range(num_phases):
        phase_idx = i % 3
        candidates = PHASE_VARIANTS_3.get(phase_idx, FALLBACK_VARIANTS)
        subtasks.append(random.choice(candidates))
    return subtasks


def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root not found: {DATA_ROOT}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "episode_*")))
    # 数字顺序排序
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    print(f"Found {len(episode_dirs)} episodes in {DATA_ROOT}")
    print("Enriching dump_bin_bigbin subtasks (3-phase: Approach/Grasp/Pour; 6-phase: R-Approach, R-Grasp, Place, L-Approach, L-Grasp, Pour)...")

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
                instructions = ["Pick up the container and dump it into the bin."]
                data["instructions"] = instructions

            # 与 enrich_blocks_ranking_size 一致：优先 phase_info，否则从已有 subtasks 推断
            phase_info = data.get("phase_info", {})
            num_phases = phase_info.get("num_phases")
            if num_phases is None and data.get("subtasks"):
                num_phases = len(data["subtasks"][0])
            if num_phases is None:
                num_phases = 3
            num_phases = int(num_phases)

            new_subtasks_list = []
            for _ in range(len(instructions)):
                subtasks = build_subtasks_for_num_phases(num_phases)
                new_subtasks_list.append(subtasks)

            data["subtasks"] = new_subtasks_list

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            success_count += 1

        except Exception as e:
            print(f"Error processing {ep_dir}: {e}")

    print(f"\nDone! Successfully updated {success_count}/{len(episode_dirs)} episodes.")
    print("Subtask descriptions are now diverse while keeping order: 3-phase or 6-phase (R-Approach → R-Grasp → Place → L-Approach → L-Grasp → Pour).")


if __name__ == "__main__":
    main()

