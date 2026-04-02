"""
blocks_ranking_size_v1 子任务描述丰富化（CycleVLA 风格，动词开头）
13 阶段：小→最右, 中→中间, 大→最左（每尺寸：接近→抓取→搬运→释放）+ 复位。
"""

import os
import json
import glob
import random
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/blocks_ranking_size-aloha-agilex_randomized_500-200"

# === 13 阶段多样化词库 (小->最右, 中->中间, 大->最左) ===
PHASE_VARIANTS = {
    0: [
        "Move the gripper above the small block.",
        "Position the gripper over the smallest cube.",
        "Navigate the arm above the small block.",
        "Bring the end-effector over the small object.",
        "Hover the gripper above the small block.",
        "Direct the gripper to the top of the small cube.",
        "Align the gripper above the small block.",
        "Reach the space directly above the smallest cube.",
        "Move towards the top of the small block.",
        "Approach the small block from above.",
    ],
    1: [
        "Close the gripper to grasp the small block.",
        "Grasp the small block by closing the fingers.",
        "Secure the small cube with the gripper.",
        "Pick up the small block.",
        "Clamp down on the smallest object.",
        "Take hold of the small block.",
        "Close the fingers around the small cube.",
        "Squeeze the gripper to hold the small block.",
        "Grip the small block tightly.",
        "Engage the gripper to pick the smallest cube.",
    ],
    2: [
        "Move the gripper to the far right target position while holding the small block.",
        "Transport the small block to the far right.",
        "Carry the small cube to the rightmost target.",
        "Shift the grasped small block to the extreme right.",
        "Navigate to the far right position with the small block.",
        "Bring the small object to the target on the far right.",
        "Move the small cube towards the rightmost placement area.",
        "Transfer the small block to the far right.",
        "Relocate the small block to the extreme right target.",
        "Guide the small cube to the right end of the workspace.",
    ],
    3: [
        "Open the gripper to release the small block.",
        "Release the small block by opening the fingers.",
        "Drop the small cube at the current position.",
        "Let go of the small block.",
        "Open the fingers to place the small object.",
        "Unclasp the gripper to release the small block.",
        "Place the small block down by opening the gripper.",
        "Release grip on the smallest cube.",
        "Set the small block down and open the gripper.",
        "Free the small block from the gripper.",
    ],
    4: [
        "Move the gripper above the medium block.",
        "Position the gripper over the medium cube.",
        "Navigate the arm above the medium block.",
        "Bring the end-effector over the medium-sized object.",
        "Hover the gripper above the medium block.",
        "Direct the gripper to the top of the medium cube.",
        "Align the gripper above the medium block.",
        "Reach the space directly above the medium cube.",
        "Move towards the top of the medium block.",
        "Approach the medium block from above.",
    ],
    5: [
        "Close the gripper to grasp the medium block.",
        "Grasp the medium block by closing the fingers.",
        "Secure the medium cube with the gripper.",
        "Pick up the medium block.",
        "Clamp down on the medium object.",
        "Take hold of the medium block.",
        "Close the fingers around the medium cube.",
        "Squeeze the gripper to hold the medium block.",
        "Grip the medium block tightly.",
        "Engage the gripper to pick the medium cube.",
    ],
    6: [
        "Move the gripper to the middle target position while holding the medium block.",
        "Transport the medium block to the center.",
        "Carry the medium cube to the middle target.",
        "Shift the grasped medium block to the center.",
        "Navigate to the middle position with the medium block.",
        "Bring the medium object to the target in the middle.",
        "Move the medium cube towards the central placement area.",
        "Transfer the medium block to the middle.",
        "Relocate the medium block to the center target.",
        "Guide the medium cube to the middle of the workspace.",
    ],
    7: [
        "Open the gripper to release the medium block.",
        "Release the medium block by opening the fingers.",
        "Drop the medium cube at the current position.",
        "Let go of the medium block.",
        "Open the fingers to place the medium object.",
        "Unclasp the gripper to release the medium block.",
        "Place the medium block down by opening the gripper.",
        "Release grip on the medium cube.",
        "Set the medium block down and open the gripper.",
        "Free the medium block from the gripper.",
    ],
    8: [
        "Move the gripper above the large block.",
        "Position the gripper over the largest cube.",
        "Navigate the arm above the large block.",
        "Bring the end-effector over the large object.",
        "Hover the gripper above the large block.",
        "Direct the gripper to the top of the large cube.",
        "Align the gripper above the large block.",
        "Reach the space directly above the largest cube.",
        "Move towards the top of the large block.",
        "Approach the large block from above.",
    ],
    9: [
        "Close the gripper to grasp the large block.",
        "Grasp the large block by closing the fingers.",
        "Secure the large cube with the gripper.",
        "Pick up the large block.",
        "Clamp down on the largest object.",
        "Take hold of the large block.",
        "Close the fingers around the large cube.",
        "Squeeze the gripper to hold the large block.",
        "Grip the large block tightly.",
        "Engage the gripper to pick the largest cube.",
    ],
    10: [
        "Move the gripper to the far left target position while holding the large block.",
        "Transport the large block to the far left.",
        "Carry the large cube to the leftmost target.",
        "Shift the grasped large block to the extreme left.",
        "Navigate to the far left position with the large block.",
        "Bring the large object to the target on the far left.",
        "Move the large cube towards the leftmost placement area.",
        "Transfer the large block to the far left.",
        "Relocate the large block to the extreme left target.",
        "Guide the large cube to the left end of the workspace.",
    ],
    11: [
        "Open the gripper to release the large block.",
        "Release the large block by opening the fingers.",
        "Drop the large cube at the current position.",
        "Let go of the large block.",
        "Open the fingers to place the large object.",
        "Unclasp the gripper to release the large block.",
        "Place the large block down by opening the gripper.",
        "Release grip on the largest cube.",
        "Set the large block down and open the gripper.",
        "Free the large block from the gripper.",
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

FALLBACK_VARIANTS = [
    "Complete the task.",
    "Continue with the next step.",
    "Proceed to the next phase.",
]

CORRECTION_VARIANTS = {
    "grasp_position_offset": [
        "[Correction] Open the gripper, adjust XY position, and retry grasping.",
        "[Correction] Release, correct XY alignment, and try to grasp again.",
        "[Correction] Open the fingers, shift XY position, and regrasp.",
        "[Correction] Unclasp, realign in the XY plane, and attempt grasp.",
        "[Correction] Open gripper, fix XY offset, and grasp once more.",
    ],
    "grasp_orientation_mismatch": [
        "[Correction] Open the gripper, adjust wrist rotation, and retry grasping.",
        "[Correction] Release, fix wrist angle, and try to grasp again.",
        "[Correction] Open the fingers, correct orientation, and regrasp.",
        "[Correction] Unclasp, rotate the wrist to align, and attempt grasp.",
        "[Correction] Open gripper, tune the rotation, and grasp once more.",
    ],
    "premature_close": [
        "[Correction] Open the gripper, move down to the target, and close it.",
        "[Correction] Release, descend to the correct height, and grasp.",
        "[Correction] Open the fingers, lower to the object, and close.",
        "[Correction] Unclasp, move further down, and secure the grip.",
        "[Correction] Open gripper, drop to the target level, and shut it.",
    ],
    "grasp_slip": [
        "[Correction] Track the dropped object, return to it, and regrasp.",
        "[Correction] Locate the fallen object, move back, and grab it.",
        "[Correction] Find the dropped item, go to it, and secure grip.",
        "[Correction] Trace the slipped object, approach it, and regrasp.",
        "[Correction] Track the item, return to its position, and grasp again.",
    ],
}

BASE_PHASE_DESCS = {
    "Move the gripper above the small block.": 0,
    "Close the gripper to grasp the small block.": 1,
    "Move the gripper to the far right target position while holding the small block.": 2,
    "Open the gripper to release the small block.": 3,
    "Move the gripper above the medium block.": 4,
    "Close the gripper to grasp the medium block.": 5,
    "Move the gripper to the middle target position while holding the medium block.": 6,
    "Open the gripper to release the medium block.": 7,
    "Move the gripper above the large block.": 8,
    "Close the gripper to grasp the large block.": 9,
    "Move the gripper to the far left target position while holding the large block.": 10,
    "Open the gripper to release the large block.": 11,
    "Return to a neutral position.": 12,
}


def main(data_root=None):
    root = (data_root or DATA_ROOT).rstrip("/")
    if not os.path.exists(root):
        print(f"Error: Data root not found: {root}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(root, "episode_*")))
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    print(f"Found {len(episode_dirs)} episodes in {root}")
    print("Enriching subtasks (13 phases: small→far right, medium→middle, large→far left)...")

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

            phase_info = data.get("phase_info", {}) or {}
            subtasks = data.get("subtasks") or []

            if not subtasks:
                num_phases = phase_info.get("num_phases")
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
            else:
                ep_name = os.path.basename(ep_dir)
                try:
                    ep_idx = int(ep_name.split("_")[1])
                except Exception:
                    ep_idx = -1

                error_type = ""
                meta_dir = Path(root) / "metadata"
                if ep_idx >= 0 and meta_dir.exists():
                    meta_path = meta_dir / f"episode{ep_idx}_metadata.json"
                    if meta_path.exists():
                        try:
                            with open(meta_path, "r", encoding="utf-8") as mf:
                                meta = json.load(mf)
                            summary = meta.get("episode_summary") or {}
                            etypes = summary.get("error_types") or []
                            if etypes:
                                error_type = etypes[0]
                        except Exception:
                            error_type = ""

                def _infer_phase_idx(text: str):
                    s = (text or "").strip()
                    if not s or "[MASKED]" in s:
                        return None
                    if " [Subtask] " in s:
                        s = s.split(" [Subtask] ")[-1].strip()
                    for base_desc, idx_phase in BASE_PHASE_DESCS.items():
                        if base_desc in s or s == base_desc:
                            return idx_phase
                    return None

                new_subtasks_list = []
                for phases in subtasks:
                    variant_phases = []
                    for phase_text in phases:
                        txt = phase_text or ""
                        if "[MASKED]" in txt:
                            variant_phases.append(txt)
                            continue

                        phase_idx = _infer_phase_idx(txt)
                        if phase_idx is None:
                            variant_phases.append(txt)
                            continue

                        subtask_variant = random.choice(
                            PHASE_VARIANTS.get(phase_idx, FALLBACK_VARIANTS)
                        )
                        if "[Correction]" in txt and error_type in CORRECTION_VARIANTS:
                            corr_variant = random.choice(CORRECTION_VARIANTS[error_type])
                            enriched = f"{corr_variant} [Subtask] {subtask_variant}"
                        else:
                            enriched = subtask_variant
                        variant_phases.append(enriched)
                    new_subtasks_list.append(variant_phases)

                data["subtasks"] = new_subtasks_list

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            success_count += 1

        except Exception as e:
            print(f"Error processing {ep_dir}: {e}")

    print(f"\nDone! Successfully updated {success_count}/{len(episode_dirs)} episodes.")
    print("Subtask descriptions are now diverse (CycleVLA style) for small→far right, medium→middle, large→far left.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="blocks_ranking_size_v1 子任务语言丰富化（clean/random 共用一套）")
    p.add_argument("--data_dir", type=str, default=None, help="processed_data 目录")
    args = p.parse_args()
    main(data_root=args.data_dir)
