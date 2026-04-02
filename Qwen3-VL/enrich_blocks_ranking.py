"""
blocks_ranking_rgb_v1 子任务描述丰富化（CycleVLA 风格，动词开头）
13 阶段：红→左, 绿→中, 蓝→右（每色：接近→抓取→搬运→释放）+ 复位。
"""

import os
import json
import glob
import random
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/blocks_ranking_rgb-aloha-agilex_randomized_500-200"

# === 13 阶段多样化词库 (红->左, 绿->中, 蓝->右) ===
PHASE_VARIANTS = {
    0: [
        "Move the gripper above the red block.",
        "Position the gripper over the red cube.",
        "Navigate the arm above the red block.",
        "Bring the end-effector over the red object.",
        "Hover the gripper above the red block.",
        "Direct the gripper to the top of the red cube.",
        "Align the gripper above the red block.",
        "Reach the space directly above the red cube.",
        "Move towards the top of the red block.",
        "Approach the red block from above.",
    ],
    1: [
        "Close the gripper to grasp the red block.",
        "Grasp the red block by closing the fingers.",
        "Secure the red cube with the gripper.",
        "Pick up the red block.",
        "Clamp down on the red object.",
        "Take hold of the red block.",
        "Close the fingers around the red cube.",
        "Squeeze the gripper to hold the red block.",
        "Grip the red block tightly.",
        "Engage the gripper to pick the red cube.",
    ],
    2: [
        "Move the gripper to the left target position while holding the red block.",
        "Transport the red block to the left side.",
        "Carry the red cube to the left target.",
        "Shift the grasped red block to the left.",
        "Navigate to the left position with the red block.",
        "Bring the red object to the target on the left.",
        "Move the red cube towards the left placement area.",
        "Transfer the red block to the left.",
        "Relocate the red block to the left target.",
        "Guide the red cube to the left side of the workspace.",
    ],
    3: [
        "Open the gripper to release the red block.",
        "Release the red block by opening the fingers.",
        "Drop the red cube at the current position.",
        "Let go of the red block.",
        "Open the fingers to place the red object.",
        "Unclasp the gripper to release the red block.",
        "Place the red block down by opening the gripper.",
        "Release grip on the red cube.",
        "Set the red block down and open the gripper.",
        "Free the red block from the gripper.",
    ],
    4: [
        "Move the gripper above the green block.",
        "Position the gripper over the green cube.",
        "Navigate the arm above the green block.",
        "Bring the end-effector over the green object.",
        "Hover the gripper above the green block.",
        "Direct the gripper to the top of the green cube.",
        "Align the gripper above the green block.",
        "Reach the space directly above the green cube.",
        "Move towards the top of the green block.",
        "Approach the green block from above.",
    ],
    5: [
        "Close the gripper to grasp the green block.",
        "Grasp the green block by closing the fingers.",
        "Secure the green cube with the gripper.",
        "Pick up the green block.",
        "Clamp down on the green object.",
        "Take hold of the green block.",
        "Close the fingers around the green cube.",
        "Squeeze the gripper to hold the green block.",
        "Grip the green block tightly.",
        "Engage the gripper to pick the green cube.",
    ],
    6: [
        "Move the gripper to the middle target position while holding the green block.",
        "Transport the green block to the center.",
        "Carry the green cube to the middle target.",
        "Shift the grasped green block to the center.",
        "Navigate to the middle position with the green block.",
        "Bring the green object to the target in the middle.",
        "Move the green cube towards the central placement area.",
        "Transfer the green block to the middle.",
        "Relocate the green block to the center target.",
        "Guide the green cube to the middle of the workspace.",
    ],
    7: [
        "Open the gripper to release the green block.",
        "Release the green block by opening the fingers.",
        "Drop the green cube at the current position.",
        "Let go of the green block.",
        "Open the fingers to place the green object.",
        "Unclasp the gripper to release the green block.",
        "Place the green block down by opening the gripper.",
        "Release grip on the green cube.",
        "Set the green block down and open the gripper.",
        "Free the green block from the gripper.",
    ],
    8: [
        "Move the gripper above the blue block.",
        "Position the gripper over the blue cube.",
        "Navigate the arm above the blue block.",
        "Bring the end-effector over the blue object.",
        "Hover the gripper above the blue block.",
        "Direct the gripper to the top of the blue cube.",
        "Align the gripper above the blue block.",
        "Reach the space directly above the blue cube.",
        "Move towards the top of the blue block.",
        "Approach the blue block from above.",
    ],
    9: [
        "Close the gripper to grasp the blue block.",
        "Grasp the blue block by closing the fingers.",
        "Secure the blue cube with the gripper.",
        "Pick up the blue block.",
        "Clamp down on the blue object.",
        "Take hold of the blue block.",
        "Close the fingers around the blue cube.",
        "Squeeze the gripper to hold the blue block.",
        "Grip the blue block tightly.",
        "Engage the gripper to pick the blue cube.",
    ],
    10: [
        "Move the gripper to the right target position while holding the blue block.",
        "Transport the blue block to the right side.",
        "Carry the blue cube to the right target.",
        "Shift the grasped blue block to the right.",
        "Navigate to the right position with the blue block.",
        "Bring the blue object to the target on the right.",
        "Move the blue cube towards the right placement area.",
        "Transfer the blue block to the right.",
        "Relocate the blue block to the right target.",
        "Guide the blue cube to the right side of the workspace.",
    ],
    11: [
        "Open the gripper to release the blue block.",
        "Release the blue block by opening the fingers.",
        "Drop the blue cube at the current position.",
        "Let go of the blue block.",
        "Open the fingers to place the blue object.",
        "Unclasp the gripper to release the blue block.",
        "Place the blue block down by opening the gripper.",
        "Release grip on the blue cube.",
        "Set the blue block down and open the gripper.",
        "Free the blue block from the gripper.",
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

FALLBACK_VARIANTS = ["Complete the task.", "Continue with the next step."]

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
    "Move the gripper above the red block.": 0,
    "Close the gripper to grasp the red block.": 1,
    "Move the gripper to the left target position while holding the red block.": 2,
    "Open the gripper to release the red block.": 3,
    "Move the gripper above the green block.": 4,
    "Close the gripper to grasp the green block.": 5,
    "Move the gripper to the middle target position while holding the green block.": 6,
    "Open the gripper to release the green block.": 7,
    "Move the gripper above the blue block.": 8,
    "Close the gripper to grasp the blue block.": 9,
    "Move the gripper to the right target position while holding the blue block.": 10,
    "Open the gripper to release the blue block.": 11,
    "Return to a neutral position.": 12,
}

def main(data_root=None):
    root = (data_root or DATA_ROOT).rstrip("/")
    if not os.path.exists(root):
        print(f"Error: Data root not found: {root}")
        return

    # 查找所有 episode 文件夹
    episode_dirs = sorted(glob.glob(os.path.join(root, "episode_*")))
    
    # 按照数字顺序排序 (防止 episode_10 排在 episode_2 前面)
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

    print(f"Found {len(episode_dirs)} episodes in {root}")
    print("Enriching subtasks (13 phases: Red->Left, Green->Middle, Blue->Right)...")

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
                instructions = ["Sort the blocks by color: red on the left, green in the middle, blue on the right."]
                data["instructions"] = instructions

            phase_info = data.get("phase_info", {}) or {}
            subtasks = data.get("subtasks") or []

            # 老数据无 subtasks：按阶段数直接生成
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
                # recovery/多阶段：保留 MASKED，按 error_type 丰富 correction 文本
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
    print("Subtask descriptions are now diverse (CycleVLA style) for Red->Left, Green->Middle, Blue->Right.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="blocks_ranking_rgb_v1 子任务语言丰富化（clean/random 共用一套）")
    p.add_argument("--data_dir", type=str, default=None, help="processed_data 目录")
    args = p.parse_args()
    main(data_root=args.data_dir)