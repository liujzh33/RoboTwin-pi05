"""
beat_block_hammer 子任务描述丰富化（CycleVLA 风格，动词开头）
4 阶段：找锤子 → 抓锤子 → 移动到红块上方 → 敲击。
"""

import os
import json
import glob
import random
from pathlib import Path
from tqdm import tqdm

DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/beat_block_hammer-aloha-agilex_randomized_500-200"

PHASE_VARIANTS = {
    # Phase 0: Approach Hammer
    0: [
        "Move the gripper above the hammer.",
        "Position the gripper over the hammer handle.",
        "Navigate the arm towards the hammer.",
        "Bring the end-effector above the tool.",
        "Hover the gripper over the hammer.",
        "Direct the arm to the hammer on the table.",
        "Align the gripper with the hammer.",
        "Reach the space directly above the hammer.",
        "Move towards the hammer to pick it up.",
        "Approach the hammer from above.",
    ],
    # Phase 1: Grasp Hammer
    1: [
        "Close the gripper to grasp the hammer.",
        "Grasp the hammer firmly.",
        "Secure the hammer with the gripper.",
        "Pick up the hammer from the table.",
        "Clamp down on the hammer handle.",
        "Take hold of the hammer.",
        "Close the fingers to hold the tool.",
        "Squeeze the gripper around the hammer.",
        "Grip the hammer tightly.",
        "Engage the gripper to secure the hammer.",
    ],
    # Phase 2: Move Hammer above Block
    2: [
        "Move the hammer above the red block.",
        "Transport the hammer over the red cube.",
        "Navigate the tool to the top of the red block.",
        "Position the grasped hammer above the target block.",
        "Carry the hammer over to the red cube.",
        "Bring the hammer directly above the red block.",
        "Align the hammer with the red target.",
        "Hover the hammer over the red object.",
        "Move the arm holding the hammer towards the red block.",
        "Aim the hammer at the top of the red cube.",
    ],
    # Phase 3: Hit Block
    3: [
        "Move the hammer down to hit the red block.",
        "Strike the red block with the hammer.",
        "Hit the red cube using the hammer.",
        "Swing the hammer down onto the red block.",
        "Beat the red block with the tool.",
        "Drive the hammer down to strike the target.",
        "Smash the top of the red block.",
        "Lower the hammer aggressively to hit the red cube.",
        "Pound the red block with the grasped hammer.",
        "Execute a downward strike on the red block.",
    ],
}

CORRECTION_VARIANTS = {
    # grasp_position_offset (位置偏移抓空)
    "grasp_position_offset": [
        "[Correction] Open the gripper, adjust XY position, and retry grasping.",
        "[Correction] Release, correct XY alignment, and try to grasp again.",
        "[Correction] Open the fingers, shift XY position, and regrasp.",
        "[Correction] Unclasp, realign in the XY plane, and attempt grasp.",
        "[Correction] Open gripper, fix XY offset, and grasp once more.",
    ],
    # grasp_orientation_mismatch (姿态角度不对导致碰撞)
    "grasp_orientation_mismatch": [
        "[Correction] Open the gripper, adjust wrist rotation, and retry grasping.",
        "[Correction] Release, fix wrist angle, and try to grasp again.",
        "[Correction] Open the fingers, correct orientation, and regrasp.",
        "[Correction] Unclasp, rotate the wrist to align, and attempt grasp.",
        "[Correction] Open gripper, tune the rotation, and grasp once more.",
    ],
    # premature_close (在半空中提前闭合)
    "premature_close": [
        "[Correction] Open the gripper, move down to the target, and close it.",
        "[Correction] Release, descend to the correct height, and grasp.",
        "[Correction] Open the fingers, lower to the object, and close.",
        "[Correction] Unclasp, move further down, and secure the grip.",
        "[Correction] Open gripper, drop to the target level, and shut it.",
    ],
    # grasp_slip (搬运途中打滑掉落)
    "grasp_slip": [
        "[Correction] Track the dropped object, return to it, and regrasp.",
        "[Correction] Locate the fallen object, move back, and grab it.",
        "[Correction] Find the dropped item, go to it, and secure grip.",
        "[Correction] Trace the slipped object, approach it, and regrasp.",
        "[Correction] Track the item, return to its position, and grasp again.",
    ],
}

BASE_PHASE_DESCS = {
    "Move the gripper above the hammer.": 0,
    "Close the gripper to grasp the hammer.": 1,
    "Move the hammer above the red block.": 2,
    "Move the hammer down to hit the red block.": 3,
}


def main(data_root=None):
    root = (data_root or DATA_ROOT).rstrip("/")
    if not os.path.exists(root):
        print(f"Error: Data root not found: {root}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(root, "episode_*")))
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    print(f"Found {len(episode_dirs)} episodes. Enriching (4 phases: Approach hammer → Grasp → Move above block → Hit)...")

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
                instructions = ["Use the hammer to hit the red block."]
                data["instructions"] = instructions

            phase_info = data.get("phase_info", {}) or {}
            subtasks = data.get("subtasks") or []

            # 若没有现成的 subtasks（老数据），退回到旧的 4 阶段生成逻辑
            if not subtasks:
                num_phases = phase_info.get("num_phases") or 4
                new_subtasks_list = []
                for _ in range(len(instructions)):
                    variant = []
                    for phase_idx in range(num_phases):
                        if phase_idx in PHASE_VARIANTS:
                            variant.append(random.choice(PHASE_VARIANTS[phase_idx]))
                        else:
                            # 理论上不会走到这里；保底用 Phase 0 文本
                            variant.append(random.choice(PHASE_VARIANTS[0]))
                    new_subtasks_list.append(variant)
                data["subtasks"] = new_subtasks_list
            else:
                # Recovery/多阶段数据：基于现有 subtasks + error_type 丰富 correction 与 subtask
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

                def _infer_phase_idx(text: str) -> int | None:
                    s = (text or "").strip()
                    if not s or "[MASKED]" in s:
                        return None
                    # 拆掉 [Correction] / [Subtask] 包装，只看子任务部分
                    if " [Subtask] " in s:
                        s = s.split(" [Subtask] ")[-1].strip()
                    for base_desc, idx_phase in BASE_PHASE_DESCS.items():
                        if base_desc in s or s == base_desc:
                            return idx_phase
                    return None

                new_subtasks_list: list[list[str]] = []
                for phases in subtasks:
                    variant_phases: list[str] = []
                    for phase_text in phases:
                        txt = phase_text or ""
                        if "[MASKED]" in txt:
                            # 不对 MASKED 段做任何丰富化
                            variant_phases.append(txt)
                            continue

                        phase_idx = _infer_phase_idx(txt)
                        if phase_idx is None:
                            # 无法识别阶段，保留原文本以免破坏语义
                            variant_phases.append(txt)
                            continue

                        # 选择子任务表述
                        subtask_variant = random.choice(PHASE_VARIANTS[phase_idx])

                        # 是否需要加入 / 更新 Correction 文本
                        if "[Correction]" in txt and error_type in CORRECTION_VARIANTS:
                            corr_variant = random.choice(CORRECTION_VARIANTS[error_type])
                            enriched = f"{corr_variant} [Subtask] {subtask_variant}"
                        else:
                            # 普通子任务阶段：只替换子任务描述
                            enriched = subtask_variant

                        variant_phases.append(enriched)

                    new_subtasks_list.append(variant_phases)

                data["subtasks"] = new_subtasks_list
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            success_count += 1
        except Exception as e:
            print(f"Error processing {ep_dir}: {e}")

    print(f"\nDone! Updated {success_count}/{len(episode_dirs)} episodes.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="beat_block_hammer 子任务语言丰富化（clean/random 共用一套 PHASE_VARIANTS）")
    p.add_argument("--data_dir", type=str, default=None, help="processed_data 目录，如 processed_data/beat_block_hammer-aloha-agilex_clean_50-50 或 ...-randomized_500-500")
    args = p.parse_args()
    main(data_root=args.data_dir)
