import os
import json
import glob
import random
from tqdm import tqdm

# ================= 配置区域 =================
# 您的数据路径 (请确保路径正确)
DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/blocks_ranking_rgb-aloha-agilex_randomized_500-200"

# === 7阶段多样化词库 (对应 3次抓取 + 3次放置 + 1次复位) ===
PHASE_VARIANTS = {
    # Phase 0: 抓取红色 (Pick Red)
    0: [
        "Pick up the red block",
        "Grasp the red cube",
        "Grab the red object",
        "Retrieve the red block from the table",
        "Secure the red cube with the gripper",
        "Take hold of the red block",
        "Reach for and grasp the red block",
        "Initiate the task by picking up the red block",
        "Locate and grab the red cube"
    ],
    # Phase 1: 放置红色 (Place Red - Far Left)
    1: [
        "Place the red block on the far left",
        "Set the red block down at the leftmost position",
        "Position the red cube on the far left side",
        "Drop the red block to start the sequence on the left",
        "Move the red block to the far left and release it",
        "Establish the starting point by placing red on the left",
        "Put the red object on the extreme left",
        "Align the red block on the left side of the workspace"
    ],
    # Phase 2: 抓取绿色 (Pick Green)
    2: [
        "Pick up the green block",
        "Grasp the green cube",
        "Grab the green object",
        "Now, take the green block",
        "Retrieve the green cube",
        "Secure the green block",
        "Move to grasp the green block",
        "Select the green block next",
        "Pick up the green one"
    ],
    # Phase 3: 放置绿色 (Place Green - Next to Red)
    3: [
        "Place the green block next to the red block",
        "Put the green block beside the red one",
        "Position the green cube to the right of the red block",
        "Set the green block adjacent to the red block",
        "Align the green block next to the red cube",
        "Drop the green object beside the red one",
        "Place green directly next to red",
        "Continue the sequence by placing green next to red"
    ],
    # Phase 4: 抓取蓝色 (Pick Blue)
    4: [
        "Pick up the blue block",
        "Grasp the blue cube",
        "Grab the blue object",
        "Finally, take the blue block",
        "Retrieve the blue cube",
        "Secure the blue block with the gripper",
        "Pick up the last block, the blue one",
        "Grasp the blue square",
        "Take hold of the blue cube"
    ],
    # Phase 5: 放置蓝色 (Place Blue - Next to Green)
    5: [
        "Place the blue block next to the green block",
        "Put the blue block beside the green one",
        "Position the blue cube to the right of the green block",
        "Set the blue block adjacent to the green block",
        "Complete the line by placing blue next to green",
        "Align the blue block next to the green cube",
        "Finish the arrangement by putting blue beside green",
        "Place the blue object at the end of the row"
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
        "Finish task and reset"
    ]
}
# ===========================================

def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root not found: {DATA_ROOT}")
        return

    # 查找所有 episode 文件夹
    episode_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "episode_*")))
    
    # 按照数字顺序排序 (防止 episode_10 排在 episode_2 前面)
    episode_dirs.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

    print(f"Found {len(episode_dirs)} episodes in {DATA_ROOT}")
    print("Generating diverse instructions (Sequence: Red -> Green -> Blue)...")

    success_count = 0
    updated_count = 0

    for ep_dir in tqdm(episode_dirs):
        json_path = os.path.join(ep_dir, "instructions.json")
        
        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            instructions = data.get("instructions", [])
            # 如果没有instructions字段，初始化一个默认的
            if not instructions:
                instructions = ["Sort the blocks by color: red, green, blue."]
                data["instructions"] = instructions

            new_subtasks_list = []
            
            # 为每一条 high-level instruction 生成对应的 subtask 序列
            for _ in range(len(instructions)):
                # 生成 7 个阶段的描述 (Phase 0 - Phase 6)
                # 确保每个阶段都存在于 PHASE_VARIANTS 中
                variant = []
                for phase_idx in range(7):
                    if phase_idx in PHASE_VARIANTS:
                        variant.append(random.choice(PHASE_VARIANTS[phase_idx]))
                    else:
                        variant.append(f"Phase {phase_idx} action") # Fallback
                
                new_subtasks_list.append(variant)

            # 更新数据
            data["subtasks"] = new_subtasks_list

            # 写回文件
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            success_count += 1

        except Exception as e:
            print(f"Error processing {ep_dir}: {e}")

    print(f"\nDone! Successfully updated {success_count}/{len(episode_dirs)} episodes.")
    print("Instructions have been enriched with diverse vocabulary while strictly maintaining the R->G->B sorting logic.")

if __name__ == "__main__":
    main()