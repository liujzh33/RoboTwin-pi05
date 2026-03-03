import os
import json
import glob
import random
from tqdm import tqdm

# ================= 配置区域 =================
# 你的数据路径
DATA_ROOT = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/beat_block_hammer-aloha-agilex_randomized_500-200"

# === 超级扩充词库 ===
# 我们混合了不同的动词、形容词和句式结构
PHASE_VARIANTS = {
    # Phase 0: 接近/对齐 (Reach)
    0: [
        "Reach for the hammer",
        "Move the arm towards the silver hammer",
        "Approach the tool on the table",
        "Position the gripper near the hammer handle",
        "Go to the hammer",
        "Align the gripper with the tool",
        "Move end-effector to the hammer",
        "Reach out to the object",
        "Locate and approach the hammer",
        "Get ready to grab the hammer",
        "Move closer to the silver tool",
        "Target the hammer handle"
    ],
    # Phase 1: 抓取 (Grasp)
    1: [
        "Grasp the hammer",
        "Grab the handle tightly",
        "Close the gripper on the tool",
        "Seize the hammer",
        "Pick up the handle",
        "Securely hold the hammer",
        "Actuate gripper to close",
        "Firmly grip the black handle",
        "Clench the hammer",
        "Grasp the object",
        "Close gripper",
        "Grab the silver hammer"
    ],
    # Phase 2: 抬起 (Lift)
    2: [
        "Lift the hammer up",
        "Raise the tool vertically",
        "Pick the hammer up from the table",
        "Elevate the hammer",
        "Lift it up high",
        "Move the hammer upwards",
        "Hoist the tool",
        "Raise the arm",
        "Lift the object off the surface",
        "Bring the hammer up",
        "Execute lifting motion",
        "Clear the table surface"
    ],
    # Phase 3: 敲击 (Strike)
    3: [
        "Strike the block with the hammer",
        "Hit the block",
        "Smash the target block",
        "Beat the block",
        "Hammer the block",
        "Perform a downward strike",
        "Hit the blue block hard",
        "Strike down on the object",
        "Use the hammer to hit the target",
        "Swing down onto the block",
        "Crash into the block",
        "Apply force to the block"
    ]
}
# ===========================================

def main():
    if not os.path.exists(DATA_ROOT):
        print(f"Error: Data root not found: {DATA_ROOT}")
        return

    episode_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "episode_*")))
    print(f"Found {len(episode_dirs)} episodes. Generating diverse instructions...")

    success_count = 0

    for ep_dir in tqdm(episode_dirs):
        json_path = os.path.join(ep_dir, "instructions.json")
        
        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # 获取指令数量 (例如 100 条)
            instructions = data.get("instructions", [])
            if not instructions:
                continue

            # === 核心逻辑：随机混合生成 ===
            new_subtasks_list = []
            
            for _ in range(len(instructions)):
                # 随机组合，产生 "蒙太奇" 效果
                # 例如：Phase0(详细) + Phase1(简洁) + Phase2(动作) + Phase3(物体特征)
                variant = [
                    random.choice(PHASE_VARIANTS[0]), 
                    random.choice(PHASE_VARIANTS[1]), 
                    random.choice(PHASE_VARIANTS[2]), 
                    random.choice(PHASE_VARIANTS[3])  
                ]
                new_subtasks_list.append(variant)

            # 覆盖旧数据
            data["subtasks"] = new_subtasks_list

            # 保存
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            success_count += 1

        except Exception as e:
            print(f"Error processing {ep_dir}: {e}")

    print(f"\nDone! Successfully updated {success_count}/{len(episode_dirs)} episodes.")
    print("Diverse descriptions generated without using LLM.")

if __name__ == "__main__":
    main()