#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path
import re

# [关键修改] 这里必须指向您截图里那个存在的、用于画单张图的脚本
# 您的截图显示该文件名为: analyze_trajectory_blocks_ranking_rgb.py
ANALYZE_SCRIPT_NAME = "analyze_trajectory_blocks_ranking_rgb.py"

def process_all_episodes(data_dir, output_dir):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. 自动查找脚本路径
    # 优先找当前脚本所在目录下的 analyze_trajectory_blocks_ranking_rgb.py
    current_dir = Path(__file__).resolve().parent
    script_path = current_dir / ANALYZE_SCRIPT_NAME
    
    if not script_path.exists():
        print(f"Error: Cannot find analysis script at {script_path}")
        print(f"Please check if '{ANALYZE_SCRIPT_NAME}' exists in the 'scripts' folder.")
        return

    episodes = sorted(list(data_path.glob("episode_*")))
    print(f"Found {len(episodes)} episodes.")
    print(f"Using analysis script: {script_path}")
    print(f"Saving results to: {out_path}")

    success_count = 0
    fail_count = 0

    for ep_dir in episodes:
        if not ep_dir.is_dir(): continue
        
        # 提取编号
        match = re.search(r"episode_(\d+)", ep_dir.name)
        if not match: continue
        ep_num = match.group(1)
        
        hdf5_file = ep_dir / f"{ep_dir.name}.hdf5"
        if not hdf5_file.exists(): continue

        save_file = out_path / f"analysis_episode_{ep_num}.png"
        
        print(f"Processing Episode {ep_num}...", end=" ", flush=True)
        try:
            # 调用画图脚本
            cmd = [
                sys.executable, str(script_path),
                str(hdf5_file),
                "--save", str(save_file)
            ]
            
            # 捕获输出以便出错时打印
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Done.")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print("Failed.")
            print(f"  Error output: {e.stderr.strip()}") # 打印具体报错原因
            fail_count += 1

    print(f"\nProcessing complete. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    process_all_episodes(args.data_dir, args.output_dir)