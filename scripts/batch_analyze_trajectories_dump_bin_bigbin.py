#!/usr/bin/env python3
"""
批量可视化 dump_bin_bigbin 轨迹。

参考 batch_analyze_trajectories_blocks_ranking_rgb.py：
- 输入 processed_data 路径
- 对每个 episode_* 调用 analyze_trajectory_dump_bin_bigbin.py
- 在输出目录下生成一张张 png
"""

import sys
import subprocess
from pathlib import Path
import re

ANALYZE_SCRIPT_NAME = "analyze_trajectory_dump_bin_bigbin.py"


def process_all_episodes(data_dir: str, output_dir: str):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 自动查找分析脚本路径（与本脚本同目录）
    current_dir = Path(__file__).resolve().parent
    script_path = current_dir / ANALYZE_SCRIPT_NAME

    if not script_path.exists():
        print(f"Error: Cannot find analysis script at {script_path}")
        print(f"Please check if '{ANALYZE_SCRIPT_NAME}' exists in the 'scripts' folder.")
        return

    episodes = sorted(list(data_path.glob("episode_*")))
    print(f"Found {len(episodes)} episodes in {data_path}")
    print(f"Using analysis script: {script_path}")
    print(f"Saving results to: {out_path}")

    success_count = 0
    fail_count = 0

    for ep_dir in episodes:
        if not ep_dir.is_dir():
            continue

        match = re.search(r"episode_(\d+)", ep_dir.name)
        if not match:
            continue
        ep_num = match.group(1)

        hdf5_file = ep_dir / f"{ep_dir.name}.hdf5"
        if not hdf5_file.exists():
            continue

        save_file = out_path / f"analysis_episode_{ep_num}.png"

        print(f"Processing Episode {ep_num}...", end=" ", flush=True)
        try:
            cmd = [
                sys.executable,
                str(script_path),
                str(hdf5_file),
                "--save",
                str(save_file),
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Done.")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print("Failed.")
            print(f"  Error output: {e.stderr.strip()}")
            fail_count += 1

    print(f"\nProcessing complete. Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="processed_data/.../dump_bin_bigbin-...-N")
    parser.add_argument("--output_dir", required=True, help="directory to save analysis images")
    args = parser.parse_args()

    process_all_episodes(args.data_dir, args.output_dir)

