#!/usr/bin/env python3
"""
批量分析轨迹脚本

自动处理 processed_data 目录下的所有 episode，生成分析图片
"""

import os
import sys
import subprocess
from pathlib import Path
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_episode_number(episode_path: Path) -> int | None:
    """
    从路径中提取 episode 编号
    
    例如: episode_178/episode_178.hdf5 -> 178
    """
    match = re.search(r"episode_(\d+)", str(episode_path))
    if match:
        return int(match.group(1))
    return None


def find_raw_episode_path(episode_num: int, processed_data_dir: Path = None) -> Path:
    """
    根据 episode 编号和 processed 目录名构建原始 episode 文件路径。
    processed_data_dir 如 .../beat_block_hammer-aloha-agilex_clean_50-50，则用 dataset；
    若含 recovery（如 beat_block_hammer-demo_clean-recovery-66），则用 dataset_recovery。
    """
    if processed_data_dir is not None and "recovery" in processed_data_dir.name:
        # e.g. beat_block_hammer-demo_clean-recovery-66 -> demo_clean
        setting = "demo_clean"
        if "demo_randomized" in processed_data_dir.name:
            setting = "demo_randomized"
        return Path(
            "/mnt/data1/liujingzhi/dataset_recovery/beat_block_hammer/"
            f"{setting}/data/episode{episode_num}.hdf5"
        )
    setting = "aloha-agilex_randomized_500"
    if processed_data_dir is not None and "clean_50" in processed_data_dir.name:
        setting = "aloha-agilex_clean_50"
    return Path(
        "/mnt/data1/liujingzhi/dataset/beat_block_hammer/"
        f"{setting}/data/episode{episode_num}.hdf5"
    )


def process_all_episodes(
    processed_data_dir: Path,
    output_dir: Path = None,
    episode_range: str = None
):
    """
    批量处理所有 episode
    
    Args:
        processed_data_dir: processed_data 目录路径
        output_dir: 输出目录（如果为None，则在每个episode目录下保存为analysis.png）
        episode_range: episode范围，例如 "0-10" 或 "0,5,10"，如果为None则处理所有
    """
    # 查找所有 episode 目录
    episode_dirs = sorted(
        [d for d in processed_data_dir.iterdir() 
         if d.is_dir() and d.name.startswith("episode_")],
        key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0
    )
    
    if len(episode_dirs) == 0:
        print(f"Error: No episode directories found in {processed_data_dir}")
        return
    
    # 解析 episode 范围
    if episode_range:
        if '-' in episode_range:
            start, end = map(int, episode_range.split('-'))
            episode_indices = list(range(start, end + 1))
        else:
            episode_indices = [int(x) for x in episode_range.split(',')]
        episode_dirs = [
            d for d in episode_dirs 
            if int(d.name.split("_")[1]) in episode_indices
        ]
    
    print(f"Found {len(episode_dirs)} episodes to process")
    print(f"Output directory: {output_dir if output_dir else 'Same as episode directory'}")
    print()
    
    # 获取脚本路径
    analyze_script = project_root / "scripts" / "analyze_trajectory_beat_block_hammer .py"
    if not analyze_script.exists():
        print(f"Error: analyze_trajectory_beat_block_hammer .py not found at {analyze_script}")
        return
    
    success_count = 0
    fail_count = 0
    
    for episode_dir in episode_dirs:
        # 查找 HDF5 文件
        hdf5_files = list(episode_dir.glob("episode_*.hdf5"))
        if len(hdf5_files) == 0:
            print(f"⚠️  {episode_dir.name}: No HDF5 file found, skipping")
            fail_count += 1
            continue
        
        hdf5_path = hdf5_files[0]
        
        # 提取 episode 编号
        episode_num = find_episode_number(hdf5_path)
        if episode_num is None:
            print(f"⚠️  {episode_dir.name}: Cannot extract episode number, skipping")
            fail_count += 1
            continue
        
        # 查找原始 episode 文件（根据 data_dir 区分 clean_50 / randomized_500）
        raw_episode_path = find_raw_episode_path(episode_num, processed_data_dir)
        
        # 确定输出路径
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"episode_{episode_num}_analysis.png"
        else:
            save_path = episode_dir / "analysis.png"
        
        print(f"Processing {episode_dir.name} (episode {episode_num})...", end=" ")
        
        # 构建命令
        cmd = [
            sys.executable,
            str(analyze_script),
            str(hdf5_path),
            "--save", str(save_path)
        ]
        
        # 如果原始文件存在，传递路径（虽然现在会自动推断，但显式传递更可靠）
        if raw_episode_path.exists():
            cmd.extend(["--raw_episode", str(raw_episode_path)])
        
        # 运行分析脚本
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(project_root)
            )
            
            if result.returncode == 0:
                print(f"✓ Saved to {save_path}")
                success_count += 1
            else:
                print(f"✗ Error:")
                print(f"  {result.stderr}")
                fail_count += 1
        except Exception as e:
            print(f"✗ Exception: {e}")
            fail_count += 1
    
    print()
    print("=" * 60)
    print(f"Batch processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(episode_dirs)}")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="批量分析轨迹数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理所有 episode，图片保存在各自目录下
  python batch_analyze_trajectories.py --data_dir processed_data/beat_block_hammer-aloha-agilex_randomized_500-200
  
  # 处理所有 episode，图片保存到指定目录
  python batch_analyze_trajectories.py --data_dir processed_data/beat_block_hammer-aloha-agilex_randomized_500-200 --output_dir analysis_results
  
  # 只处理 episode 0-10
  python batch_analyze_trajectories.py --data_dir processed_data/beat_block_hammer-aloha-agilex_randomized_500-200 --episode_range 0-10
        """
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="processed_data 目录路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（如果为None，则在每个episode目录下保存）"
    )
    parser.add_argument(
        "--episode_range", type=str, default=None,
        help="episode范围，例如 '0-10' 或 '0,5,10'"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    try:
        process_all_episodes(data_dir, output_dir, args.episode_range)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

