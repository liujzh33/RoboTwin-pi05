#!/usr/bin/env python3
"""
快速测试脚本：检测单个 episode 的夹爪状态
用于验证脚本是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.detect_gripper_phases import process_episode

if __name__ == "__main__":
    # 测试路径
    test_hdf5 = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/beat_block_hammer-aloha-agilex_randomized_500-200/episode_0/episode_0.hdf5"
    
    if os.path.exists(test_hdf5):
        print("Testing gripper phase detection...")
        result = process_episode(test_hdf5, threshold=0.5, visualize=False)
        print("\nTest completed successfully!")
        print(f"Phase 1 (Grasp): {result['phase1_steps']} steps")
        print(f"Phase 2 (Strike): {result['phase2_steps']} steps")
    else:
        print(f"Test file not found: {test_hdf5}")
        print("Please check the path and try again.")

