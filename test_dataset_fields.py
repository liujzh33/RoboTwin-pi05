#!/usr/bin/env python3
"""Test dataset field access"""
import sys
sys.path.insert(0, '/mnt/data1/liujingzhi/RoboTwin/policy/pi05/src')

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.transforms import RepackTransform, flatten_dict

# Load dataset
dataset = LeRobotDataset('blocks_ranking_rgb_pi05_200')
sample = dataset[0]

print("=" * 80)
print("Raw sample keys:", list(sample.keys())[:20])
print("=" * 80)

# Test flatten
flat = flatten_dict(sample)
print("\nFlattened keys (first 30):")
for i, key in enumerate(list(flat.keys())[:30]):
    print(f"  {i+1}. {key}")
print("=" * 80)

# Check specific fields
print("\nChecking specific fields:")
for field in ['instructions', 'subtasks', 'frame_idx', 'phase_info']:
    print(f"  {field}: {field in flat}")
    if field in flat:
        val = flat[field]
        print(f"    Type: {type(val)}, Value: {val if len(str(val)) < 100 else str(val)[:100] + '...'}")

print("=" * 80)

# Test RepackTransform
repack = RepackTransform({
    "images": {
        "cam_high": "observation.images.cam_high",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    },
    "state": "observation.state",
    "actions": "action",
    "instructions": "instructions",
    "subtasks": "subtasks",
    "frame_idx": "frame_idx",
    "phase_info": "phase_info",
})

print("\nTrying RepackTransform...")
try:
    repacked = repack(sample)
    print("SUCCESS! Repacked keys:", list(repacked.keys()))
    for key in ['instructions', 'subtasks', 'frame_idx', 'phase_info']:
        if key in repacked:
            val = repacked[key]
            print(f"  {key}: {type(val)}, {val if len(str(val)) < 100 else str(val)[:100] + '...'}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
