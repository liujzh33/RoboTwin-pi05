#!/usr/bin/env python3
"""Test single data sample through transform pipeline"""
import sys
sys.path.insert(0, '/mnt/data1/liujingzhi/RoboTwin/policy/pi05/src')

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.transforms import RepackTransform, LoadSubtaskFromInstructions, flatten_dict
import json

# Load single sample
print("=" * 80)
print("Step 1: Load raw sample from LeRobot dataset")
print("=" * 80)
dataset = LeRobotDataset('blocks_ranking_rgb_pi05_200')
raw_sample = dataset[100]  # Get sample 100

print(f"\nRaw sample keys: {list(raw_sample.keys())}")
print(f"\nChecking specific fields in raw sample:")
for key in ['instructions', 'subtasks', 'frame_idx', 'phase_info', 'observation.images.cam_high']:
    if key in raw_sample:
        val = raw_sample[key]
        print(f"  ✓ {key}: {type(val).__name__}")
        if key in ['instructions', 'subtasks']:
            print(f"    Value: {val}")
    else:
        print(f"  ✗ {key}: NOT FOUND")

# Test flatten
print("\n" + "=" * 80)
print("Step 2: Flatten the sample")
print("=" * 80)
flat = flatten_dict(raw_sample)
print(f"\nFlattened keys (showing first 20):")
for i, key in enumerate(list(flat.keys())[:20]):
    print(f"  {i+1}. {key}")

# Check if our fields are in flattened dict
print(f"\nChecking required fields in flattened dict:")
for key in ['instructions', 'subtasks', 'frame_idx', 'phase_info']:
    exists = key in flat
    print(f"  {key}: {'✓ FOUND' if exists else '✗ NOT FOUND'}")
    if exists:
        val = flat[key]
        if isinstance(val, (str, list)):
            print(f"    Value: {val}")

# Test RepackTransform with correct structure
print("\n" + "=" * 80)
print("Step 3: Apply RepackTransform")
print("=" * 80)

repack_structure = {
    "images": {
        "cam_high": "observation/images/cam_high",
        "cam_left_wrist": "observation/images/cam_left_wrist",
        "cam_right_wrist": "observation/images/cam_right_wrist",
    },
    "state": "observation/state",
    "actions": "action",
    "instructions": "instructions",
    "subtasks": "subtasks",
    "frame_idx": "frame_idx",
    "phase_info": "phase_info",
}

print(f"Repack structure: {json.dumps(repack_structure, indent=2)}")

repack = RepackTransform(repack_structure)

try:
    repacked = repack(raw_sample)
    print(f"\n✓ SUCCESS! Repacked data keys: {list(repacked.keys())}")
    
    print(f"\nChecking fields after repack:")
    for key in ['instructions', 'subtasks', 'frame_idx', 'phase_info']:
        if key in repacked:
            val = repacked[key]
            print(f"  ✓ {key}: {type(val).__name__}")
            if key in ['instructions', 'subtasks']:
                print(f"    Value: {val}")
        else:
            print(f"  ✗ {key}: MISSING after repack!")
    
    # Test LoadSubtaskFromInstructions
    print("\n" + "=" * 80)
    print("Step 4: Apply LoadSubtaskFromInstructions")
    print("=" * 80)
    
    load_subtask = LoadSubtaskFromInstructions(use_first_instruction=True)
    try:
        result = load_subtask(repacked)
        print(f"✓ SUCCESS!")
        print(f"Result keys: {list(result.keys())}")
        if 'high_prompt' in result:
            print(f"  high_prompt: {result['high_prompt']}")
        if 'low_prompt' in result:
            print(f"  low_prompt: {result['low_prompt']}")
    except Exception as e:
        print(f"✗ ERROR in LoadSubtaskFromInstructions: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"\n✗ ERROR in RepackTransform: {e}")
    import traceback
    traceback.print_exc()
    
    # Let's check what keys are actually available in flat dict
    print(f"\n\nDEBUG: All flattened keys:")
    for key in sorted(flat.keys()):
        print(f"  {key}")
