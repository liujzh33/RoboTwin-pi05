#!/usr/bin/env python
"""
Script to diversify subtask descriptions in instructions.json files.

This script adds variety to subtask descriptions, similar to how instructions
have multiple variations. It processes all episode directories and updates
the subtasks field with diverse descriptions.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Tuple


# Variants for the first subtask (grabbing the hammer)
GRAB_HAMMER_VARIANTS = [
    "Grab the hammer",
    "Pick up the hammer",
    "Take the hammer",
    "Lift the hammer",
    "Hold the hammer",
    "Grasp the hammer",
    "Seize the hammer",
    "Retrieve the hammer",
    "Acquire the hammer",
    "Obtain the hammer",
    "Collect the hammer",
    "Secure the hammer",
    "Get the hammer",
    "Fetch the hammer",
    "Pick the hammer",
    "Catch the hammer",
    "Grip the hammer",
    "Clutch the hammer",
    "Snatch the hammer",
    "Capture the hammer",
]

# Variants for the second subtask (striking with the hammer)
STRIKE_BLOCK_VARIANTS = [
    "Strike the block with the hammer",
    "Hit the block with the hammer",
    "Beat the block with the hammer",
    "Pound the block with the hammer",
    "Hammer the block",
    "Strike the block",
    "Hit the block",
    "Beat the block",
    "Pound the block",
    "Smash the block",
    "Strike the block using the hammer",
    "Hit the block using the hammer",
    "Beat the block using the hammer",
    "Pound the block using the hammer",
    "Use the hammer to strike the block",
    "Use the hammer to hit the block",
    "Use the hammer to beat the block",
    "Use the hammer to pound the block",
    "Strike the block",
    "Hit the block",
]


def diversify_subtasks(
    subtasks_list: List[List[str]], 
    seed: int = None
) -> List[List[str]]:
    """
    Diversify subtask descriptions by randomly selecting variants.
    
    Args:
        subtasks_list: Original list of subtask pairs
        seed: Random seed for reproducibility (uses episode index if None)
        
    Returns:
        Diversified list of subtask pairs
    """
    if seed is not None:
        random.seed(seed)
    
    diversified = []
    for subtasks in subtasks_list:
        if len(subtasks) >= 2:
            # Randomly select variants for each subtask
            grab_variant = random.choice(GRAB_HAMMER_VARIANTS)
            strike_variant = random.choice(STRIKE_BLOCK_VARIANTS)
            diversified.append([grab_variant, strike_variant])
        else:
            # Keep original if structure is unexpected
            diversified.append(subtasks)
    
    return diversified


def process_episode(episode_path: Path, episode_idx: int) -> bool:
    """
    Process a single episode's instructions.json file.
    
    Args:
        episode_path: Path to episode directory
        episode_idx: Episode index (for seeding)
        
    Returns:
        True if successful, False otherwise
    """
    instructions_file = episode_path / "instructions.json"
    
    if not instructions_file.exists():
        print(f"  Warning: {instructions_file} not found, skipping...")
        return False
    
    try:
        # Load existing instructions
        with open(instructions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if subtasks exist
        if 'subtasks' not in data:
            print(f"  Warning: No 'subtasks' field in {instructions_file}, skipping...")
            return False
        
        # Diversify subtasks
        original_subtasks = data['subtasks']
        diversified_subtasks = diversify_subtasks(original_subtasks, seed=episode_idx)
        
        # Update data
        data['subtasks'] = diversified_subtasks
        
        # Write back to file
        with open(instructions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"  Error processing {instructions_file}: {e}")
        return False


def main():
    """Main function to process all episodes."""
    # Dataset directory
    dataset_dir = Path("/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/beat_block_hammer-aloha-agilex_randomized_500-200")
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    # Find all episode directories
    episode_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")],
        key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0
    )
    
    print(f"Found {len(episode_dirs)} episode directories")
    print(f"Processing episodes in: {dataset_dir}")
    print()
    
    success_count = 0
    fail_count = 0
    
    for episode_dir in episode_dirs:
        # Extract episode index
        try:
            episode_idx = int(episode_dir.name.split("_")[1])
        except (ValueError, IndexError):
            episode_idx = 0
        
        print(f"Processing {episode_dir.name}...", end=" ")
        
        if process_episode(episode_dir, episode_idx):
            success_count += 1
            print("✓")
        else:
            fail_count += 1
            print("✗")
    
    print()
    print(f"Processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(episode_dirs)}")


if __name__ == "__main__":
    main()

