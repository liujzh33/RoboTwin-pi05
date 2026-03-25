#!/usr/bin/env python3
"""
Batch analysis of success video frame counts per task.
Usage: python script/batch_analyze_success_videos.py
"""

import os
import subprocess
import sys
import json
from pathlib import Path

# List of tasks: eval_dir (set yourpath to your eval result root) and task_name
TASKS = [
    {"eval_dir": "yourpath/eval_result/adjust_bottle/pi05/demo_clean/motus/2026-01-26 00:22:47", "task_name": "adjust_bottle"},
    {"eval_dir": "yourpath/eval_result/beat_block_hammer/pi05/demo_clean/motus/2026-01-26-00:48:06", "task_name": "beat_block_hammer"},
    {"eval_dir": "yourpath/eval_result/blocks_ranking_rgb/pi05/demo_clean/motus/2026-01-26 02:16:53", "task_name": "blocks_ranking_rgb"},
    {"eval_dir": "yourpath/eval_result/blocks_ranking_size/pi05/demo_clean/motus/2026-01-26 02:16:53", "task_name": "blocks_ranking_size"},
    {"eval_dir": "yourpath/eval_result/click_alarmclock/pi05/demo_clean/motus/2026-01-26 02:16:52", "task_name": "click_alarmclock"},
    {"eval_dir": "yourpath/eval_result/click_bell/pi05/demo_clean/motus/2026-01-26 02:16:53", "task_name": "click_bell"},
    {"eval_dir": "yourpath/eval_result/dump_bin_bigbin/pi05/demo_clean/motus/2026-01-26 03:45:17", "task_name": "dump_bin_bigbin"},
    {"eval_dir": "yourpath/eval_result/grab_roller/pi05/demo_clean/motus/2026-01-26 03:56:22", "task_name": "grab_roller"},
    {"eval_dir": "yourpath/eval_result/handover_block/pi05/demo_clean/motus/2026-01-26 02:37:55", "task_name": "handover_block"},
    {"eval_dir": "yourpath/eval_result/handover_mic/pi05/demo_clean/motus/2026-01-26 02:40:25", "task_name": "handover_mic"},
    {"eval_dir": "yourpath/eval_result/hanging_mug/pi05/demo_clean/motus/2026-01-26 04:23:07", "task_name": "hanging_mug"},
    {"eval_dir": "yourpath/eval_result/lift_pot/pi05/demo_clean/motus/2026-01-26 04:13:28", "task_name": "lift_pot"},
    {"eval_dir": "yourpath/eval_result/move_can_pot/pi05/demo_clean/motus/2026-01-26 03:46:33", "task_name": "move_can_pot"},
    {"eval_dir": "yourpath/eval_result/move_pillbottle_pad/pi05/demo_clean/motus/2026-01-26 03:17:35", "task_name": "move_pillbottle_pad"},
    {"eval_dir": "yourpath/eval_result/move_playingcard_away/pi05/demo_clean/motus/2026-01-26 05:47:42", "task_name": "move_playingcard_away"},
    {"eval_dir": "yourpath/eval_result/move_stapler_pad/pi05/demo_clean/motus/2026-01-26 04:53:50", "task_name": "move_stapler_pad"},
    {"eval_dir": "yourpath/eval_result/open_laptop/pi05/demo_clean/motus/2026-01-26 04:28:23", "task_name": "open_laptop"},
    {"eval_dir": "yourpath/eval_result/open_microwave/pi05/demo_clean/motus/2026-01-26 03:57:19", "task_name": "open_microwave"},
    {"eval_dir": "yourpath/eval_result/pick_diverse_bottles/pi05/demo_clean/motus/2026-01-26 06:09:08", "task_name": "pick_diverse_bottles"},
    {"eval_dir": "yourpath/eval_result/pick_dual_bottles/pi05/demo_clean/motus/2026-01-26 05:29:07", "task_name": "pick_dual_bottles"},
    {"eval_dir": "yourpath/eval_result/place_a2b_left/pi05/demo_clean/motus/2026-01-26 05:08:47", "task_name": "place_a2b_left"},
    {"eval_dir": "yourpath/eval_result/place_a2b_right/pi05/demo_clean/motus/2026-01-26 05:22:28", "task_name": "place_a2b_right"},
    {"eval_dir": "yourpath/eval_result/place_bread_basket/pi05/demo_clean/motus/2026-01-26 06:48:42", "task_name": "place_bread_basket"},
    {"eval_dir": "yourpath/eval_result/place_bread_skillet/pi05/demo_clean/motus/2026-01-26 06:04:40", "task_name": "place_bread_skillet"},
    {"eval_dir": "yourpath/eval_result/place_burger_fries/pi05/demo_clean/motus/2026-01-26 05:35:27", "task_name": "place_burger_fries"},
    {"eval_dir": "yourpath/eval_result/place_can_basket/pi05/demo_clean/motus/2026-01-26 00:22:47", "task_name": "place_can_basket"},
    {"eval_dir": "yourpath/eval_result/place_cans_plasticbox/pi05/demo_clean/motus/2026-01-26 05:52:26", "task_name": "place_cans_plasticbox"},
    {"eval_dir": "yourpath/eval_result/place_container_plate/pi05/demo_clean/motus/2026-01-26 07:36:02", "task_name": "place_container_plate"},
    {"eval_dir": "yourpath/eval_result/place_dual_shoes/pi05/demo_clean/motus/2026-01-26 06:53:39", "task_name": "place_dual_shoes"},
    {"eval_dir": "yourpath/eval_result/place_empty_cup/pi05/demo_clean/motus/2026-01-26 06:20:13", "task_name": "place_empty_cup"},
    {"eval_dir": "yourpath/eval_result/place_fan/pi05/demo_clean/motus/2026-01-26 07:06:24", "task_name": "place_fan"},
    {"eval_dir": "yourpath/eval_result/place_mouse_pad/pi05/demo_clean/motus/2026-01-26 08:02:36", "task_name": "place_mouse_pad"},
    {"eval_dir": "yourpath/eval_result/place_object_basket/pi05/demo_clean/motus/2026-01-26 07:54:12", "task_name": "place_object_basket"},
    {"eval_dir": "yourpath/eval_result/place_object_scale/pi05/demo_clean/motus/2026-01-26 06:52:19", "task_name": "place_object_scale"},
    {"eval_dir": "yourpath/eval_result/place_object_stand/pi05/demo_clean/motus/2026-01-26 07:37:12", "task_name": "place_object_stand"},
    {"eval_dir": "yourpath/eval_result/place_phone_stand/pi05/demo_clean/motus/2026-01-26 08:34:11", "task_name": "place_phone_stand"},
    {"eval_dir": "yourpath/eval_result/place_shoe/pi05/demo_clean/motus/2026-01-26 08:53:58", "task_name": "place_shoe"},
    {"eval_dir": "yourpath/eval_result/press_stapler/pi05/demo_clean/motus/2026-01-26 07:22:51", "task_name": "press_stapler"},
    {"eval_dir": "yourpath/eval_result/put_bottles_dustbin/pi05/demo_clean/motus/2026-01-26 07:59:54", "task_name": "put_bottles_dustbin"},
    {"eval_dir": "yourpath/eval_result/put_object_cabinet/pi05/demo_clean/motus/2026-01-26 09:02:46", "task_name": "put_object_cabinet"},
    {"eval_dir": "yourpath/eval_result/rotate_qrcode/pi05/demo_clean/motus/2026-01-26 09:28:45", "task_name": "rotate_qrcode"},
    {"eval_dir": "yourpath/eval_result/scan_object/pi05/demo_clean/motus/2026-01-26 07:46:05", "task_name": "scan_object"},
    {"eval_dir": "yourpath/eval_result/shake_bottle/pi05/demo_clean/motus/2026-01-26 10:36:14", "task_name": "shake_bottle"},
    {"eval_dir": "yourpath/eval_result/shake_bottle_horizontally/pi05/demo_clean/motus/2026-01-26 10:08:52", "task_name": "shake_bottle_horizontally"},
    {"eval_dir": "yourpath/eval_result/stack_blocks_three/pi05/demo_clean/motus/2026-01-26 09:59:49", "task_name": "stack_blocks_three"},
    {"eval_dir": "yourpath/eval_result/stack_blocks_two/pi05/demo_clean/motus/2026-01-26 08:43:00", "task_name": "stack_blocks_two"},
    {"eval_dir": "yourpath/eval_result/stack_bowls_three/pi05/demo_clean/motus/2026-01-26 10:29:18", "task_name": "stack_bowls_three"},
    {"eval_dir": "yourpath/eval_result/stack_bowls_two/pi05/demo_clean/motus/2026-01-26 10:58:18", "task_name": "stack_bowls_two"},
    {"eval_dir": "yourpath/eval_result/stamp_seal/pi05/demo_clean/motus/2026-01-26 11:33:05", "task_name": "stamp_seal"},
    {"eval_dir": "yourpath/eval_result/turn_switch/pi05/demo_clean/motus/2026-01-26 09:40:35", "task_name": "turn_switch"},
]


def analyze_task(eval_dir, task_name, task_lim_dir=None):
    """Run frame-count analysis for one task."""
    print(f"\n{'='*80}")
    print(f"Analyzing task: {task_name}")
    print(f"Eval dir: {eval_dir}")
    print(f"{'='*80}")

    if not os.path.exists(eval_dir):
        print(f"\033[91mError: Directory not found: {eval_dir}\033[0m")
        return False

    video_dir = os.path.join(eval_dir, "video")
    if not os.path.exists(video_dir):
        video_dir = eval_dir

    import glob
    video_files = glob.glob(os.path.join(video_dir, "episode*.mp4"))
    if len(video_files) == 0:
        print(f"\033[91mError: No video files found in {video_dir}\033[0m")
        return False

    script_path = os.path.join(os.path.dirname(__file__), "analyze_success_videos.py")
    cmd = [sys.executable, script_path, "--eval_dir", eval_dir, "--task_name", task_name]
    if task_lim_dir:
        cmd.extend(["--task_lim_dir", task_lim_dir])

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"\033[92m✓ Successfully analyzed {task_name}\033[0m")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\033[91m✗ Failed to analyze {task_name}: {e}\033[0m")
        return False
    except Exception as e:
        print(f"\033[91m✗ Error analyzing {task_name}: {e}\033[0m")
        return False


def main():
    print("\n" + "="*80)
    print("Batch Analysis of Success Videos")
    print("="*80)

    task_lim_dir = "/data2/liangxiwen/zkd/cry/RoboTwin/task_lim"
    os.makedirs(task_lim_dir, exist_ok=True)
    print(f"\nTask threshold directory: {task_lim_dir}")

    results = []
    for task in TASKS:
        success = analyze_task(task["eval_dir"], task["task_name"], task_lim_dir=task_lim_dir)
        results.append({"task_name": task["task_name"], "eval_dir": task["eval_dir"], "success": success})

    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    print(f"\n\033[92mSuccessful: {len(successful)}/{len(results)}\033[0m")
    for r in successful:
        print(f"  ✓ {r['task_name']}")
    if failed:
        print(f"\n\033[91mFailed: {len(failed)}/{len(results)}\033[0m")
        for r in failed:
            print(f"  ✗ {r['task_name']}: {r['eval_dir']}")

    print("\n" + "="*80)
    print("Generated Statistics Files")
    print("="*80)
    print("\n1. Per-eval stats: success_video_stats.json in each eval dir")
    for task in TASKS:
        stats_file = os.path.join(task["eval_dir"], "success_video_stats.json")
        status = "\033[92m✓\033[0m" if os.path.exists(stats_file) else "\033[93m⚠ not found\033[0m"
        print(f"  {task['task_name']}: {stats_file} {status}")

    print(f"\n2. Unified threshold files in {task_lim_dir}:")
    for task in TASKS:
        threshold_file = os.path.join(task_lim_dir, f"{task['task_name']}.json")
        if os.path.exists(threshold_file):
            try:
                with open(threshold_file, 'r') as f:
                    data = json.load(f)
                    thresholds = data.get('thresholds', {})
                    print(f"  \033[92m✓ {task['task_name']}\033[0m Max: {thresholds.get('max', 'N/A')}, "
                          f"P99: {thresholds.get('p99', 'N/A')}, P95: {thresholds.get('p95', 'N/A')}, "
                          f"P90: {thresholds.get('p90', 'N/A')}, Mean: {thresholds.get('mean', 'N/A')}")
            except Exception as e:
                print(f"  \033[93m⚠ {task['task_name']}: read error {e}\033[0m")
        else:
            print(f"  \033[93m⚠ {task['task_name']}: {threshold_file} not found\033[0m")

    print("\n" + "="*80)
    print("Analysis complete. Thresholds saved in:", task_lim_dir)
    print("="*80)


if __name__ == "__main__":
    main()
