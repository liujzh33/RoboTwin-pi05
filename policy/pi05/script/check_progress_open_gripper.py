#!/usr/bin/env python3
"""
检查带夹爪恢复测试的批量评估任务完成情况
"""

import os
from pathlib import Path
from datetime import datetime
import time

# 任务列表（50个任务）
tasks = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]

base_path = Path("/data2/liangxiwen/zkd/cry/RoboTwin/eval_result")
completed_tasks = []
in_progress_tasks = []
not_started_tasks = []

# 查找最新的批次时间戳（从日志文件名中提取）
log_dir = Path("/data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs")
latest_timestamp = None
batch_start_time = None

for log_file in log_dir.glob("gpu*_open_gripper_*.log"):
    timestamp_str = log_file.stem.split("_open_gripper_")[-1]
    if latest_timestamp is None or timestamp_str > latest_timestamp:
        latest_timestamp = timestamp_str
        # 解析时间戳: 20260127_135708 -> 2026-01-27 13:57:08
        try:
            batch_start_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except:
            pass

print("=" * 80)
if latest_timestamp:
    print(f"📋 任务完成情况统计 (批次时间戳: {latest_timestamp})")
    if batch_start_time:
        elapsed_time = datetime.now() - batch_start_time
        hours = int(elapsed_time.total_seconds() // 3600)
        minutes = int((elapsed_time.total_seconds() % 3600) // 60)
        print(f"⏰ 批次开始时间: {batch_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  已运行时间: {hours}小时{minutes}分钟")
else:
    print("📋 任务完成情况统计")
print("=" * 80)
print()

for task in tasks:
    task_path = base_path / task / "pi05" / "demo_clean" / "motus"
    
    # 检查任务路径是否存在
    if not task_path.exists():
        # 如果批次已开始，但任务文件夹不存在，说明未开始
        if batch_start_time:
            not_started_tasks.append(task)
        continue
    
    # 找到所有时间文件夹
    time_dirs = []
    for item in task_path.iterdir():
        if item.is_dir():
            try:
                time_str = item.name
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                time_dirs.append((dt, item))
            except:
                pass
    
    if not time_dirs:
        # 如果批次已开始，但没有时间文件夹，说明未开始
        if batch_start_time:
            not_started_tasks.append(task)
        continue
    
    # 找到最新的时间文件夹
    latest_time, latest_dir = max(time_dirs, key=lambda x: x[0])
    result_file = latest_dir / "_result.txt"
    
    # 只统计最新批次的任务
    if batch_start_time and latest_time < batch_start_time:
        # 这是旧批次的任务，不算在当前批次中
        if batch_start_time:
            not_started_tasks.append(task)
        continue
    
    # 检查结果文件是否存在
    if result_file.exists():
        # 读取结果文件，获取成功率信息
        try:
            with open(result_file, 'r') as f:
                result_content = f.read()
                # 尝试提取成功率
                lines = result_content.strip().split('\n')
                success_rate = None
                for line in lines:
                    try:
                        rate = float(line.strip())
                        if 0 <= rate <= 1:
                            success_rate = rate
                            break
                    except:
                        pass
            completed_tasks.append((task, latest_time, success_rate))
        except:
            completed_tasks.append((task, latest_time, None))
    else:
        # 有文件夹但没有结果文件，可能正在运行
        # 检查是否有视频文件（说明正在运行）
        video_files = list(latest_dir.glob("episode*.mp4"))
        in_progress_tasks.append((task, latest_time, len(video_files)))

# 按时间排序
completed_tasks.sort(key=lambda x: x[1])
in_progress_tasks.sort(key=lambda x: x[1])

# 显示已完成任务
print(f"✅ 已完成任务: {len(completed_tasks)}/{len(tasks)} ({len(completed_tasks)/len(tasks)*100:.1f}%)")
print("-" * 80)
if completed_tasks:
    for task, time, success_rate in completed_tasks:
        if success_rate is not None:
            print(f"  ✓ {task:35s} ({time.strftime('%Y-%m-%d %H:%M:%S')}) - 成功率: {success_rate*100:.1f}%")
        else:
            print(f"  ✓ {task:35s} ({time.strftime('%Y-%m-%d %H:%M:%S')})")
else:
    print("  (暂无已完成的任务)")
print()

# 显示进行中任务
print(f"🔄 进行中任务: {len(in_progress_tasks)}/{len(tasks)}")
print("-" * 80)
if in_progress_tasks:
    for task, time, video_count in in_progress_tasks:
        elapsed = datetime.now() - time
        elapsed_min = int(elapsed.total_seconds() // 60)
        print(f"  ⏳ {task:35s} ({time.strftime('%Y-%m-%d %H:%M:%S')}) - 已运行{elapsed_min}分钟 - {video_count}个视频")
else:
    print("  (暂无进行中的任务)")
print()

# 显示未开始任务
print(f"⏸️  未开始任务: {len(not_started_tasks)}/{len(tasks)}")
print("-" * 80)
if not_started_tasks:
    # 每行显示5个任务
    for i in range(0, len(not_started_tasks), 5):
        tasks_line = not_started_tasks[i:i+5]
        print(f"  {' '.join(f'{t:20s}' for t in tasks_line)}")
else:
    print("  (所有任务都已开始)")
print()

# 总进度统计
total_progress = len(completed_tasks) + len(in_progress_tasks)
progress_bar_length = 50
completed_bar = int(progress_bar_length * len(completed_tasks) / len(tasks))
in_progress_bar = int(progress_bar_length * len(in_progress_tasks) / len(tasks))
not_started_bar = progress_bar_length - completed_bar - in_progress_bar

print("=" * 80)
print(f"📊 总进度统计")
print("-" * 80)
print(f"已完成: {len(completed_tasks)}/{len(tasks)} ({len(completed_tasks)/len(tasks)*100:.1f}%)")
print(f"进行中: {len(in_progress_tasks)}/{len(tasks)} ({len(in_progress_tasks)/len(tasks)*100:.1f}%)")
print(f"未开始: {len(not_started_tasks)}/{len(tasks)} ({len(not_started_tasks)/len(tasks)*100:.1f}%)")
print()
print("进度条: [" + "█" * completed_bar + "░" * in_progress_bar + " " * not_started_bar + "]")
print(f"        {'已完成':^20s}{'进行中':^20s}{'未开始':^20s}")
print("=" * 80)
