#!/usr/bin/env python3
"""
汇总评估结果，生成Notion格式的成功率表格
"""

import os
from pathlib import Path
from datetime import datetime

# 任务列表（按照用户提供的顺序）
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
log_dir = Path("/data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs")

# 查找最新的批次时间戳
latest_timestamp = None
batch_start_time = None

for log_file in log_dir.glob("gpu*_open_gripper_*.log"):
    timestamp_str = log_file.stem.split("_open_gripper_")[-1]
    if latest_timestamp is None or timestamp_str > latest_timestamp:
        latest_timestamp = timestamp_str
        try:
            batch_start_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except:
            pass

# 收集所有任务的成功率
results = {}
for task in tasks:
    task_path = base_path / task / "pi05" / "demo_clean" / "motus"
    
    if not task_path.exists():
        results[task] = None
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
        results[task] = None
        continue
    
    # 找到最新的时间文件夹
    latest_time, latest_dir = max(time_dirs, key=lambda x: x[0])
    result_file = latest_dir / "_result.txt"
    
    # 只统计最新批次的任务
    if batch_start_time and latest_time < batch_start_time:
        results[task] = None
        continue
    
    # 检查结果文件是否存在
    if result_file.exists():
        try:
            with open(result_file, 'r') as f:
                result_content = f.read()
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
                results[task] = success_rate
        except:
            results[task] = None
    else:
        # 检查是否有视频文件（可能正在运行）
        video_files = list(latest_dir.glob("episode*.mp4"))
        if video_files:
            # 如果有很多视频但没有结果文件，可能是卡住了
            # 标记为进行中，不计算成功率
            results[task] = "in_progress"
        else:
            results[task] = None

# 生成Markdown表格格式（可以直接复制粘贴到Notion）
print("\n" + "=" * 80)
print("Markdown表格格式（可以直接复制粘贴到Notion）:")
print("=" * 80)
print()

success_rates = []
# Markdown表格格式
print("| Task Name | Success Rate |")
print("| --- | --- |")

# 数据行
for task in tasks:
    if results[task] is None:
        print(f"| {task} | N/A |")
    elif results[task] == "in_progress":
        print(f"| {task} | In Progress |")
    else:
        success_rate = results[task]  # 保持为小数格式（0.0-1.0）
        success_rates.append(success_rate)
        print(f"| {task} | {success_rate:.2f} |")

# 平均成功率
print("| Average | ", end="")
if success_rates:
    avg_rate = sum(success_rates) / len(success_rates)
    print(f"{avg_rate:.2f} |")
else:
    print("N/A |")

print("\n" + "=" * 50)
print(f"统计信息:")
print(f"  已完成任务: {len([r for r in results.values() if r is not None and r != 'in_progress'])}/{len(tasks)}")
print(f"  进行中任务: {len([r for r in results.values() if r == 'in_progress'])}/{len(tasks)}")
print(f"  未完成任务: {len([r for r in results.values() if r is None])}/{len(tasks)}")
if success_rates:
    print(f"  平均成功率: {avg_rate:.1f}%")
    print(f"  最高成功率: {max(success_rates):.1f}%")
    print(f"  最低成功率: {min(success_rates):.1f}%")
