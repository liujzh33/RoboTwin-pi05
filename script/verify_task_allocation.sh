#!/bin/bash

# 验证4GPU任务分配脚本
# 确保没有重复任务，并显示每个GPU分配的任务

RESULT_BASE_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/eval_result"

# 50个任务列表
ALL_TASKS=(
    "adjust_bottle"
    "beat_block_hammer"
    "blocks_ranking_rgb"
    "blocks_ranking_size"
    "click_alarmclock"
    "click_bell"
    "dump_bin_bigbin"
    "grab_roller"
    "handover_block"
    "handover_mic"
    "hanging_mug"
    "lift_pot"
    "move_can_pot"
    "move_pillbottle_pad"
    "move_playingcard_away"
    "move_stapler_pad"
    "open_laptop"
    "open_microwave"
    "pick_diverse_bottles"
    "pick_dual_bottles"
    "place_a2b_left"
    "place_a2b_right"
    "place_bread_basket"
    "place_bread_skillet"
    "place_burger_fries"
    "place_can_basket"
    "place_cans_plasticbox"
    "place_container_plate"
    "place_dual_shoes"
    "place_empty_cup"
    "place_fan"
    "place_mouse_pad"
    "place_object_basket"
    "place_object_scale"
    "place_object_stand"
    "place_phone_stand"
    "place_shoe"
    "press_stapler"
    "put_bottles_dustbin"
    "put_object_cabinet"
    "rotate_qrcode"
    "scan_object"
    "shake_bottle_horizontally"
    "shake_bottle"
    "stack_blocks_three"
    "stack_blocks_two"
    "stack_bowls_three"
    "stack_bowls_two"
    "stamp_seal"
    "turn_switch"
)

# 检查任务是否已完成
check_task_completed() {
    local task_name=$1
    local result_file=$(find "$RESULT_BASE_DIR/$task_name/pi05/demo_clean/motus" -name "_result.txt" -type f 2>/dev/null | head -1)
    if [ -n "$result_file" ] && [ -f "$result_file" ]; then
        return 0  # 已完成
    else
        return 1  # 未完成
    fi
}

# 筛选未完成的任务
REMAINING_TASKS=()
REMAINING_INDICES=()
for i in "${!ALL_TASKS[@]}"; do
    task_name="${ALL_TASKS[$i]}"
    if ! check_task_completed "$task_name"; then
        REMAINING_TASKS+=("$task_name")
        REMAINING_INDICES+=("$i")
    fi
done

echo "=========================================="
echo "任务分配验证"
echo "=========================================="
echo "总任务数: ${#ALL_TASKS[@]}"
echo "已完成任务数: $((${#ALL_TASKS[@]} - ${#REMAINING_TASKS[@]}))"
echo "剩余任务数: ${#REMAINING_TASKS[@]}"
echo ""

# 为每个GPU分配任务
for GPU_ID in 0 1 2 3; do
    GPU_TASKS=()
    GPU_ORIGINAL_INDICES=()
    
    for idx in "${!REMAINING_TASKS[@]}"; do
        if [ $((idx % 4)) -eq $GPU_ID ]; then
            GPU_TASKS+=("${REMAINING_TASKS[$idx]}")
            GPU_ORIGINAL_INDICES+=("${REMAINING_INDICES[$idx]}")
        fi
    done
    
    echo "GPU $GPU_ID: ${#GPU_TASKS[@]} 个任务"
    for i in "${!GPU_TASKS[@]}"; do
        orig_idx="${GPU_ORIGINAL_INDICES[$i]}"
        echo "  [原始索引 $orig_idx] ${GPU_TASKS[$i]}"
    done
    echo ""
done

# 验证是否有重复
echo "=========================================="
echo "重复检查"
echo "=========================================="
ALL_ALLOCATED_TASKS=()
for GPU_ID in 0 1 2 3; do
    for idx in "${!REMAINING_TASKS[@]}"; do
        if [ $((idx % 4)) -eq $GPU_ID ]; then
            ALL_ALLOCATED_TASKS+=("${REMAINING_TASKS[$idx]}")
        fi
    done
done

# 检查重复
DUPLICATES=$(printf '%s\n' "${ALL_ALLOCATED_TASKS[@]}" | sort | uniq -d)
if [ -z "$DUPLICATES" ]; then
    echo "✅ 无重复任务 - 所有任务正确分配"
else
    echo "❌ 发现重复任务:"
    echo "$DUPLICATES"
    exit 1
fi

# 检查是否所有剩余任务都被分配
if [ ${#ALL_ALLOCATED_TASKS[@]} -eq ${#REMAINING_TASKS[@]} ]; then
    echo "✅ 所有剩余任务都已分配"
else
    echo "❌ 任务分配不完整:"
    echo "  剩余任务数: ${#REMAINING_TASKS[@]}"
    echo "  已分配任务数: ${#ALL_ALLOCATED_TASKS[@]}"
    exit 1
fi

echo ""
echo "=========================================="
echo "权重路径验证"
echo "=========================================="
MOTUS_CHECKPOINT_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/policy/pi05/checkpoint_torch/motus"
echo "权重路径: $MOTUS_CHECKPOINT_DIR"
if [ -d "$MOTUS_CHECKPOINT_DIR" ]; then
    echo "✅ 权重目录存在"
    echo "   文件数: $(find "$MOTUS_CHECKPOINT_DIR" -type f | wc -l)"
    echo "   总大小: $(du -sh "$MOTUS_CHECKPOINT_DIR" | cut -f1)"
else
    echo "❌ 权重目录不存在"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 验证通过 - 可以安全启动4GPU评估"
echo "=========================================="

