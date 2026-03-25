#!/bin/bash

# 并行批量运行50个任务的评估脚本
# 使用 GPU 0 和 GPU 1 分别运行25个任务
# 如果任务出错，会停止并输出错误信息，然后继续下一个任务

# 设置权重路径
MOTUS_CHECKPOINT_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/policy/pi05/checkpoint_torch/motus"
SCRIPT_PATH="/data2/liangxiwen/zkd/cry/RoboTwin/script/eval_policy_initial_motus.py"
CONFIG_PATH="/data2/liangxiwen/zkd/cry/RoboTwin/policy/pi05/deploy_policy.yml"

# 日志文件
LOG_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/batch_eval_parallel_$(date +%Y%m%d_%H%M%S).log"

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

# 测试次数
TEST_NUM=50

# 获取参数：GPU ID 和任务范围
GPU_ID=${1:-0}  # 默认 GPU 0
START_IDX=${2:-0}  # 默认从第0个任务开始
END_IDX=${3:-24}  # 默认到第24个任务（25个任务）

# 验证参数
if [ "$GPU_ID" != "0" ] && [ "$GPU_ID" != "1" ]; then
    echo "错误: GPU ID 必须是 0 或 1"
    exit 1
fi

# 提取指定范围的任务
TASKS=()
for i in $(seq $START_IDX $END_IDX); do
    if [ $i -lt ${#ALL_TASKS[@]} ]; then
        TASKS+=("${ALL_TASKS[$i]}")
    fi
done

# 统计信息
TOTAL_TASKS=${#TASKS[@]}
SUCCESSFUL_TASKS=0
FAILED_TASKS=0
FAILED_TASK_LIST=()

echo "==========================================" | tee -a "$MAIN_LOG"
echo "GPU $GPU_ID - 批量评估开始时间: $(date)" | tee -a "$MAIN_LOG"
echo "GPU ID: $GPU_ID" | tee -a "$MAIN_LOG"
echo "任务范围: [$START_IDX-$END_IDX] (共 $TOTAL_TASKS 个任务)" | tee -a "$MAIN_LOG"
echo "每个任务测试次数: $TEST_NUM" | tee -a "$MAIN_LOG"
echo "权重路径: $MOTUS_CHECKPOINT_DIR" | tee -a "$MAIN_LOG"
echo "==========================================" | tee -a "$MAIN_LOG"

# 遍历每个任务
for i in "${!TASKS[@]}"; do
    TASK_NAME="${TASKS[$i]}"
    TASK_NUM=$((i + 1))
    GLOBAL_TASK_NUM=$((START_IDX + i + 1))
    
    echo "" | tee -a "$MAIN_LOG"
    echo "==========================================" | tee -a "$MAIN_LOG"
    echo "[GPU $GPU_ID] [$TASK_NUM/$TOTAL_TASKS] [全局: $GLOBAL_TASK_NUM/50] 开始任务: $TASK_NAME" | tee -a "$MAIN_LOG"
    echo "开始时间: $(date)" | tee -a "$MAIN_LOG"
    echo "==========================================" | tee -a "$MAIN_LOG"
    
    # 为每个任务创建单独的日志文件
    TASK_LOG="$LOG_DIR/gpu${GPU_ID}_${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
    
    # 运行任务（直接运行并捕获错误）
    # 注意：如果任务出错，会停止当前任务并继续下一个
    if CUDA_VISIBLE_DEVICES=$GPU_ID python "$SCRIPT_PATH" \
        --config "$CONFIG_PATH" \
        --overrides \
        --task_name "$TASK_NAME" \
        --task_config demo_clean \
        --train_config_name pi05_aloha_full_base_all \
        --model_name pi05_ft_4a100 \
        --ckpt_setting motus \
        --seed 0 \
        --policy_name pi05 \
        --use_torch_checkpoint True \
        --torch_checkpoint_dir "$MOTUS_CHECKPOINT_DIR" \
        --test_num "$TEST_NUM" \
        > "$TASK_LOG" 2>&1; then
        
        # 任务成功完成
        echo "[GPU $GPU_ID] [$TASK_NUM/$TOTAL_TASKS] ✅ 任务完成: $TASK_NAME" | tee -a "$MAIN_LOG"
        echo "完成时间: $(date)" | tee -a "$MAIN_LOG"
        SUCCESSFUL_TASKS=$((SUCCESSFUL_TASKS + 1))
        
    else
        # 任务失败
        EXIT_CODE=$?
        echo "[GPU $GPU_ID] [$TASK_NUM/$TOTAL_TASKS] ❌ 任务失败: $TASK_NAME (退出码: $EXIT_CODE)" | tee -a "$MAIN_LOG"
        echo "失败时间: $(date)" | tee -a "$MAIN_LOG"
        echo "错误日志位置: $TASK_LOG" | tee -a "$MAIN_LOG"
        echo "最后50行错误日志:" | tee -a "$MAIN_LOG"
        tail -50 "$TASK_LOG" | tee -a "$MAIN_LOG"
        FAILED_TASKS=$((FAILED_TASKS + 1))
        FAILED_TASK_LIST+=("$TASK_NAME")
        
        # 继续下一个任务（不退出）
        echo "继续下一个任务..." | tee -a "$MAIN_LOG"
    fi
done

# 输出总结
echo "" | tee -a "$MAIN_LOG"
echo "==========================================" | tee -a "$MAIN_LOG"
echo "GPU $GPU_ID - 批量评估完成时间: $(date)" | tee -a "$MAIN_LOG"
echo "==========================================" | tee -a "$MAIN_LOG"
echo "总任务数: $TOTAL_TASKS" | tee -a "$MAIN_LOG"
echo "成功任务数: $SUCCESSFUL_TASKS" | tee -a "$MAIN_LOG"
echo "失败任务数: $FAILED_TASKS" | tee -a "$MAIN_LOG"
echo "成功率: $(echo "scale=2; $SUCCESSFUL_TASKS * 100 / $TOTAL_TASKS" | bc)%" | tee -a "$MAIN_LOG"

if [ ${#FAILED_TASK_LIST[@]} -gt 0 ]; then
    echo "" | tee -a "$MAIN_LOG"
    echo "失败的任务列表:" | tee -a "$MAIN_LOG"
    for failed_task in "${FAILED_TASK_LIST[@]}"; do
        echo "  - $failed_task" | tee -a "$MAIN_LOG"
    done
fi

echo "==========================================" | tee -a "$MAIN_LOG"
echo "GPU $GPU_ID 主日志文件: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "任务日志目录: $LOG_DIR" | tee -a "$MAIN_LOG"

