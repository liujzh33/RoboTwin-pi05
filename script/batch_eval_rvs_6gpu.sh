#!/bin/bash

# 并行批量运行20个任务的评估脚本（6个GPU版本）
# 10个任务 × 2种环境（clean + random）= 20个任务
# 使用 GPU 0, 1, 2, 3, 4, 5 分别运行任务
# 自动跳过已完成的任务（通过检查 _result.txt 文件）
# 如果任务出错，会停止并输出错误信息，然后继续下一个任务
# 使用 RvS 权重（recovery2 openpi 架构，带 reward 输入）

# 设置路径
ROOT="/data2/liangxiwen/zkd/cry/RoboTwin"
RVS_SCRIPT="$ROOT/script/run_eval_rvs.sh"
RESULT_BASE_DIR="$ROOT/eval_result"

# 默认 reward 值（可以通过环境变量覆盖）
REWARD_VALUE="${REWARD_VALUE:-1.0}"

# 获取参数：GPU ID（先获取，用于日志清理）
GPU_ID=${1:-0}  # 默认 GPU 0

# 日志文件
LOG_DIR="$ROOT/batch_eval_recovery_logs"
mkdir -p "$LOG_DIR"
# 清除旧的日志文件（只清除当前GPU相关的日志）
rm -f "$LOG_DIR/gpu${GPU_ID}_*.log" "$LOG_DIR/gpu${GPU_ID}_wrapper_*.log" 2>/dev/null
MAIN_LOG="$LOG_DIR/batch_eval_rvs_6gpu_$(date +%Y%m%d_%H%M%S).log"

# 10个任务列表
BASE_TASKS=(
    "blocks_ranking_rgb"
    "blocks_ranking_size"
    "hanging_mug"
    "lift_pot"
    "move_stapler_pad"
    "open_laptop"
    "place_bread_skillet"
    "place_can_basket"
    "place_mouse_pad"
    "press_stapler"
)

# 环境配置：clean 和 random
ENV_CONFIGS=("demo_clean" "demo_randomized")

# 生成所有任务（任务名 + 环境配置）
ALL_TASKS=()
for task in "${BASE_TASKS[@]}"; do
    for env_config in "${ENV_CONFIGS[@]}"; do
        ALL_TASKS+=("${task}|${env_config}")
    done
done

# 测试次数
TEST_NUM=20

# 验证参数
if [ "$GPU_ID" != "0" ] && [ "$GPU_ID" != "1" ] && [ "$GPU_ID" != "2" ] && [ "$GPU_ID" != "3" ] && [ "$GPU_ID" != "4" ] && [ "$GPU_ID" != "5" ]; then
    echo "错误: GPU ID 必须是 0, 1, 2, 3, 4 或 5"
    exit 1
fi

# 检查任务是否已完成
check_task_completed() {
    local task_config=$1  # 格式: "task_name|env_config"
    local task_name=$(echo "$task_config" | cut -d'|' -f1)
    local env_config=$(echo "$task_config" | cut -d'|' -f2)
    
    # 查找该任务的 _result.txt 文件
    # 结果路径结构: eval_result/task_name/pi05_rvs/env_config/rvs_10000_reward_*/{timestamp}/_result.txt
    # 在 pi05_rvs/env_config 目录下查找所有 rvs_10000_reward_* 子目录中的 _result.txt 文件
    local result_dir="$RESULT_BASE_DIR/$task_name/pi05_rvs/$env_config"
    if [ -d "$result_dir" ]; then
        local result_file=$(find "$result_dir" -path "*/rvs_10000_reward_*/*/_result.txt" -type f 2>/dev/null | head -1)
        if [ -n "$result_file" ] && [ -f "$result_file" ]; then
            return 0  # 已完成
        fi
    fi
    return 1  # 未完成
}

# 筛选未完成的任务
REMAINING_TASKS=()
for i in "${!ALL_TASKS[@]}"; do
    task_config="${ALL_TASKS[$i]}"
    if ! check_task_completed "$task_config"; then
        REMAINING_TASKS+=("$task_config")
    else
        task_name=$(echo "$task_config" | cut -d'|' -f1)
        env_config=$(echo "$task_config" | cut -d'|' -f2)
        echo "[跳过] 任务已完成: $task_name ($env_config)" | tee -a "$MAIN_LOG"
    fi
done

TOTAL_REMAINING=${#REMAINING_TASKS[@]}
echo "" | tee -a "$MAIN_LOG"
echo "==========================================" | tee -a "$MAIN_LOG"
echo "剩余未完成任务数: $TOTAL_REMAINING / 20" | tee -a "$MAIN_LOG"
echo "==========================================" | tee -a "$MAIN_LOG"

if [ $TOTAL_REMAINING -eq 0 ]; then
    echo "所有任务已完成！" | tee -a "$MAIN_LOG"
    exit 0
fi

# 将剩余任务分配给6个GPU（确保不重复）
# GPU 0: 索引 0, 6, 12, 18, ... (i % 6 == 0)
# GPU 1: 索引 1, 7, 13, 19, ... (i % 6 == 1)
# GPU 2: 索引 2, 8, 14, ... (i % 6 == 2)
# GPU 3: 索引 3, 9, 15, ... (i % 6 == 3)
# GPU 4: 索引 4, 10, 16, ... (i % 6 == 4)
# GPU 5: 索引 5, 11, 17, ... (i % 6 == 5)
# 这样确保每个任务只被分配给一个GPU
TASKS=()
TASK_INDICES=()
for i in "${!REMAINING_TASKS[@]}"; do
    if [ $((i % 6)) -eq $GPU_ID ]; then
        TASKS+=("${REMAINING_TASKS[$i]}")
        TASK_INDICES+=("$i")
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
echo "分配任务数: $TOTAL_TASKS (从 $TOTAL_REMAINING 个剩余任务中分配)" | tee -a "$MAIN_LOG"
echo "每个任务测试次数: $TEST_NUM" | tee -a "$MAIN_LOG"
echo "Reward 值: $REWARD_VALUE" | tee -a "$MAIN_LOG"
echo "使用 RvS 权重（recovery2 openpi 架构）" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "GPU $GPU_ID 分配的任务列表 (索引 % 6 == $GPU_ID):" | tee -a "$MAIN_LOG"
for idx in "${!TASKS[@]}"; do
    original_idx="${TASK_INDICES[$idx]}"
    task_config="${TASKS[$idx]}"
    task_name=$(echo "$task_config" | cut -d'|' -f1)
    env_config=$(echo "$task_config" | cut -d'|' -f2)
    echo "  [$original_idx] $task_name ($env_config)" | tee -a "$MAIN_LOG"
done
echo "==========================================" | tee -a "$MAIN_LOG"

# 遍历每个任务
for i in "${!TASKS[@]}"; do
    TASK_CONFIG="${TASKS[$i]}"
    TASK_NAME=$(echo "$TASK_CONFIG" | cut -d'|' -f1)
    ENV_CONFIG=$(echo "$TASK_CONFIG" | cut -d'|' -f2)
    TASK_NUM=$((i + 1))
    
    echo "" | tee -a "$MAIN_LOG"
    echo "==========================================" | tee -a "$MAIN_LOG"
    echo "[GPU $GPU_ID] [$TASK_NUM/$TOTAL_TASKS] 开始任务: $TASK_NAME ($ENV_CONFIG)" | tee -a "$MAIN_LOG"
    echo "开始时间: $(date)" | tee -a "$MAIN_LOG"
    echo "==========================================" | tee -a "$MAIN_LOG"
    
    # 为每个任务创建单独的日志文件
    TASK_LOG="$LOG_DIR/gpu${GPU_ID}_${TASK_NAME}_${ENV_CONFIG}_$(date +%Y%m%d_%H%M%S).log"
    
    # 运行任务（使用 run_eval_rvs.sh 脚本）
    # 参数: [任务名] [环境] [GPU] [测试条数] [reward值]
    if bash "$RVS_SCRIPT" "$TASK_NAME" "$ENV_CONFIG" "$GPU_ID" "$TEST_NUM" "$REWARD_VALUE" > "$TASK_LOG" 2>&1; then
        
        # 任务成功完成
        echo "[GPU $GPU_ID] [$TASK_NUM/$TOTAL_TASKS] ✅ 任务完成: $TASK_NAME ($ENV_CONFIG)" | tee -a "$MAIN_LOG"
        echo "完成时间: $(date)" | tee -a "$MAIN_LOG"
        SUCCESSFUL_TASKS=$((SUCCESSFUL_TASKS + 1))
        
    else
        # 任务失败
        EXIT_CODE=$?
        echo "[GPU $GPU_ID] [$TASK_NUM/$TOTAL_TASKS] ❌ 任务失败: $TASK_NAME ($ENV_CONFIG) (退出码: $EXIT_CODE)" | tee -a "$MAIN_LOG"
        echo "失败时间: $(date)" | tee -a "$MAIN_LOG"
        echo "错误日志位置: $TASK_LOG" | tee -a "$MAIN_LOG"
        echo "最后50行错误日志:" | tee -a "$MAIN_LOG"
        tail -50 "$TASK_LOG" | tee -a "$MAIN_LOG"
        FAILED_TASKS=$((FAILED_TASKS + 1))
        FAILED_TASK_LIST+=("$TASK_NAME ($ENV_CONFIG)")
        
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
if [ $TOTAL_TASKS -gt 0 ]; then
    echo "成功率: $(echo "scale=2; $SUCCESSFUL_TASKS * 100 / $TOTAL_TASKS" | bc)%" | tee -a "$MAIN_LOG"
fi

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
