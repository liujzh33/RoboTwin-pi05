#!/bin/bash

# 并行批量运行20个任务组合（10任务×2环境），每任务10次eval，lfwj_rl 权重、无扰动，6 GPU
# 使用 eval_policy_initial_motus.py，权重: .../checkpoints/pi05_rl_aloha_reward/lfwj_rl/15000
# norm、任务等与 RvS 版本相同
# 10个任务 × 2种环境（clean + random）= 20个任务
# 使用 GPU 0, 1, 2, 3, 4, 5 分别运行任务

# 设置路径
ROOT="/data2/liangxiwen/zkd/cry/RoboTwin"
RVS_SCRIPT="$ROOT/script/run_eval_rvs_lfwj.sh"
RESULT_BASE_DIR="$ROOT/eval_result"

REWARD_VALUE="${REWARD_VALUE:-1.0}"
GPU_ID=${1:-0}

# 日志文件
LOG_DIR="$ROOT/batch_eval_recovery_logs"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/gpu${GPU_ID}_*_lfwj_*.log" "$LOG_DIR/gpu${GPU_ID}_wrapper_lfwj_*.log" 2>/dev/null
MAIN_LOG="$LOG_DIR/batch_eval_rvs_6gpu_lfwj_$(date +%Y%m%d_%H%M%S).log"

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
ENV_CONFIGS=("demo_clean" "demo_randomized")

ALL_TASKS=()
for task in "${BASE_TASKS[@]}"; do
    for env_config in "${ENV_CONFIGS[@]}"; do
        ALL_TASKS+=("${task}|${env_config}")
    done
done

TEST_NUM=10

if [ "$GPU_ID" != "0" ] && [ "$GPU_ID" != "1" ] && [ "$GPU_ID" != "2" ] && [ "$GPU_ID" != "3" ] && [ "$GPU_ID" != "4" ] && [ "$GPU_ID" != "5" ]; then
    echo "错误: GPU ID 必须是 0, 1, 2, 3, 4 或 5"
    exit 1
fi

# 检查任务是否已完成（lfwj_rl 结果目录名含 lfwj_rl）
check_task_completed() {
    local task_config=$1
    local task_name=$(echo "$task_config" | cut -d'|' -f1)
    local env_config=$(echo "$task_config" | cut -d'|' -f2)
    local result_dir="$RESULT_BASE_DIR/$task_name/pi05_rvs/$env_config"
    if [ -d "$result_dir" ]; then
        local result_file=$(find "$result_dir" -path "*lfwj_rl*/*/_result.txt" -type f 2>/dev/null | head -1)
        if [ -n "$result_file" ] && [ -f "$result_file" ]; then
            return 0
        fi
    fi
    return 1
}

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

TASKS=()
TASK_INDICES=()
for i in "${!REMAINING_TASKS[@]}"; do
    if [ $((i % 6)) -eq $GPU_ID ]; then
        TASKS+=("${REMAINING_TASKS[$i]}")
        TASK_INDICES+=("$i")
    fi
done

TOTAL_TASKS=${#TASKS[@]}
SUCCESSFUL_TASKS=0
FAILED_TASKS=0
FAILED_TASK_LIST=()

echo "==========================================" | tee -a "$MAIN_LOG"
echo "GPU $GPU_ID - 批量评估开始时间: $(date)" | tee -a "$MAIN_LOG"
echo "GPU ID: $GPU_ID" | tee -a "$MAIN_LOG"
echo "分配任务数: $TOTAL_TASKS (从 $TOTAL_REMAINING 个剩余任务中分配)" | tee -a "$MAIN_LOG"
echo "每个任务测试次数: $TEST_NUM" | tee -a "$MAIN_LOG"
echo "权重: lfwj_rl/15000（无扰动，eval_policy_initial_motus.py）" | tee -a "$MAIN_LOG"
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
    
    TASK_LOG="$LOG_DIR/gpu${GPU_ID}_${TASK_NAME}_${ENV_CONFIG}_lfwj_$(date +%Y%m%d_%H%M%S).log"
    
    if bash "$RVS_SCRIPT" "$TASK_NAME" "$ENV_CONFIG" "$GPU_ID" "$TEST_NUM" "$REWARD_VALUE" > "$TASK_LOG" 2>&1; then
        echo "[GPU $GPU_ID] [$TASK_NUM/$TOTAL_TASKS] ✅ 任务完成: $TASK_NAME ($ENV_CONFIG)" | tee -a "$MAIN_LOG"
        echo "完成时间: $(date)" | tee -a "$MAIN_LOG"
        SUCCESSFUL_TASKS=$((SUCCESSFUL_TASKS + 1))
    else
        EXIT_CODE=$?
        echo "[GPU $GPU_ID] [$TASK_NUM/$TOTAL_TASKS] ❌ 任务失败: $TASK_NAME ($ENV_CONFIG) (退出码: $EXIT_CODE)" | tee -a "$MAIN_LOG"
        echo "失败时间: $(date)" | tee -a "$MAIN_LOG"
        echo "错误日志位置: $TASK_LOG" | tee -a "$MAIN_LOG"
        echo "最后50行错误日志:" | tee -a "$MAIN_LOG"
        tail -50 "$TASK_LOG" | tee -a "$MAIN_LOG"
        FAILED_TASKS=$((FAILED_TASKS + 1))
        FAILED_TASK_LIST+=("$TASK_NAME ($ENV_CONFIG)")
        echo "继续下一个任务..." | tee -a "$MAIN_LOG"
    fi
done

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
