#!/bin/bash
# 批量测试不同 reward 值的 RvS 评估实验
# 用法: bash run_eval_rvs_multi_reward.sh [任务名] [环境] [GPU起始] [测试条数] [reward值列表]
# 示例: bash run_eval_rvs_multi_reward.sh blocks_ranking_rgb demo_clean 4 50 "0.0 0.5 1.0 -1.0"

set -e
ROOT="/data2/liangxiwen/zkd/cry/RoboTwin"
SCRIPT_DIR="${ROOT}/script"

TASK_NAME="${1:-blocks_ranking_rgb}"
TASK_CONFIG="${2:-demo_clean}"
GPU_START="${3:-4}"
TEST_NUM="${4:-50}"
REWARD_VALUES="${5:-0.0 0.5 1.0 -1.0}"  # 默认测试这些 reward 值

echo "=========================================="
echo "RvS 多 Reward 值批量实验"
echo "  任务: $TASK_NAME ($TASK_CONFIG)"
echo "  测试条数: $TEST_NUM"
echo "  Reward 值列表: $REWARD_VALUES"
echo "  GPU 起始: $GPU_START"
echo "=========================================="

GPU_ID=$GPU_START
for REWARD in $REWARD_VALUES; do
    # 将 reward 值中的负号和点号替换为下划线，用于目录名
    REWARD_SAFE=$(echo "$REWARD" | sed 's/-/neg/g' | sed 's/\./p/g')
    CKPT_SETTING="rvs_10000_reward_${REWARD_SAFE}"
    
    echo ""
    echo "=========================================="
    echo "启动实验: Reward = $REWARD"
    echo "  GPU: $GPU_ID"
    echo "  ckpt_setting: $CKPT_SETTING"
    echo "=========================================="
    
    # 后台运行，通过环境变量传递 CKPT_SETTING
    CKPT_SETTING="$CKPT_SETTING" nohup bash "$SCRIPT_DIR/run_eval_rvs.sh" "$TASK_NAME" "$TASK_CONFIG" "$GPU_ID" "$TEST_NUM" "$REWARD" > "${ROOT}/batch_eval_recovery_logs/rvs_reward_${REWARD_SAFE}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    PID=$!
    echo "  进程 PID: $PID"
    echo "  日志: batch_eval_recovery_logs/rvs_reward_${REWARD}_gpu${GPU_ID}_*.log"
    
    GPU_ID=$((GPU_ID + 1))
    sleep 2  # 避免同时启动太多进程
done

echo ""
echo "=========================================="
echo "所有实验已启动"
echo "=========================================="
echo "查看进程:"
ps aux | grep -E "run_eval_rvs.sh.*$TASK_NAME" | grep -v grep
echo ""
echo "查看日志:"
echo "  tail -f ${ROOT}/batch_eval_recovery_logs/rvs_reward_*.log"
echo ""
echo "停止所有实验:"
echo "  pkill -f 'run_eval_rvs.sh.*$TASK_NAME'"
