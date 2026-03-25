#!/bin/bash

# 先清除 GPU 0-5 上所有评估相关主进程，再启动 6 GPU 批量（lfwj_rl 权重 + open_gripper 扰动）
# 10个任务 × 2种环境 = 20个任务组合，每任务 10 次 eval

SCRIPT_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/script"
BATCH_SCRIPT="$SCRIPT_DIR/batch_eval_rvs_6gpu_lfwj_open_gripper.sh"
LOG_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_recovery_logs"

REWARD_VALUE="${REWARD_VALUE:-1.0}"

echo "=========================================="
echo "步骤 1: 清除 GPU 0-5 上所有评估相关进程"
echo "=========================================="
# 只杀子进程，不杀当前脚本（run_batch_* 会被 batch_eval_rvs/run_eval_rvs 误匹配）
pkill -f "eval_policy_initial_motus.py" 2>/dev/null && echo "  已结束 Python 评估进程" || true
pkill -f "run_eval_rvs_lfwj_open_gripper.sh" 2>/dev/null && echo "  已结束 run_eval_rvs_lfwj_open_gripper 进程" || true
pkill -f "run_eval_rvs_open_gripper.sh" 2>/dev/null && echo "  已结束 run_eval_rvs_open_gripper 进程" || true
pkill -f "run_eval_rvs_lfwj.sh" 2>/dev/null && echo "  已结束 run_eval_rvs_lfwj 进程" || true
pkill -f "run_eval_rvs.sh" 2>/dev/null && echo "  已结束 run_eval_rvs 进程" || true
# 只杀 batch_xxx.sh 0/1/2/3/4/5（带数字的 worker），不杀 run_batch_xxx.sh
for g in 0 1 2 3 4 5; do
  pkill -f "batch_eval_rvs_6gpu_lfwj_open_gripper.sh $g" 2>/dev/null && echo "  已结束 batch worker GPU $g" || true
done
pkill -f "batch_eval_rvs_6gpu_open_gripper.sh [0-5]" 2>/dev/null && echo "  已结束 batch_eval_rvs_6gpu_open_gripper 进程" || true
pkill -f "batch_eval_rvs_6gpu_lfwj.sh [0-5]" 2>/dev/null && echo "  已结束 batch_eval_rvs_6gpu_lfwj 进程" || true
pkill -f "batch_eval_rvs_6gpu.sh [0-5]" 2>/dev/null && echo "  已结束 batch_eval_rvs_6gpu 进程" || true
sleep 2
echo "检查残留进程..."
if ps aux | grep -E "eval_policy_initial_motus\.py|run_eval_rvs.*\.sh|batch_eval_rvs.*\.sh" | grep -v grep | grep -v "run_batch_eval"; then
    echo "  仍有残留，可手动执行下面命令结束:"
    echo "  pkill -f eval_policy_initial_motus.py; pkill -f run_eval_rvs; pkill -f 'batch_eval_rvs.*\.sh'"
else
    echo "  GPU 0-5 评估进程已清除"
fi
echo ""

echo "=========================================="
echo "步骤 2: 清除旧的 lfwj_open_gripper 日志"
echo "=========================================="
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/gpu*_wrapper_lfwj_open_gripper_*.log "$LOG_DIR"/batch_eval_rvs_6gpu_lfwj_open_gripper_*.log 2>/dev/null
rm -f "$LOG_DIR"/gpu*_*_lfwj_open_gripper_*.log 2>/dev/null
echo "lfwj_open_gripper 日志已清除"
echo ""

echo "=========================================="
echo "步骤 3: 启动 6 GPU 并行批量（lfwj_rl + open_gripper）"
echo "=========================================="
echo "GPU 0, 1, 2, 3, 4, 5: 自动分配剩余未完成任务"
echo "任务配置: 10个任务 × 2种环境（clean + random）= 20个任务"
echo "每个任务测试 10 次"
echo "评估脚本: eval_policy_initial_motus_open_gripper.py (--open_gripper)"
echo "权重路径: /data2/liangxiwen/zkd/cry/RoboTwin/policy/recovery2/openpi-main/checkpoints/pi05_rl_aloha_reward/lfwj_rl/15000"
echo "norm_stats: /data2/liangxiwen/zkd/cry/RoboTwin/policy/recovery2/openpi-main/norm_stats.json"
echo "自动跳过已完成的任务"
echo ""

echo "启动 GPU 0 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 0 > "$LOG_DIR/gpu0_wrapper_lfwj_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU0=$!

echo "启动 GPU 1 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 1 > "$LOG_DIR/gpu1_wrapper_lfwj_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU1=$!

echo "启动 GPU 2 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 2 > "$LOG_DIR/gpu2_wrapper_lfwj_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU2=$!

echo "启动 GPU 3 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 3 > "$LOG_DIR/gpu3_wrapper_lfwj_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU3=$!

echo "启动 GPU 4 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 4 > "$LOG_DIR/gpu4_wrapper_lfwj_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU4=$!

echo "启动 GPU 5 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 5 > "$LOG_DIR/gpu5_wrapper_lfwj_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU5=$!

echo ""
echo "✅ 6 GPU 并行批量已启动（lfwj_rl 权重 + open_gripper 扰动）"
echo "=========================================="
echo "GPU 0 进程 PID: $PID_GPU0"
echo "GPU 1 进程 PID: $PID_GPU1"
echo "GPU 2 进程 PID: $PID_GPU2"
echo "GPU 3 进程 PID: $PID_GPU3"
echo "GPU 4 进程 PID: $PID_GPU4"
echo "GPU 5 进程 PID: $PID_GPU5"
echo ""
echo "查看实时日志:"
echo "  tail -f $LOG_DIR/gpu*_*lfwj_open_gripper*.log"
echo "  主日志: tail -f $LOG_DIR/batch_eval_rvs_6gpu_lfwj_open_gripper_*.log"
echo ""
echo "停止任务:"
echo "  kill $PID_GPU0 $PID_GPU1 $PID_GPU2 $PID_GPU3 $PID_GPU4 $PID_GPU5"
echo "=========================================="
