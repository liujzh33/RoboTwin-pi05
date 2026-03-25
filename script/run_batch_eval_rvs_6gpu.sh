#!/bin/bash

# 并行启动批量评估脚本（6个GPU版本）
# 使用 GPU 0, 1, 2, 3, 4, 5 分别运行剩余任务
# 自动跳过已完成的任务
# 10个任务 × 2种环境（clean + random）= 20个任务
# 使用 RvS 权重（recovery2 openpi 架构，带 reward 输入）

SCRIPT_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/script"
BATCH_SCRIPT="$SCRIPT_DIR/batch_eval_rvs_6gpu.sh"
LOG_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_recovery_logs"

# 默认 reward 值（可以通过环境变量覆盖）
REWARD_VALUE="${REWARD_VALUE:-1.0}"

# 创建日志目录
mkdir -p "$LOG_DIR"
# 清除旧的日志文件（清除所有GPU的旧日志）
echo "清除旧的日志文件..."
rm -f "$LOG_DIR"/*.log 2>/dev/null
echo "日志已清除"

echo "=========================================="
echo "启动6GPU并行批量评估任务（RvS权重）"
echo "=========================================="
echo "GPU 0, 1, 2, 3, 4, 5: 自动分配剩余未完成任务"
echo "任务配置: 10个任务 × 2种环境（clean + random）= 20个任务"
echo "每个任务测试 20 次"
echo "Reward 值: $REWARD_VALUE"
echo "权重路径: /data2/liangxiwen/zkd/cry/RoboTwin/policy/recovery2/openpi-main/RvS/10000"
echo "使用 RvS 权重（recovery2 openpi 架构，带 reward 输入）"
echo "自动跳过已完成的任务"
echo ""

# 启动6个GPU的任务
echo "启动 GPU 0 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 0 > "$LOG_DIR/gpu0_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU0=$!

echo "启动 GPU 1 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 1 > "$LOG_DIR/gpu1_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU1=$!

echo "启动 GPU 2 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 2 > "$LOG_DIR/gpu2_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU2=$!

echo "启动 GPU 3 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 3 > "$LOG_DIR/gpu3_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU3=$!

echo "启动 GPU 4 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 4 > "$LOG_DIR/gpu4_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU4=$!

echo "启动 GPU 5 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 5 > "$LOG_DIR/gpu5_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU5=$!

echo ""
echo "✅ 6GPU并行批量评估任务已启动"
echo "=========================================="
echo "GPU 0 进程 PID: $PID_GPU0"
echo "GPU 1 进程 PID: $PID_GPU1"
echo "GPU 2 进程 PID: $PID_GPU2"
echo "GPU 3 进程 PID: $PID_GPU3"
echo "GPU 4 进程 PID: $PID_GPU4"
echo "GPU 5 进程 PID: $PID_GPU5"
echo ""
echo "查看实时日志:"
echo "  GPU 0: tail -f $LOG_DIR/gpu0_*.log"
echo "  GPU 1: tail -f $LOG_DIR/gpu1_*.log"
echo "  GPU 2: tail -f $LOG_DIR/gpu2_*.log"
echo "  GPU 3: tail -f $LOG_DIR/gpu3_*.log"
echo "  GPU 4: tail -f $LOG_DIR/gpu4_*.log"
echo "  GPU 5: tail -f $LOG_DIR/gpu5_*.log"
echo "  主日志: tail -f $LOG_DIR/batch_eval_rvs_6gpu_*.log"
echo ""
echo "查看进程状态:"
echo "  ps -p $PID_GPU0,$PID_GPU1,$PID_GPU2,$PID_GPU3,$PID_GPU4,$PID_GPU5"
echo ""
echo "停止任务:"
echo "  kill $PID_GPU0 $PID_GPU1 $PID_GPU2 $PID_GPU3 $PID_GPU4 $PID_GPU5"
echo ""
echo "监控 GPU 使用:"
echo "  watch -n 1 nvidia-smi"
echo "=========================================="
