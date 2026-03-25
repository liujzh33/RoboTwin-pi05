#!/bin/bash

# 并行启动批量评估脚本
# 使用 GPU 0 和 GPU 1 分别运行25个任务

SCRIPT_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/script"
BATCH_SCRIPT="$SCRIPT_DIR/batch_eval_motus_parallel.sh"
LOG_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "启动并行批量评估任务"
echo "=========================================="
echo "GPU 0: 任务 0-24 (25个任务)"
echo "GPU 1: 任务 25-49 (25个任务)"
echo "每个任务测试 50 次"
echo ""

# 启动 GPU 0 的任务 (0-24)
echo "启动 GPU 0 的评估任务..."
nohup bash "$BATCH_SCRIPT" 0 0 24 > "$LOG_DIR/gpu0_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU0=$!

# 启动 GPU 1 的任务 (25-49)
echo "启动 GPU 1 的评估任务..."
nohup bash "$BATCH_SCRIPT" 1 25 49 > "$LOG_DIR/gpu1_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU1=$!

echo ""
echo "✅ 并行批量评估任务已启动"
echo "=========================================="
echo "GPU 0 进程 PID: $PID_GPU0"
echo "GPU 1 进程 PID: $PID_GPU1"
echo ""
echo "查看实时日志:"
echo "  GPU 0: tail -f $LOG_DIR/gpu0_*.log"
echo "  GPU 1: tail -f $LOG_DIR/gpu1_*.log"
echo "  主日志: tail -f $LOG_DIR/batch_eval_parallel_*.log"
echo ""
echo "查看进程状态:"
echo "  ps -p $PID_GPU0,$PID_GPU1"
echo ""
echo "停止任务:"
echo "  kill $PID_GPU0 $PID_GPU1"
echo ""
echo "监控 GPU 使用:"
echo "  watch -n 1 nvidia-smi"
echo "=========================================="

