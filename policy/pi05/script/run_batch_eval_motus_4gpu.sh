#!/bin/bash

# 并行启动批量评估脚本（4个GPU版本）
# 使用 GPU 0, 1, 2, 3 分别运行剩余任务
# 自动跳过已完成的任务

SCRIPT_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/script"
BATCH_SCRIPT="$SCRIPT_DIR/batch_eval_motus_4gpu.sh"
LOG_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "启动4GPU并行批量评估任务"
echo "=========================================="
echo "GPU 0, 1, 2, 3: 自动分配剩余未完成任务"
echo "每个任务测试 50 次"
echo "自动跳过已完成的任务"
echo ""

# 启动4个GPU的任务
echo "启动 GPU 0 的评估任务..."
nohup bash "$BATCH_SCRIPT" 0 > "$LOG_DIR/gpu0_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU0=$!

echo "启动 GPU 1 的评估任务..."
nohup bash "$BATCH_SCRIPT" 1 > "$LOG_DIR/gpu1_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU1=$!

echo "启动 GPU 2 的评估任务..."
nohup bash "$BATCH_SCRIPT" 2 > "$LOG_DIR/gpu2_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU2=$!

echo "启动 GPU 3 的评估任务..."
nohup bash "$BATCH_SCRIPT" 3 > "$LOG_DIR/gpu3_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU3=$!

echo ""
echo "✅ 4GPU并行批量评估任务已启动"
echo "=========================================="
echo "GPU 0 进程 PID: $PID_GPU0"
echo "GPU 1 进程 PID: $PID_GPU1"
echo "GPU 2 进程 PID: $PID_GPU2"
echo "GPU 3 进程 PID: $PID_GPU3"
echo ""
echo "查看实时日志:"
echo "  GPU 0: tail -f $LOG_DIR/gpu0_*.log"
echo "  GPU 1: tail -f $LOG_DIR/gpu1_*.log"
echo "  GPU 2: tail -f $LOG_DIR/gpu2_*.log"
echo "  GPU 3: tail -f $LOG_DIR/gpu3_*.log"
echo "  主日志: tail -f $LOG_DIR/batch_eval_4gpu_*.log"
echo ""
echo "查看进程状态:"
echo "  ps -p $PID_GPU0,$PID_GPU1,$PID_GPU2,$PID_GPU3"
echo ""
echo "停止任务:"
echo "  kill $PID_GPU0 $PID_GPU1 $PID_GPU2 $PID_GPU3"
echo ""
echo "监控 GPU 使用:"
echo "  watch -n 1 nvidia-smi"
echo "=========================================="

