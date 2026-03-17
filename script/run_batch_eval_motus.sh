#!/bin/bash

# 快速启动批量评估脚本的包装器
# 使用 nohup 在后台运行，并输出 PID

SCRIPT_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/script"
BATCH_SCRIPT="$SCRIPT_DIR/batch_eval_motus_50_tasks.sh"
LOG_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 使用 nohup 在后台运行批量脚本
echo "启动批量评估任务..."
echo "脚本路径: $BATCH_SCRIPT"
echo "日志目录: $LOG_DIR"
echo ""

# 运行脚本并获取 PID
nohup bash "$BATCH_SCRIPT" > "$LOG_DIR/batch_eval_wrapper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID=$!

echo "✅ 批量评估任务已启动"
echo "进程 PID: $PID"
echo ""
echo "查看实时日志:"
echo "  tail -f $LOG_DIR/batch_eval_*.log"
echo ""
echo "查看进程状态:"
echo "  ps -p $PID"
echo ""
echo "停止任务:"
echo "  kill $PID"

