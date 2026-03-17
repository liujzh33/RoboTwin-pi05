#!/bin/bash

# 并行启动批量评估脚本（6个GPU，open_gripper 扰动）
# 使用 eval_policy_initial_motus_open_gripper.py，其余不变：RvS 权重、norm_stats、任务、6 张卡
# 使用 GPU 0, 1, 2, 3, 4, 5 分别运行剩余任务
# 自动跳过已完成的任务
# 10个任务 × 2种环境（clean + random）= 20个任务

SCRIPT_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/script"
BATCH_SCRIPT="$SCRIPT_DIR/batch_eval_rvs_6gpu_open_gripper.sh"
LOG_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_recovery_logs"

# 默认 reward 值（可以通过环境变量覆盖）
REWARD_VALUE="${REWARD_VALUE:-1.0}"

# 创建日志目录
mkdir -p "$LOG_DIR"
# 清除旧的 open_gripper 相关日志（保留非 open_gripper 的日志）
echo "清除旧的 open_gripper 日志文件..."
rm -f "$LOG_DIR"/gpu*_wrapper_open_gripper_*.log "$LOG_DIR"/batch_eval_rvs_6gpu_open_gripper_*.log 2>/dev/null
rm -f "$LOG_DIR"/gpu*_*_open_gripper_*.log 2>/dev/null
echo "open_gripper 日志已清除"

echo "=========================================="
echo "启动6GPU并行批量评估任务（RvS + open_gripper 扰动）"
echo "=========================================="
echo "GPU 0, 1, 2, 3, 4, 5: 自动分配剩余未完成任务"
echo "任务配置: 10个任务 × 2种环境（clean + random）= 20个任务"
echo "每个任务测试 20 次"
echo "Reward 值: $REWARD_VALUE"
echo "评估脚本: eval_policy_initial_motus_open_gripper.py (--open_gripper)"
echo "权重路径: /data2/liangxiwen/zkd/cry/RoboTwin/policy/recovery2/openpi-main/RvS/10000"
echo "norm_stats: /data2/liangxiwen/zkd/cry/RoboTwin/policy/recovery2/openpi-main/norm_stats.json"
echo "自动跳过已完成的任务"
echo ""

# 启动6个GPU的任务
echo "启动 GPU 0 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 0 > "$LOG_DIR/gpu0_wrapper_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU0=$!

echo "启动 GPU 1 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 1 > "$LOG_DIR/gpu1_wrapper_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU1=$!

echo "启动 GPU 2 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 2 > "$LOG_DIR/gpu2_wrapper_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU2=$!

echo "启动 GPU 3 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 3 > "$LOG_DIR/gpu3_wrapper_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU3=$!

echo "启动 GPU 4 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 4 > "$LOG_DIR/gpu4_wrapper_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU4=$!

echo "启动 GPU 5 的评估任务..."
REWARD_VALUE="$REWARD_VALUE" nohup bash "$BATCH_SCRIPT" 5 > "$LOG_DIR/gpu5_wrapper_open_gripper_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
PID_GPU5=$!

echo ""
echo "✅ 6GPU 并行批量评估任务已启动（open_gripper 扰动）"
echo "=========================================="
echo "GPU 0 进程 PID: $PID_GPU0"
echo "GPU 1 进程 PID: $PID_GPU1"
echo "GPU 2 进程 PID: $PID_GPU2"
echo "GPU 3 进程 PID: $PID_GPU3"
echo "GPU 4 进程 PID: $PID_GPU4"
echo "GPU 5 进程 PID: $PID_GPU5"
echo ""
echo "查看实时日志:"
echo "  GPU 0: tail -f $LOG_DIR/gpu0_*open_gripper*.log"
echo "  GPU 1: tail -f $LOG_DIR/gpu1_*open_gripper*.log"
echo "  ..."
echo "  主日志: tail -f $LOG_DIR/batch_eval_rvs_6gpu_open_gripper_*.log"
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
