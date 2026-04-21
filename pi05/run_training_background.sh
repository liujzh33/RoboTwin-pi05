#!/bin/bash
# 后台运行训练脚本

train_config_name=$1
model_name=$2
gpu_use=$3

# 设置日志文件
log_dir="logs"
mkdir -p $log_dir
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="$log_dir/training_${train_config_name}_${model_name}_${timestamp}.log"

echo "=========================================="
echo "开始后台训练"
echo "训练配置: $train_config_name"
echo "模型名称: $model_name"
echo "使用GPU: $gpu_use"
echo "日志文件: $log_file"
echo "=========================================="

# 使用 nohup 后台运行
nohup bash finetune.sh $train_config_name $model_name $gpu_use > $log_file 2>&1 &

# 获取进程ID
pid=$!
echo "训练进程已启动，PID: $pid"
echo "查看日志: tail -f $log_file"
echo "查看进程: ps -p $pid"
echo ""
echo "要停止训练，运行: kill $pid"

