#!/bin/bash

# 后台批量评估脚本 - 直接后台运行版本
# 使用方法: nohup bash script/run_batch_eval_background.sh > /tmp/batch_eval.log 2>&1 &

# Batch evaluation script for pi05 policy on 50 tasks using 4 GPUs (0, 1, 2, 3)
# WITH GRIPPER RECOVERY TEST (--open_gripper enabled)
# All tasks will use demo_clean configuration for clean environment testing
# Runs in background with logging

# Configuration
POLICY_NAME="pi05"
TASK_CONFIG="demo_clean"  # Use clean environment for all tasks
TRAIN_CONFIG_NAME="pi05_aloha_full_base_all"
MODEL_NAME="pi05_ft_4a100"
CKPT_SETTING="motus"
SEED=0
TEST_NUM=50
TORCH_CHECKPOINT_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/policy/pi05/checkpoint_torch/motus"

# Available GPUs (4 GPUs: 0, 1, 2, 3)
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

# Task list (50 tasks)
TASKS=(
    "adjust_bottle"
    "beat_block_hammer"
    "blocks_ranking_rgb"
    "blocks_ranking_size"
    "click_alarmclock"
    "click_bell"
    "dump_bin_bigbin"
    "grab_roller"
    "handover_block"
    "handover_mic"
    "hanging_mug"
    "lift_pot"
    "move_can_pot"
    "move_pillbottle_pad"
    "move_playingcard_away"
    "move_stapler_pad"
    "open_laptop"
    "open_microwave"
    "pick_diverse_bottles"
    "pick_dual_bottles"
    "place_a2b_left"
    "place_a2b_right"
    "place_bread_basket"
    "place_bread_skillet"
    "place_burger_fries"
    "place_can_basket"
    "place_cans_plasticbox"
    "place_container_plate"
    "place_dual_shoes"
    "place_empty_cup"
    "place_fan"
    "place_mouse_pad"
    "place_object_basket"
    "place_object_scale"
    "place_object_stand"
    "place_phone_stand"
    "place_shoe"
    "press_stapler"
    "put_bottles_dustbin"
    "put_object_cabinet"
    "rotate_qrcode"
    "scan_object"
    "shake_bottle_horizontally"
    "shake_bottle"
    "stack_blocks_three"
    "stack_blocks_two"
    "stack_bowls_three"
    "stack_bowls_two"
    "stamp_seal"
    "turn_switch"
)

# Create log directory
LOG_DIR="/data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs"
mkdir -p "$LOG_DIR"

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/batch_eval_4gpu_open_gripper_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/batch_eval_pids_${TIMESTAMP}.txt"

# Function to run evaluation for tasks on a specific GPU
run_eval_gpu() {
    local gpu_id=$1
    local gpu_index=$2
    local log_file="$LOG_DIR/gpu${gpu_id}_open_gripper_${TIMESTAMP}.log"
    local failed_tasks_file="$LOG_DIR/failed_tasks_gpu${gpu_id}_open_gripper_${TIMESTAMP}.txt"
    
    {
        echo "=========================================="
        echo "GPU ${gpu_id} Evaluation Started (WITH GRIPPER RECOVERY TEST)"
        echo "Timestamp: $(date)"
        echo "=========================================="
        echo ""
        
        cd /data2/liangxiwen/zkd/cry/RoboTwin
        
        # Activate pi05 virtual environment
        source /data2/liangxiwen/zkd/cry/RoboTwin/policy/pi05/.venv/bin/activate
        
        # Set GPU
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        
        # Initialize counters
        local task_count=0
        local success_count=0
        local failed_count=0
        
        # Get tasks for this GPU (round-robin distribution)
        for i in "${!TASKS[@]}"; do
            if [ $((i % NUM_GPUS)) -eq $gpu_index ]; then
                task="${TASKS[$i]}"
                task_index=$((i + 1))
                ((task_count++))
                
                echo "=========================================="
                echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting evaluation: Task $task_index/${#TASKS[@]}"
                echo "Task: $task"
                echo "GPU: $gpu_id"
                echo "Config: $TASK_CONFIG"
                echo "Test episodes: $TEST_NUM"
                echo "Checkpoint: $TORCH_CHECKPOINT_DIR"
                echo "Gripper Recovery Test: ENABLED (30 frames)"
                echo "=========================================="
                
                # Run evaluation with timeout (optional, set to 0 to disable)
                # Using timeout to prevent hanging tasks
                # NOTE: Using eval_policy_initial_motus_open_gripper.py with --open_gripper flag
                PYTHONWARNINGS=ignore::UserWarning \
                timeout 7200 python script/eval_policy_initial_motus_open_gripper.py \
                    --config policy/$POLICY_NAME/deploy_policy.yml \
                    --open_gripper \
                    --overrides \
                    --task_name ${task} \
                    --task_config ${TASK_CONFIG} \
                    --train_config_name ${TRAIN_CONFIG_NAME} \
                    --model_name ${MODEL_NAME} \
                    --ckpt_setting ${CKPT_SETTING} \
                    --seed ${SEED} \
                    --policy_name ${POLICY_NAME} \
                    --use_torch_checkpoint True \
                    --torch_checkpoint_dir ${TORCH_CHECKPOINT_DIR} \
                    --test_num ${TEST_NUM} 2>&1
                
                local exit_code=$?
                
                # Handle different exit codes
                if [ $exit_code -eq 0 ]; then
                    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ✅ Task $task completed successfully on GPU $gpu_id"
                    ((success_count++))
                elif [ $exit_code -eq 124 ]; then
                    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ⏱️  Task $task TIMEOUT on GPU $gpu_id (exceeded 2 hours)"
                    echo "${task}|${gpu_id}|TIMEOUT|$(date +'%Y-%m-%d %H:%M:%S')" >> "$failed_tasks_file"
                    ((failed_count++))
                else
                    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ❌ Task $task FAILED on GPU $gpu_id (exit code: $exit_code)"
                    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  Continuing with next task..."
                    echo "${task}|${gpu_id}|FAILED|$(date +'%Y-%m-%d %H:%M:%S')|exit_code:$exit_code" >> "$failed_tasks_file"
                    ((failed_count++))
                fi
                echo ""
            fi
        done
        
        echo "=========================================="
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] GPU ${gpu_id} Evaluation Completed"
        echo "Total tasks processed: $task_count"
        echo "Successful: $success_count"
        echo "Failed: $failed_count"
        if [ $failed_count -gt 0 ]; then
            echo "Failed tasks saved to: $failed_tasks_file"
        fi
        echo "=========================================="
    } >> "$log_file" 2>&1
}

# Main execution
{
    echo "=========================================="
    echo "Batch Evaluation Script for pi05 Policy"
    echo "WITH GRIPPER RECOVERY TEST (--open_gripper)"
    echo "=========================================="
    echo "Total tasks: ${#TASKS[@]}"
    echo "Number of GPUs: $NUM_GPUS"
    echo "GPUs: ${GPUS[@]}"
    echo "Task config: $TASK_CONFIG (clean environment)"
    echo "Test episodes per task: $TEST_NUM"
    echo "Checkpoint dir: $TORCH_CHECKPOINT_DIR"
    echo "Gripper Recovery Test: ENABLED (30 frames)"
    echo "Log directory: $LOG_DIR"
    echo "Timestamp: $TIMESTAMP"
    echo "=========================================="
    echo ""
} | tee "$MAIN_LOG"

# Check if checkpoint directory exists
if [ ! -d "$TORCH_CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory not found: $TORCH_CHECKPOINT_DIR" | tee -a "$MAIN_LOG"
    exit 1
fi

echo "✅ Checkpoint directory verified: $TORCH_CHECKPOINT_DIR" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Start background processes for each GPU using nohup and setsid
declare -a pids=()
for i in "${!GPUS[@]}"; do
    gpu_id=${GPUS[$i]}
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] 启动 GPU ${gpu_id} 的评估任务 (带夹爪恢复测试)..." | tee -a "$MAIN_LOG"
    
    # Create a wrapper script for each GPU to ensure proper execution
    wrapper_script="/tmp/run_gpu${gpu_id}_open_gripper_${TIMESTAMP}.sh"
    cat > "$wrapper_script" << EOF
#!/bin/bash
cd /data2/liangxiwen/zkd/cry/RoboTwin
source policy/pi05/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Export all necessary variables
export POLICY_NAME="${POLICY_NAME}"
export TASK_CONFIG="${TASK_CONFIG}"
export TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME}"
export MODEL_NAME="${MODEL_NAME}"
export CKPT_SETTING="${CKPT_SETTING}"
export SEED=${SEED}
export TEST_NUM=${TEST_NUM}
export TORCH_CHECKPOINT_DIR="${TORCH_CHECKPOINT_DIR}"
export LOG_DIR="${LOG_DIR}"
export TIMESTAMP="${TIMESTAMP}"
export NUM_GPUS=${NUM_GPUS}

# Export tasks array
TASKS=($(printf '"%s" ' "${TASKS[@]}"))

# Define the function
$(declare -f run_eval_gpu)

# Run the evaluation
run_eval_gpu ${gpu_id} ${i}
EOF
    chmod +x "$wrapper_script"
    
    # Use setsid and nohup to ensure background execution
    setsid nohup bash "$wrapper_script" > /dev/null 2>&1 &
    pids+=($!)
    sleep 2
done

# Disown all background processes to prevent termination on terminal close
for pid in "${pids[@]}"; do
    disown $pid 2>/dev/null || true
done

# Save PID info
{
    echo ""
    echo "=========================================="
    echo "Batch evaluation started at: $(date)"
    echo "Mode: WITH GRIPPER RECOVERY TEST (--open_gripper)"
    echo "Total tasks: ${#TASKS[@]}"
    echo "GPUs: ${GPUS[@]}"
    echo "Test episodes per task: $TEST_NUM"
    echo "Checkpoint dir: $TORCH_CHECKPOINT_DIR"
    echo "Gripper Recovery Test: ENABLED (30 frames)"
    echo ""
    echo "GPU 进程 PID:"
    for i in "${!GPUS[@]}"; do
        gpu_id=${GPUS[$i]}
        echo "  GPU ${gpu_id}: ${pids[$i]}"
    done
    echo ""
    echo "日志文件:"
    for i in "${!GPUS[@]}"; do
        gpu_id=${GPUS[$i]}
        echo "  GPU ${gpu_id}: ${LOG_DIR}/gpu${gpu_id}_open_gripper_${TIMESTAMP}.log"
    done
    echo ""
    echo "任务分配:"
    for i in "${!GPUS[@]}"; do
        gpu_id=${GPUS[$i]}
        echo "  GPU ${gpu_id} tasks:"
        for j in "${!TASKS[@]}"; do
            if [ $((j % NUM_GPUS)) -eq $i ]; then
                echo "    - ${TASKS[$j]}"
            fi
        done
    done
    echo ""
    echo "失败任务将保存到:"
    for i in "${!GPUS[@]}"; do
        gpu_id=${GPUS[$i]}
        echo "  GPU ${gpu_id}: ${LOG_DIR}/failed_tasks_gpu${gpu_id}_open_gripper_${TIMESTAMP}.txt"
    done
    echo ""
    echo "=========================================="
    echo ""
    echo "查看实时日志命令:"
    for i in "${!GPUS[@]}"; do
        gpu_id=${GPUS[$i]}
        echo "  tail -f ${LOG_DIR}/gpu${gpu_id}_open_gripper_${TIMESTAMP}.log"
    done
    echo ""
    echo "查看进程状态:"
    echo "  ps -p $(IFS=','; echo "${pids[*]}")"
    echo ""
    echo "停止任务:"
    echo "  kill $(IFS=' '; echo "${pids[*]}")"
    echo ""
    echo "检查进度:"
    echo "  python script/check_progress_open_gripper.py"
    echo ""
    echo "监控 GPU 使用:"
    echo "  watch -n 1 nvidia-smi"
    echo ""
    echo "=========================================="
} | tee -a "$MAIN_LOG"

# Save PIDs to file
echo "${pids[@]}" > "$PID_FILE"
echo "PIDs saved to: $PID_FILE" | tee -a "$MAIN_LOG"

echo ""
echo "✅ ${NUM_GPUS} GPU并行批量评估任务已启动 (带夹爪恢复测试)"
echo "主日志已保存到: $MAIN_LOG"
echo ""
