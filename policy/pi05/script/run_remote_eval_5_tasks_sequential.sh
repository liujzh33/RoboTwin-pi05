#!/bin/bash
# =============================================================================
# 远程评估：5 个任务，每个跑 clean + random，共 10 次；每次 5 个 eval，输出打终端
# 任务: press_stapler, place_mouse_pad, open_laptop, lift_pot, place_bread_skillet
# 用法: ./script/run_remote_eval_5_tasks_sequential.sh
# 可选: CLIENT_GPU=1
# =============================================================================

set -e
cd /data2/liangxiwen/zkd/cry/RoboTwin
source policy/pi05/.venv/bin/activate

if [ -n "${CLIENT_GPU}" ]; then
    export CUDA_VISIBLE_DEVICES="${CLIENT_GPU}"
    export NVIDIA_VISIBLE_DEVICES="${CLIENT_GPU}"
fi

REMOTE_HOST="${REMOTE_HOST:-36.163.20.107}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
TEST_NUM="${TEST_NUM:-50}"
SEED=0

# 只跑这 5 个任务，每个跑 demo_clean + demo_randomized → 共 10 次
BASE_TASKS=(
    "blocks_ranking_rgb"
    "blocks_ranking_size"
    "hanging_mug"
    "lift_pot"
    "place_mouse_pad"
    "press_stapler"
    "move_stapler_pad"
    "open_laptop"
    "place_bread_skillet"
    "place_can_basket"
)
ENV_CONFIGS=("demo_clean")
TOTAL=10

echo "=========================================="
echo "Remote Eval: 5 tasks × 2 envs = 10 runs, ${TEST_NUM} evals each (output to terminal)"
echo "Tasks: press_stapler, place_mouse_pad, open_laptop, lift_pot, place_bread_skillet"
echo "Server: ${REMOTE_HOST}:${REMOTE_PORT}"
echo "Started: $(date)"
echo "=========================================="

total=0
success_count=0
failed_tasks=()

for task_name in "${BASE_TASKS[@]}"; do
    for task_config in "${ENV_CONFIGS[@]}"; do
        total=$((total + 1))
        echo ""
        echo "========== [${total}/${TOTAL}] ${task_name} | ${task_config} =========="

        set +e
        python script/eval_policy_client.py \
            --config policy/pi05/deploy_policy.yml \
            --host "$REMOTE_HOST" \
            --port "$REMOTE_PORT" \
            --use-websocket \
            --overrides \
            task_name "$task_name" \
            task_config "$task_config" \
            train_config_name pi05_aloha_full_base_all \
            model_name pi05_ft_4a100 \
            ckpt_setting rvs_10000 \
            seed "$SEED" \
            policy_name pi05 \
            test_num "$TEST_NUM"
        rc=$?
        set -e

        if [ "$rc" -eq 0 ]; then
            success_count=$((success_count + 1))
            echo "[${total}/${TOTAL}] ✅ ${task_name}|${task_config}"
        else
            failed_tasks+=("${task_name}|${task_config}")
            echo "[${total}/${TOTAL}] ❌ ${task_name}|${task_config}"
        fi
    done
done

echo ""
echo "=========================================="
echo "Finished: $(date)  Completed: ${success_count}/${TOTAL}"
if [ ${#failed_tasks[@]} -gt 0 ]; then
    echo "Failed:"
    printf '  %s\n' "${failed_tasks[@]}"
fi
echo "=========================================="
