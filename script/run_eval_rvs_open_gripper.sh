#!/bin/bash
# 使用 RvS 权重 + open_gripper 扰动评估（eval_policy_initial_motus_open_gripper.py）
# 权重、norm_stats、任务与 run_eval_rvs.sh 相同，仅增加 --open_gripper

set -e
ROOT="/data2/liangxiwen/zkd/cry/RoboTwin"
OPENPI_RECOVERY2_SRC="${ROOT}/policy/recovery2/openpi-main/src"
VENV_PYTHON="${ROOT}/policy/pi05/.venv/bin/python"
CONFIG="${ROOT}/policy/pi05/deploy_policy.yml"
SCRIPT="${ROOT}/script/eval_policy_initial_motus_open_gripper.py"

RVS_CHECKPOINT_DIR="${ROOT}/policy/recovery2/openpi-main/RvS/10000"
NORM_STATS_PATH="${ROOT}/policy/recovery2/openpi-main/norm_stats.json"

TASK_NAME="${TASK_NAME:-blocks_ranking_rgb}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
GPU_ID="${GPU_ID:-0}"
TEST_NUM="${TEST_NUM:-20}"
REWARD_VALUE="${REWARD_VALUE:-1.0}"
# open_gripper 结果单独目录，便于与无扰动结果区分
CKPT_SETTING="${CKPT_SETTING:-rvs_10000_reward_$(echo "$REWARD_VALUE" | sed 's/-/neg/g' | sed 's/\./p/g')_open_gripper}"

if [[ -n "$1" ]]; then TASK_NAME="$1"; fi
if [[ -n "$2" ]]; then TASK_CONFIG="$2"; fi
if [[ -n "$3" ]]; then GPU_ID="$3"; fi
if [[ -n "$4" ]]; then TEST_NUM="$4"; fi
if [[ -n "$5" ]]; then REWARD_VALUE="$5"; fi

echo "=========================================="
echo "RvS 权重评估 (open_gripper 扰动)"
echo "  任务: $TASK_NAME ($TASK_CONFIG)"
echo "  GPU: $GPU_ID, 测试条数: $TEST_NUM"
echo "  Reward 值: $REWARD_VALUE"
echo "  权重: $RVS_CHECKPOINT_DIR"
echo "  norm_stats: $NORM_STATS_PATH"
echo "  PYTHONPATH: $OPENPI_RECOVERY2_SRC"
echo "=========================================="

cd "$ROOT"
export PYTHONPATH="${OPENPI_RECOVERY2_SRC}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$VENV_PYTHON" "$SCRIPT" \
  --config "$CONFIG" \
  --open_gripper \
  --overrides \
  --task_name "$TASK_NAME" \
  --task_config "$TASK_CONFIG" \
  --train_config_name pi05_rl_aloha_reward \
  --model_name pi05_ft_4a100 \
  --ckpt_setting "$CKPT_SETTING" \
  --seed 0 \
  --policy_name pi05_rvs \
  --use_torch_checkpoint True \
  --torch_checkpoint_dir "$RVS_CHECKPOINT_DIR" \
  --norm_stats_path "$NORM_STATS_PATH" \
  --pi0_step 32 \
  --reward_value "$REWARD_VALUE" \
  --test_num "$TEST_NUM"
