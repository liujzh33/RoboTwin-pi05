#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
checkpoint_id=${7:-1}  # Default to 1 if not provided

# Parse optional flags (from position 8 onward)
USE_JUDGE=false
JUDGE_GPU=""
for arg in "${@:8}"; do
    case "$arg" in
        --use_judge_model) USE_JUDGE=true ;;
        --judge_gpu=*) JUDGE_GPU="${arg#*=}" ;;
    esac
done

echo -e "\033[33mcheckpoint_id: ${checkpoint_id}\033[0m"

source .venv/bin/activate

deploy_config="deploy_policy.yml"

if [ "$USE_JUDGE" = true ]; then
    echo -e "\033[92m[Judge Mode] Enabled — using deploy_policy_with_judge\033[0m"
    export USE_JUDGE_MODEL=1
    deploy_config="deploy_policy_with_judge.yml"
    JUDGE_USE_REMOTE=${JUDGE_USE_REMOTE:-1}
    export JUDGE_USE_REMOTE

    if [ "$JUDGE_USE_REMOTE" = "1" ]; then
        # Judge runs in another process/environment; Pi0.5 only uses its own GPU.
        export CUDA_VISIBLE_DEVICES=${gpu_id}
        export JUDGE_SERVER_URL=${JUDGE_SERVER_URL:-http://127.0.0.1:18080}
        echo -e "\033[33mCUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}\033[0m"
        echo -e "\033[33mPi0.5 device: cuda:0 (physical GPU ${gpu_id})\033[0m"
        echo -e "\033[33mJudge remote URL: ${JUDGE_SERVER_URL}\033[0m"
    else
        if [ -n "$JUDGE_GPU" ]; then
            # Make both Pi0.5 GPU and Judge GPU visible
            export CUDA_VISIBLE_DEVICES="${gpu_id},${JUDGE_GPU}"
            # After CUDA_VISIBLE_DEVICES remapping: pi05 is cuda:0, judge is cuda:1
            export JUDGE_DEVICE="cuda:1"
            echo -e "\033[33mCUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}\033[0m"
            echo -e "\033[33mPi0.5 device: cuda:0 (physical GPU ${gpu_id})\033[0m"
            echo -e "\033[33mJudge device: cuda:1 (physical GPU ${JUDGE_GPU})\033[0m"
        else
            # Judge shares the same GPU as Pi0.5
            export CUDA_VISIBLE_DEVICES=${gpu_id}
            export JUDGE_DEVICE="cuda:0"
            echo -e "\033[33mCUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}\033[0m"
            echo -e "\033[33mPi0.5 + Judge share cuda:0 (physical GPU ${gpu_id})\033[0m"
        fi
    fi
else
    export USE_JUDGE_MODEL=0
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
fi

cd ../.. # move to root
export PYTHONPATH="$(pwd)/policy/pi05/src:$(pwd)/policy/pi05/packages/openpi-client/src:${PYTHONPATH}"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/${deploy_config} \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --checkpoint_id ${checkpoint_id} 
