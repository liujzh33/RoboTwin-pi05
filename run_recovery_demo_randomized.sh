#!/usr/bin/env bash
# 对 dataset_recovery/beat_block_hammer/demo_randomized 做与 demo_clean 相同的处理，
# 得到 processed_data/beat_block_hammer-demo_randomized-recovery-<N>，并执行子任务划分（v3）。
# 之后需手动：丰富语义、批量画图、再跑 extract_judge_dataset 提 error 样本。
#
# 用法（在项目根目录）:
#   source .venv/bin/activate
#   export XDG_CACHE_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache
#   bash run_recovery_demo_randomized.sh
#
# 若只处理前 M 条，可先: export RECOVERY_N=200 再运行。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# 0 = 自动检测 data 目录下全部 episode
RECOVERY_N="${RECOVERY_N:-0}"

echo "========== 1. 原始 recovery (demo_randomized) → processed_data =========="
bash process_recovery_data_pi0.sh beat_block_hammer demo_randomized "$RECOVERY_N"

# 检测输出目录名（含实际 episode 数）
OUT_NAME=""
for d in processed_data/beat_block_hammer-demo_randomized-recovery-*; do
  [ -d "$d" ] && OUT_NAME=$(basename "$d") && break
done
if [ -z "$OUT_NAME" ]; then
  echo "Error: No processed_data/beat_block_hammer-demo_randomized-recovery-* found."
  exit 1
fi

echo ""
echo "========== 2. 子任务划分（与 demo_clean-recovery 相同 v3 逻辑）=========="
python scripts/process_data_generic_recovery_v3.py \
  --task_name beat_block_hammer \
  --data_dir "processed_data/$OUT_NAME"

echo ""
echo "Done. Next (optional):"
echo "  - Enrich semantics (e.g. Qwen3-VL enrich_beat_block_hammer.py for $OUT_NAME)"
echo "  - Batch plot: python \"scripts/batch_analyze_trajectories_beat_block_hammer .py\" --data_dir processed_data/$OUT_NAME"
echo "  - Extract judge error samples: python scripts/extract_judge_dataset_beat_block_hammer.py --processed_data processed_data/$OUT_NAME --raw_data_dir /mnt/data1/liujingzhi/dataset_recovery/beat_block_hammer/demo_randomized/data --out_dir judge_dataset/recovery_judge_beat_block_hammer"
