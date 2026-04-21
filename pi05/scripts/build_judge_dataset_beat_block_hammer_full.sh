#!/usr/bin/env bash
# 一键构建 Judge 训练集：从两套 recovery（error） + 两套 clean（success）提取，合并为一份 JSON。
#
# 数据源：
#   - Error: processed_data/beat_block_hammer-demo_clean-recovery-66
#           processed_data/beat_block_hammer-demo_randomized-recovery-44
#   - Success: training_data/.../beat_block_hammer-aloha-agilex_clean_50-50
#              training_data/.../beat_block_hammer-aloha-agilex_randomized_500-500
#
# 输出：judge_dataset/recovery_judge_beat_block_hammer/
#   - images/          所有图片
#   - recovery_judge_dataset.json        仅 error 样本
#   - recovery_judge_success_dataset.json 仅 success 样本
#   - recovery_judge_dataset_merged.json  error + success 打乱，供 LLaMA-Factory 训练
#
# 用法（项目根目录）:
#   cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
#   bash scripts/build_judge_dataset_beat_block_hammer_full.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

OUT_DIR="${ROOT}/judge_dataset/recovery_judge_beat_block_hammer"
PROCESSED_CLEAN="${ROOT}/processed_data/beat_block_hammer-demo_clean-recovery-66"
PROCESSED_RAND="${ROOT}/processed_data/beat_block_hammer-demo_randomized-recovery-44"
RAW_CLEAN="/mnt/data1/liujingzhi/dataset_recovery/beat_block_hammer/demo_clean/data"
RAW_RAND="/mnt/data1/liujingzhi/dataset_recovery/beat_block_hammer/demo_randomized/data"
TRAIN_CLEAN="${ROOT}/training_data/pi05_multi_task_5_v1.0/beat_block_hammer-aloha-agilex_clean_50-50"
TRAIN_RAND="${ROOT}/training_data/pi05_multi_task_5_v1.0/beat_block_hammer-aloha-agilex_randomized_500-500"
RAW_TRAIN_CLEAN="/mnt/data1/liujingzhi/dataset/beat_block_hammer/aloha-agilex_clean_50/data"
RAW_TRAIN_RAND="/mnt/data1/liujingzhi/dataset/beat_block_hammer/aloha-agilex_randomized_500/data"

echo "========== 1. 提取 error 样本（两套 recovery）=========="
python scripts/extract_judge_dataset_beat_block_hammer.py \
  --processed_data "$PROCESSED_CLEAN" "$PROCESSED_RAND" \
  --raw_data_dir "$RAW_CLEAN" "$RAW_RAND" \
  --out_dir "$OUT_DIR"

echo ""
echo "========== 2. 提取 success 样本（clean/rand 各 50 条）并合并为最终训练集 =========="
python scripts/extract_judge_success_samples_beat_block_hammer.py \
  --training_data_dirs "$TRAIN_CLEAN" "$TRAIN_RAND" \
  --raw_data_dirs "$RAW_TRAIN_CLEAN" "$RAW_TRAIN_RAND" \
  --out_dir "$OUT_DIR" \
  --max_per_source 50 \
  --merge_with_errors "$OUT_DIR/recovery_judge_dataset.json"

echo ""
echo "Done. Use for LLaMA-Factory:"
echo "  file_name: $OUT_DIR/recovery_judge_dataset_merged.json"
echo "  (or recovery_judge_dataset.json for errors-only)"
