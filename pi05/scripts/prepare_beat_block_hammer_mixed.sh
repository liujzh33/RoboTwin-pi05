#!/usr/bin/env bash
# 将干净 + recovery 四份 processed_data 合并到 training_data/beat_block_hammer_mixed_pi05 并调用 generate.sh
# 用法: 在 policy/pi05 下
#   bash scripts/prepare_beat_block_hammer_mixed.sh [processed_data 根目录，默认 ./processed_data]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PI05_DIR"

PROCESSED_ROOT="${1:-$PI05_DIR/processed_data}"
OUT_DIR="$PI05_DIR/training_data/beat_block_hammer_mixed_pi05"
REPO_ID="beat_block_hammer_mixed_pi05"

DIRS=(
  "beat_block_hammer-aloha-agilex_clean_50-50"
  "beat_block_hammer-aloha-agilex_randomized_500-500"
  "beat_block_hammer-demo_clean-recovery-66"
  "beat_block_hammer-demo_randomized-recovery-44"
)

echo "processed_data root: $PROCESSED_ROOT"
mkdir -p "$OUT_DIR"
for d in "${DIRS[@]}"; do
  src="$PROCESSED_ROOT/$d"
  if [[ ! -d "$src" ]]; then
    echo "ERROR: missing directory: $src"
    exit 1
  fi
  echo "cp -r $src -> $OUT_DIR/"
  cp -r "$src" "$OUT_DIR/"
done

echo "Running generate.sh -> repo_id=$REPO_ID"
bash "$PI05_DIR/generate.sh" "$OUT_DIR" "$REPO_ID"
echo "Done. Next:"
echo "  uv run scripts/compute_norm_stats.py --config-name pi05_aloha_beat_block_hammer_mixed_recovery"
echo "  bash finetune.sh pi05_aloha_beat_block_hammer_mixed_recovery <exp_name> <gpus>"
