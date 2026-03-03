#!/bin/bash
# 全量打包脚本：打包整个 pi05 目录（包含 .venv）
# 使用方法: bash package_full.sh [输出文件路径]

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PARENT_DIR="$(dirname "$PROJECT_DIR")"
OUTPUT_FILE="${1:-${PARENT_DIR}/pi05_full.tar.gz}"

echo "=========================================="
echo "PI0.5 全量打包脚本（包含虚拟环境）"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "输出文件: $OUTPUT_FILE"
echo ""

# 检查 .venv 是否存在
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "警告: .venv 目录不存在"
    read -p "是否继续打包? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "开始打包（包含虚拟环境，约 10GB，可能需要较长时间）..."
echo ""

# 获取输出文件名（不包含路径），用于排除
OUTPUT_BASENAME=$(basename "$OUTPUT_FILE")

# 打包整个目录，排除一些不必要的文件
cd "$PARENT_DIR"

# 构建排除输出文件的路径（相对于当前目录）
EXCLUDE_OUTPUT="pi05/${OUTPUT_BASENAME}"

tar -czf "$OUTPUT_FILE" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='wandb' \
    --exclude='checkpoints' \
    --exclude='processed_data' \
    --exclude='training_data' \
    --exclude='gripper_plots' \
    --exclude='*.zip' \
    --exclude='.ipynb_checkpoints' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    --exclude='pi05_package' \
    --exclude='pi05_full.tar.gz' \
    --exclude='package_*.sh' \
    --exclude='restore_*.sh' \
    --exclude='*_PACKAGE*.md' \
    --exclude='.venv/lib/python*/test' \
    --exclude='.venv/lib/python*/dist-packages/test' \
    --exclude='.venv/lib/python*/site-packages/test' \
    --exclude='.venv/pyvenv.cfg' \
    --exclude='.venv/bin/python*' \
    pi05/

echo ""
echo "=========================================="
echo "打包完成！"
echo "=========================================="
echo "输出文件: $OUTPUT_FILE"
echo "文件大小: $(du -sh "$OUTPUT_FILE" | cut -f1)"
echo ""
echo "传输到新服务器:"
echo "  scp $OUTPUT_FILE user@new_server:/path/to/destination/"
echo ""
echo "在新服务器解压和使用:"
echo "  cd /mnt/data1/liujingzhi/RoboTwin/policy/"
echo "  tar -xzf pi05_full.tar.gz"
echo "  cd pi05"
echo "  conda activate RoboTwin"
echo "  source .venv/bin/activate"
echo "  # 修复虚拟环境路径（见 restore_full.sh）"
echo ""

