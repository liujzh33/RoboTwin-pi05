#!/bin/bash
# 打包 pi05 项目，排除大文件夹

SOURCE_DIR="/mnt/data/guoxiaoyu/RoboTwin/policy/pi05"
OUTPUT_DIR="/mnt/data/guoxiaoyu/RoboTwin/policy"
ARCHIVE_NAME="pi05_packaged_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "=========================================="
echo "开始打包 pi05 项目"
echo "=========================================="
echo "源目录: $SOURCE_DIR"
echo "输出文件: $OUTPUT_DIR/$ARCHIVE_NAME"
echo ""
echo "排除的文件夹:"
echo "  - checkpoints/"
echo "  - processed_data/"
echo "  - training_data/"
echo "=========================================="

cd "$SOURCE_DIR/.."

# 使用 tar 打包，排除三个大文件夹
tar -czf "$OUTPUT_DIR/$ARCHIVE_NAME" \
    --exclude='pi05/checkpoints' \
    --exclude='pi05/processed_data' \
    --exclude='pi05/training_data' \
    --exclude='pi05/.git' \
    --exclude='pi05/__pycache__' \
    --exclude='pi05/**/__pycache__' \
    --exclude='pi05/**/*.pyc' \
    --exclude='pi05/**/*.pyo' \
    pi05/

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 打包完成！"
    echo "=========================================="
    echo "压缩包位置: $OUTPUT_DIR/$ARCHIVE_NAME"
    echo "压缩包大小: $(du -sh "$OUTPUT_DIR/$ARCHIVE_NAME" | cut -f1)"
    echo ""
    echo "下一步:"
    echo "1. 将压缩包传输到目标服务器"
    echo "2. 解压后运行 fix_venv_paths.sh 修复 .venv 路径"
else
    echo ""
    echo "=========================================="
    echo "❌ 打包失败！"
    echo "=========================================="
    exit 1
fi

