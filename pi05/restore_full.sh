#!/bin/bash
# 全量恢复脚本：在新服务器上恢复完整的 pi05 环境
# 使用方法: bash restore_full.sh [pi05_full.tar.gz路径] [解压目录]

set -e

TAR_FILE="${1:-pi05_full.tar.gz}"
EXTRACT_DIR="${2:-/mnt/data1/liujingzhi/RoboTwin/policy}"

echo "=========================================="
echo "PI0.5 全量环境恢复脚本"
echo "=========================================="
echo "压缩文件: $TAR_FILE"
echo "解压目录: $EXTRACT_DIR"
echo ""

# 检查文件是否存在
if [ ! -f "$TAR_FILE" ]; then
    echo "错误: 找不到文件 $TAR_FILE"
    exit 1
fi

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 conda"
    exit 1
fi

# 检查 RoboTwin 环境
echo "检查 RoboTwin conda 环境..."
if ! conda env list | grep -q "^RoboTwin "; then
    echo "警告: 未找到 RoboTwin conda 环境"
    echo "请先创建: conda create -n RoboTwin python=3.10 -y"
    exit 1
fi

ROBOTWIN_ENV_PATH=$(conda env list | grep "^RoboTwin " | awk '{print $NF}')
echo "✓ 找到 RoboTwin conda 环境: $ROBOTWIN_ENV_PATH"

# 创建解压目录
echo "创建解压目录..."
mkdir -p "$EXTRACT_DIR"

# 解压
echo "解压文件（可能需要一些时间）..."
cd "$EXTRACT_DIR"
tar -xzf "$(realpath "$TAR_FILE")"

PI05_DIR="$EXTRACT_DIR/pi05"
if [ ! -d "$PI05_DIR" ]; then
    echo "错误: 解压后未找到 pi05 目录"
    exit 1
fi

echo "✓ 解压完成: $PI05_DIR"

# 修复虚拟环境
echo ""
echo "修复虚拟环境路径..."

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate RoboTwin

if [ $? -ne 0 ]; then
    echo "错误: 无法激活 RoboTwin 环境"
    exit 1
fi

cd "$PI05_DIR"

# 修复 pyvenv.cfg
if [ -f ".venv/pyvenv.cfg" ]; then
    PYTHON_HOME=$(dirname $(dirname $(which python)))
    PYTHON_VERSION=$(python --version | cut -d' ' -f2)
    echo "更新 pyvenv.cfg..."
    cat > .venv/pyvenv.cfg << EOF
home = $PYTHON_HOME
include-system-site-packages = false
version = $PYTHON_VERSION
EOF
fi

# 修复 Python 可执行文件链接
echo "修复 Python 可执行文件链接..."
PYTHON_BIN=$(which python)
PYTHON_VERSION=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)

if [ -d ".venv/lib/python${PYTHON_VERSION}" ] && [ -n "$PYTHON_BIN" ]; then
    rm -f .venv/bin/python .venv/bin/python3 .venv/bin/python${PYTHON_VERSION} 2>/dev/null
    ln -sf "$PYTHON_BIN" .venv/bin/python
    ln -sf "$PYTHON_BIN" .venv/bin/python3
    ln -sf "$PYTHON_BIN" .venv/bin/python${PYTHON_VERSION}
    echo "✓ Python 链接已修复"
else
    echo "警告: 无法修复 Python 链接，可能需要手动处理"
fi

# 修复 pyproject.toml 中的路径
echo "修复 pyproject.toml 中的路径..."
if [ -f "pyproject.toml" ]; then
    sed -i "s|path = \".*lerobot\"|path = \"$PI05_DIR/lerobot\"|g" pyproject.toml
    echo "✓ pyproject.toml 路径已修复"
fi

# 验证环境
echo ""
echo "验证环境..."
source .venv/bin/activate
python --version
python -c "import sys; print(f'Python路径: {sys.executable}')" || echo "警告: Python 导入失败"

echo ""
echo "=========================================="
echo "环境恢复完成！"
echo "=========================================="
echo "项目目录: $PI05_DIR"
echo ""
echo "使用方法:"
echo "  cd $PI05_DIR"
echo "  conda activate RoboTwin"
echo "  source .venv/bin/activate"
echo "  python -c 'import openpi; print(\"OK\")'"
echo ""







