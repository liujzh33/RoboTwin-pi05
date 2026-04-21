#!/bin/bash
# 环境恢复脚本：在新服务器上恢复 pi05 环境
# 使用方法: bash restore_environment.sh [项目安装路径]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$SCRIPT_DIR/pi05"
INSTALL_DIR="${1:-/mnt/data1/liujingzhi/RoboTwin/policy/pi05}"

echo "=========================================="
echo "PI0.5 环境恢复脚本"
echo "=========================================="
echo "安装目录: $INSTALL_DIR"
echo ""

# 检查必要工具
echo "检查必要工具..."
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 conda"
    exit 1
fi

# 检查 RoboTwin conda 环境
echo "检查 RoboTwin conda 环境..."
if conda env list | grep -q "^RoboTwin "; then
    echo "✓ 找到 RoboTwin conda 环境"
    ROBOTWIN_ENV_PATH=$(conda env list | grep "^RoboTwin " | awk '{print $NF}')
    echo "  环境路径: $ROBOTWIN_ENV_PATH"
else
    echo "警告: 未找到 RoboTwin conda 环境"
    echo "请先创建: conda create -n RoboTwin python=3.10 -y"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if ! command -v uv &> /dev/null; then
    echo "警告: 未找到 uv，将尝试安装..."
    pip install uv
fi

# 创建安装目录
echo "创建安装目录..."
mkdir -p "$INSTALL_DIR"
mkdir -p "$(dirname "$INSTALL_DIR")/../envs"

# 复制项目文件
echo "复制项目文件..."
cp -r "$PACKAGE_DIR"/* "$INSTALL_DIR/"

# 恢复虚拟环境
if [ -f "$SCRIPT_DIR/venv.tar.gz" ]; then
    echo "恢复虚拟环境（从打包文件）..."
    cd "$INSTALL_DIR"
    tar -xzf "$SCRIPT_DIR/venv.tar.gz"
    
    # 修复虚拟环境中的路径
    echo "修复虚拟环境路径..."
    # 激活 conda 环境
    eval "$(conda shell.bash hook)"
    conda activate RoboTwin
    
    if [ -f ".venv/pyvenv.cfg" ]; then
        # 重新创建 pyvenv.cfg
        PYTHON_HOME=$(dirname $(dirname $(which python)))
        PYTHON_VERSION=$(python --version | cut -d' ' -f2)
        cat > .venv/pyvenv.cfg << EOF
home = $PYTHON_HOME
include-system-site-packages = false
version = $PYTHON_VERSION
EOF
    fi
    
    # 重新创建 Python 可执行文件的符号链接
    PYTHON_VERSION=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ -n "$PYTHON_VERSION" ] && [ -d ".venv/lib/python${PYTHON_VERSION}" ]; then
        PYTHON_BIN=$(which python)
        if [ -n "$PYTHON_BIN" ]; then
            rm -f .venv/bin/python .venv/bin/python3 .venv/bin/python${PYTHON_VERSION} 2>/dev/null
            ln -sf "$PYTHON_BIN" .venv/bin/python
            ln -sf "$PYTHON_BIN" .venv/bin/python3
            ln -sf "$PYTHON_BIN" .venv/bin/python${PYTHON_VERSION}
        fi
    fi
else
    echo "未找到 venv.tar.gz，将使用 RoboTwin conda 环境重新创建虚拟环境..."
    cd "$INSTALL_DIR"
    
    # 激活 conda 环境
    echo "激活 RoboTwin conda 环境..."
    eval "$(conda shell.bash hook)"
    conda activate RoboTwin
    
    if [ $? -ne 0 ]; then
        echo "错误: 无法激活 RoboTwin 环境"
        echo "请确保已创建并激活: conda activate RoboTwin"
        exit 1
    fi
    
    echo "当前 Python: $(which python)"
    echo "Python 版本: $(python --version)"
    
    # 使用 uv 重新创建环境
    echo "使用 uv 重新创建虚拟环境..."
    if ! command -v uv &> /dev/null; then
        echo "安装 uv..."
        pip install uv
    fi
    
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    
    if [ $? -ne 0 ]; then
        echo "警告: uv sync 失败，尝试使用 pip 安装..."
        pip install -e . --no-build-isolation
    fi
fi

# 恢复 curobo
if [ -f "$SCRIPT_DIR/curobo.tar.gz" ]; then
    echo "恢复 curobo..."
    cd "$(dirname "$INSTALL_DIR")/../envs"
    tar -xzf "$SCRIPT_DIR/curobo.tar.gz"
    cd curobo
    echo "重新安装 curobo..."
    # 确保在 conda 环境中
    eval "$(conda shell.bash hook)"
    conda activate RoboTwin
    pip install -e . --no-build-isolation
fi

# 修复 pyproject.toml 中的路径
echo "修复 pyproject.toml 中的路径..."
cd "$INSTALL_DIR"
if [ -f "pyproject.toml" ]; then
    sed -i "s|path = \".*lerobot\"|path = \"$INSTALL_DIR/lerobot\"|g" pyproject.toml
fi

# 重新安装项目（确保路径正确）
echo "重新安装项目依赖..."
cd "$INSTALL_DIR"

# 确保在 conda 环境中
eval "$(conda shell.bash hook)"
conda activate RoboTwin

# 尝试激活虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "虚拟环境已激活"
    echo "当前 Python: $(which python)"
    
    # 使用 uv sync 或 pip install
    if command -v uv &> /dev/null; then
        echo "使用 uv sync 同步依赖..."
        uv sync || pip install -e . --no-build-isolation
    else
        echo "使用 pip 安装..."
        pip install -e . --no-build-isolation
    fi
else
    echo "警告: 虚拟环境未找到，请手动创建"
    echo "运行: cd $INSTALL_DIR && conda activate RoboTwin && uv sync"
fi

echo ""
echo "=========================================="
echo "环境恢复完成！"
echo "=========================================="
echo "项目已安装到: $INSTALL_DIR"
echo ""
echo "下一步:"
echo "1. 激活 conda 环境: conda activate RoboTwin"
echo "2. 进入项目目录: cd $INSTALL_DIR"
echo "3. 激活虚拟环境: source .venv/bin/activate"
echo "4. 验证安装: python -c 'import openpi; print(\"OK\")'"
echo ""
echo "注意: 如果虚拟环境不存在，运行:"
echo "  cd $INSTALL_DIR"
echo "  conda activate RoboTwin"
echo "  uv sync"
echo ""
