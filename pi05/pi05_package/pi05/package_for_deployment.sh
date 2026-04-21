#!/bin/bash
# 打包脚本：将 pi05 项目和环境打包到另一台服务器
# 使用方法: bash package_for_deployment.sh [输出目录] [--with-venv]
#   --with-venv: 包含虚拟环境打包（默认不打包，因为新服务器已有 RoboTwin 环境）

set -e

# 解析参数
PACKAGE_VENV=false
OUTPUT_DIR=""

for arg in "$@"; do
    if [ "$arg" == "--with-venv" ]; then
        PACKAGE_VENV=true
    elif [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="$arg"
    fi
done

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/pi05_package}"

echo "=========================================="
echo "PI0.5 项目打包脚本"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "打包虚拟环境: $PACKAGE_VENV"
if [ "$PACKAGE_VENV" = false ]; then
    echo "  (默认跳过，因为新服务器已有 RoboTwin conda 环境)"
fi
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
PACKAGE_DIR="$OUTPUT_DIR/pi05"

echo "步骤 1/6: 创建打包目录结构..."
mkdir -p "$PACKAGE_DIR"

echo "步骤 2/6: 复制项目代码文件..."
# 复制所有代码文件，排除不需要的目录
rsync -av --progress \
    --exclude='.venv' \
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
    "$PROJECT_DIR/" "$PACKAGE_DIR/"

echo ""
echo "步骤 3/6: 打包虚拟环境 (.venv)..."
if [ "$PACKAGE_VENV" = true ]; then
    # 打包 .venv，但排除一些不必要的文件
    if [ -d "$PROJECT_DIR/.venv" ]; then
        echo "正在打包虚拟环境 (约 9.6GB，可能需要一些时间)..."
        cd "$PROJECT_DIR"
        tar -czf "$OUTPUT_DIR/venv.tar.gz" \
            --exclude='.venv/bin/python*' \
            --exclude='.venv/pyvenv.cfg' \
            --exclude='.venv/lib/python*/test' \
            --exclude='.venv/lib/python*/dist-packages/test' \
            --exclude='.venv/lib/python*/site-packages/test' \
            .venv/
        echo "虚拟环境已打包到: $OUTPUT_DIR/venv.tar.gz"
    else
        echo "警告: .venv 目录不存在，跳过虚拟环境打包"
    fi
else
    echo "跳过虚拟环境打包（新服务器将使用已有的 RoboTwin conda 环境重新创建）"
fi

echo ""
echo "步骤 4/6: 打包 RoboTwin 相关依赖..."
# 打包 curobo（如果存在）
if [ -d "$PROJECT_DIR/../../envs/curobo" ]; then
    echo "正在打包 curobo..."
    cd "$PROJECT_DIR/../../envs"
    tar -czf "$OUTPUT_DIR/curobo.tar.gz" curobo/
    echo "curobo 已打包到: $OUTPUT_DIR/curobo.tar.gz"
fi

echo ""
echo "步骤 5/6: 创建环境信息文件..."
# 保存环境信息
cat > "$OUTPUT_DIR/environment_info.txt" << EOF
打包时间: $(date)
项目路径: $PROJECT_DIR
Python 版本: $(python3 --version 2>/dev/null || echo "未知")
UV 版本: $(uv --version 2>/dev/null || echo "未安装")
Conda 环境: RoboTwin

重要路径:
- 项目目录: $PROJECT_DIR
- 虚拟环境: $PROJECT_DIR/.venv
- curobo: $PROJECT_DIR/../../envs/curobo (如果存在)

依赖管理:
- 使用 uv 管理依赖
- 配置文件: pyproject.toml, uv.lock
- 虚拟环境: .venv (使用 uv sync 创建，基于 RoboTwin conda 环境)
- 新服务器已有 RoboTwin conda 环境，将使用该环境重新创建 .venv

本地路径依赖:
- lerobot: $PROJECT_DIR/lerobot
- openpi-client: $PROJECT_DIR/packages/openpi-client
EOF

echo ""
echo "步骤 6/6: 创建恢复脚本..."
# 创建恢复脚本
cat > "$OUTPUT_DIR/restore_environment.sh" << 'RESTORE_SCRIPT'
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
RESTORE_SCRIPT

chmod +x "$OUTPUT_DIR/restore_environment.sh"

echo ""
echo "=========================================="
echo "打包完成！"
echo "=========================================="
echo "打包内容:"
echo "  - 项目代码: $PACKAGE_DIR"
if [ -f "$OUTPUT_DIR/venv.tar.gz" ]; then
    echo "  - 虚拟环境: $OUTPUT_DIR/venv.tar.gz ($(du -sh "$OUTPUT_DIR/venv.tar.gz" | cut -f1))"
fi
if [ -f "$OUTPUT_DIR/curobo.tar.gz" ]; then
    echo "  - curobo: $OUTPUT_DIR/curobo.tar.gz ($(du -sh "$OUTPUT_DIR/curobo.tar.gz" | cut -f1))"
fi
echo "  - 恢复脚本: $OUTPUT_DIR/restore_environment.sh"
echo "  - 环境信息: $OUTPUT_DIR/environment_info.txt"
echo ""
echo "总大小:"
du -sh "$OUTPUT_DIR"
echo ""
echo "传输到新服务器:"
echo "  可以使用 scp, rsync 或 tar 压缩后传输"
echo "  例如: tar -czf pi05_package.tar.gz -C $OUTPUT_DIR ."
echo ""
echo "注意: 默认不包含虚拟环境（新服务器已有 RoboTwin conda 环境）"
echo "如需包含虚拟环境，使用: bash package_for_deployment.sh --with-venv"
echo ""

