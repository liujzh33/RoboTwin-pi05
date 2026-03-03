#!/bin/bash
# 在 openpi 环境中安装 curobo，解决 eval 时 ModuleNotFoundError: No module named 'curobo'
# 用法：在 policy/pi05 目录下执行 bash install_curobo.sh

set -e
# 若当前在 conda 环境，先退出
conda deactivate 2>/dev/null || true

# 激活 openpi 的 venv（与 eval.sh 使用同一环境）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

echo "Using Python: $(which python)"
echo "Installing curobo into current env..."

# 进入 envs，使用已有 curobo 仓库（若没有则克隆）
ROBOTWIN_ROOT="$(cd ../.. && pwd)"
cd "$ROBOTWIN_ROOT/envs"
if [ ! -d "curobo" ]; then
    echo "Cloning curobo..."
    git clone https://github.com/NVlabs/curobo.git
fi
cd curobo

# 可编辑安装（构建可能需几分钟）
pip install -e . --no-build-isolation

echo "Done. Back to policy/pi05."
cd "$SCRIPT_DIR"
