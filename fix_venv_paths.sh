#!/bin/bash
# 修复 .venv 中的硬编码路径
# 用于在新服务器上解压后运行

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$CURRENT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "错误: 找不到 .venv 目录: $VENV_DIR"
    exit 1
fi

OLD_PYTHON_PATH=""
NEW_PYTHON_PATH="$VENV_DIR/bin/python3"
PYTHON_VERSION=$(python3 --version | grep -oP '\d+\.\d+' | head -1)

echo "=========================================="
echo "修复 .venv 中的路径"
echo "=========================================="
echo "当前目录: $CURRENT_DIR"
echo ".venv 目录: $VENV_DIR"
echo "Python 版本: $PYTHON_VERSION"
echo ""

# 查找旧的 Python 路径
if [ -f "$VENV_DIR/pyvenv.cfg" ]; then
    OLD_PYTHON_PATH=$(grep "^home = " "$VENV_DIR/pyvenv.cfg" | cut -d' ' -f3)
    if [ -n "$OLD_PYTHON_PATH" ]; then
        echo "发现旧的 Python 路径: $OLD_PYTHON_PATH"
    fi
fi

# 更新 pyvenv.cfg
echo "1. 更新 pyvenv.cfg..."
if [ -f "$VENV_DIR/pyvenv.cfg" ]; then
    # 获取系统 Python 路径
    SYSTEM_PYTHON=$(which python3)
    if [ -n "$SYSTEM_PYTHON" ]; then
        SYSTEM_PYTHON_DIR=$(dirname $(dirname "$SYSTEM_PYTHON"))
        sed -i "s|^home = .*|home = $SYSTEM_PYTHON_DIR|" "$VENV_DIR/pyvenv.cfg"
        echo "   ✅ 已更新 Python home 路径为: $SYSTEM_PYTHON_DIR"
    fi
fi

# 更新所有脚本文件中的 shebang
echo "2. 更新脚本文件中的 shebang..."
find "$VENV_DIR/bin" -type f -executable ! -name "*.so" 2>/dev/null | while read script; do
    if head -1 "$script" 2>/dev/null | grep -q "^#!"; then
        # 检查是否包含旧的路径
        if grep -q "$OLD_PYTHON_PATH" "$script" 2>/dev/null; then
            # 获取脚本第一行的解释器
            INTERPRETER=$(head -1 "$script" | sed 's|^#!||' | awk '{print $1}')
            if [ -n "$INTERPRETER" ]; then
                NEW_INTERPRETER="$VENV_DIR/bin/$(basename "$INTERPRETER")"
                if [ -f "$NEW_INTERPRETER" ]; then
                    sed -i "1s|^#!.*|#!$NEW_INTERPRETER|" "$script"
                fi
            fi
        fi
    fi
done
echo "   ✅ 已更新 bin/ 目录下的脚本"

# 更新 .venv/bin/activate 中的路径
echo "3. 更新 activate 脚本..."
for activate_file in "$VENV_DIR/bin/activate" "$VENV_DIR/bin/activate.csh" "$VENV_DIR/bin/activate.fish"; do
    if [ -f "$activate_file" ]; then
        # 更新 VIRTUAL_ENV 变量的设置
        if [ -n "$OLD_PYTHON_PATH" ]; then
            sed -i "s|$OLD_PYTHON_PATH|$VENV_DIR|g" "$activate_file" 2>/dev/null
        fi
        sed -i "s|VIRTUAL_ENV=.*|VIRTUAL_ENV=\"$VENV_DIR\"|g" "$activate_file" 2>/dev/null
    fi
done
echo "   ✅ 已更新 activate 脚本"

# 更新 pip 和其他工具的路径
echo "4. 检查并更新 pip 相关路径..."
if [ -f "$VENV_DIR/bin/pip" ]; then
    # 尝试重新生成 pip
    "$VENV_DIR/bin/python3" -m pip install --upgrade pip --quiet 2>/dev/null
    echo "   ✅ 已更新 pip"
fi

# 更新所有 .pyc 文件的路径引用（需要重新编译）
echo "5. 清理 Python 缓存..."
find "$VENV_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find "$VENV_DIR" -name "*.pyc" -delete 2>/dev/null
find "$VENV_DIR" -name "*.pyo" -delete 2>/dev/null
echo "   ✅ 已清理 Python 缓存"

echo ""
echo "=========================================="
echo "✅ 路径修复完成！"
echo "=========================================="
echo ""
echo "测试激活虚拟环境:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "测试 Python 和 pip:"
echo "  python --version"
echo "  pip --version"
echo ""

