# PI0.5 项目打包和部署指南

本文档说明如何将 pi05 项目及其环境打包到另一台服务器上，实现无缝使用。

## 📦 需要打包的内容

### 1. 项目代码文件
- 所有源代码文件 (`src/`, `scripts/`, `examples/` 等)
- 配置文件 (`pyproject.toml`, `uv.lock`, `.gitignore` 等)
- 文档文件 (`*.md`)
- `lerobot/` 目录（本地路径依赖）
- `packages/openpi-client/` 目录（workspace 成员）

### 2. 虚拟环境 (`.venv`)
- 使用 uv 创建的虚拟环境（约 9.6GB）
- 包含所有已安装的 Python 包

### 3. RoboTwin 相关依赖
- `envs/curobo/` 目录（如果存在，约 301MB）

### 4. 可选数据文件（根据需要）
- `checkpoints/` - 模型检查点（约 126GB，可选）
- `processed_data/` - 处理后的数据（约 3.4GB，可选）
- `training_data/` - 训练数据（可选）

## 🚀 打包步骤

### 方法 1: 使用自动打包脚本（推荐）

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
bash package_for_deployment.sh [输出目录]
```

默认输出目录为 `./pi05_package/`

### 方法 2: 手动打包

#### 2.1 打包项目代码

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
tar -czf pi05_code.tar.gz \
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
    .
```

#### 2.2 打包虚拟环境

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
tar -czf venv.tar.gz \
    --exclude='.venv/bin/python*' \
    --exclude='.venv/pyvenv.cfg' \
    .venv/
```

#### 2.3 打包 curobo（如果存在）

```bash
cd /mnt/data1/liujingzhi/RoboTwin
tar -czf curobo.tar.gz -C envs curobo/
```

## 📤 传输到新服务器

### 方法 1: 使用 scp

```bash
# 传输整个打包目录
scp -r pi05_package/ user@new_server:/path/to/destination/
```

### 方法 2: 先压缩再传输（推荐，节省带宽）

```bash
# 压缩打包目录
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
tar -czf pi05_package.tar.gz -C pi05_package .

# 传输压缩包
scp pi05_package.tar.gz user@new_server:/path/to/destination/

# 在新服务器上解压
ssh user@new_server
cd /path/to/destination
tar -xzf pi05_package.tar.gz
```

### 方法 3: 使用 rsync（适合大文件，支持断点续传）

```bash
rsync -avz --progress pi05_package/ user@new_server:/path/to/destination/pi05_package/
```

## 🔧 在新服务器上恢复环境

### 前置要求

1. **安装 conda**（如果未安装）
   ```bash
   # 下载并安装 miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. **创建 RoboTwin conda 环境**
   ```bash
   conda create -n RoboTwin python=3.10 -y
   conda activate RoboTwin
   ```

3. **安装 uv**（如果未安装）
   ```bash
   pip install uv
   ```

### 恢复步骤

#### 方法 1: 使用自动恢复脚本（推荐）

```bash
cd /path/to/pi05_package
bash restore_environment.sh [安装路径]
```

默认安装路径为 `/mnt/data1/liujingzhi/RoboTwin/policy/pi05`

#### 方法 2: 手动恢复

##### 2.1 解压项目代码

```bash
cd /path/to/destination
tar -xzf pi05_code.tar.gz -C /mnt/data1/liujingzhi/RoboTwin/policy/pi05
```

##### 2.2 恢复虚拟环境

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
tar -xzf venv.tar.gz

# 修复虚拟环境路径
conda activate RoboTwin
PYTHON_VERSION=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
PYTHON_BIN=$(which python)
ln -sf "$PYTHON_BIN" .venv/bin/python
ln -sf "$PYTHON_BIN" .venv/bin/python3
ln -sf "$PYTHON_BIN" .venv/bin/python${PYTHON_VERSION}

# 重新创建 pyvenv.cfg
cat > .venv/pyvenv.cfg << EOF
home = $(dirname $(dirname $(which python)))
include-system-site-packages = false
version = $(python --version | cut -d' ' -f2)
EOF
```

##### 2.3 恢复 curobo

```bash
cd /mnt/data1/liujingzhi/RoboTwin
mkdir -p envs
tar -xzf curobo.tar.gz -C envs/
cd envs/curobo
conda activate RoboTwin
pip install -e . --no-build-isolation
```

##### 2.4 修复路径依赖

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05

# 修复 pyproject.toml 中的 lerobot 路径
sed -i "s|path = \".*lerobot\"|path = \"$(pwd)/lerobot\"|g" pyproject.toml

# 重新安装项目
conda activate RoboTwin
source .venv/bin/activate
uv sync
# 或者
pip install -e . --no-build-isolation
```

## ✅ 验证安装

```bash
# 激活环境
conda activate RoboTwin
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
source .venv/bin/activate

# 验证 Python 包
python -c "import openpi; print('openpi OK')"
python -c "import lerobot; print('lerobot OK')"
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import jax; print(f'jax: {jax.__version__}')"

# 运行简单测试
python examples/simple_client/main.py
```

## 🔍 故障排查

### 问题 1: 虚拟环境中的 Python 路径错误

**症状**: `bad interpreter: No such file or directory`

**解决**:
```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
conda activate RoboTwin
PYTHON_BIN=$(which python)
rm .venv/bin/python*
ln -sf "$PYTHON_BIN" .venv/bin/python
ln -sf "$PYTHON_BIN" .venv/bin/python3
```

### 问题 2: 找不到本地路径依赖

**症状**: `PackageNotFoundError` 或 `ModuleNotFoundError`

**解决**:
```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
# 检查 pyproject.toml 中的路径是否正确
cat pyproject.toml | grep -A 2 "tool.uv.sources"

# 修复路径
sed -i "s|path = \".*lerobot\"|path = \"$(pwd)/lerobot\"|g" pyproject.toml

# 重新安装
uv sync
```

### 问题 3: CUDA 相关错误

**症状**: CUDA 版本不匹配

**解决**:
- 检查新服务器的 CUDA 版本: `nvidia-smi`
- 可能需要重新安装 PyTorch 和 JAX（CUDA 版本）
- 参考 `pyproject.toml` 中的 CUDA 版本要求

### 问题 4: 如果虚拟环境恢复失败，重新创建

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
conda activate RoboTwin

# 删除旧的虚拟环境
rm -rf .venv

# 使用 uv 重新创建
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

## 📋 打包清单检查

在打包前，确认以下文件/目录已包含：

- [x] `pyproject.toml` - 项目配置
- [x] `uv.lock` - 依赖锁定文件
- [x] `src/` - 源代码
- [x] `scripts/` - 脚本文件
- [x] `examples/` - 示例代码
- [x] `lerobot/` - 本地依赖
- [x] `packages/openpi-client/` - workspace 成员
- [x] `.venv/` - 虚拟环境（或 venv.tar.gz）
- [x] `../../envs/curobo/` - curobo（或 curobo.tar.gz）
- [ ] `checkpoints/` - 模型检查点（可选，很大）
- [ ] `processed_data/` - 处理后的数据（可选）
- [ ] `training_data/` - 训练数据（可选）

## 💡 优化建议

1. **如果网络带宽有限**:
   - 只打包代码和配置文件
   - 在新服务器上使用 `uv sync` 重新创建虚拟环境
   - 这样可以减少传输大小（从 ~10GB 减少到 ~100MB）

2. **如果时间有限**:
   - 打包整个 `.venv`，虽然大但恢复快
   - 使用 rsync 支持断点续传

3. **如果存储空间有限**:
   - 排除 `checkpoints/` 和 `processed_data/`
   - 这些可以后续单独传输或重新生成

## 📞 需要帮助？

如果遇到问题，请检查：
1. `environment_info.txt` - 查看原始环境信息
2. `restore_environment.sh` - 查看恢复脚本的详细步骤
3. 项目文档 `QUICK_START.md` 和 `ARCHITECTURE.md`

