# 快速打包指南

## 🚀 一键打包

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
bash package_for_deployment.sh
```

打包完成后，会在当前目录生成 `pi05_package/` 目录。

**注意**: 默认**不打包虚拟环境**（因为新服务器已有 RoboTwin conda 环境），如果需要打包虚拟环境，使用：
```bash
bash package_for_deployment.sh --with-venv
```

## 📦 打包内容

- ✅ 项目代码（排除缓存、数据等）
- ⏭️ 虚拟环境 `.venv` (默认跳过，新服务器将使用 RoboTwin conda 环境重新创建)
- ✅ curobo 依赖（如果存在）
- ✅ 恢复脚本和说明文档

## 📤 传输到新服务器

### 方法 1: 压缩后传输（推荐）

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
tar -czf pi05_package.tar.gz -C pi05_package .

# 传输
scp pi05_package.tar.gz user@new_server:/path/to/destination/

# 在新服务器解压
ssh user@new_server
cd /path/to/destination
tar -xzf pi05_package.tar.gz
```

### 方法 2: 直接传输目录

```bash
scp -r pi05_package/ user@new_server:/path/to/destination/
```

## 🔧 在新服务器恢复

### 前置条件

```bash
# 1. 创建 conda 环境
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin

# 2. 安装 uv
pip install uv
```

### 恢复环境

```bash
cd /path/to/pi05_package
bash restore_environment.sh
```

默认安装到 `/mnt/data1/liujingzhi/RoboTwin/policy/pi05`

### 验证

```bash
conda activate RoboTwin
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
source .venv/bin/activate
python -c "import openpi; print('OK')"
```

## ⚠️ 注意事项

1. **虚拟环境默认不打包**：
   - 新服务器已有 RoboTwin conda 环境，恢复脚本会自动使用该环境重新创建 `.venv`
   - 这样可以大大减少传输大小（从 ~10GB 减少到 ~100MB）
   - 如果需要打包虚拟环境，使用 `--with-venv` 选项

2. **路径依赖**：
   - `lerobot` 和 `packages/openpi-client` 会一起打包
   - 恢复脚本会自动修复路径

3. **可选数据**：
   - `checkpoints/` (126GB) - 模型权重，根据需要单独传输
   - `processed_data/` (3.4GB) - 处理后的数据，可选

## 📋 完整文档

详细说明请查看 `PACKAGE_GUIDE.md`

