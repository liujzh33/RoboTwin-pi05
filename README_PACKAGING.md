# pi05 项目打包和使用说明

## 打包步骤

在源服务器上执行：

```bash
cd /mnt/data/guoxiaoyu/RoboTwin/policy/pi05
chmod +x pack_pi05.sh
./pack_pi05.sh
```

打包脚本会：
- 排除 `checkpoints/` (250G)
- 排除 `processed_data/` (128G)  
- 排除 `training_data/` (128G)
- 包含 `.venv/` (9.6G)
- 包含所有其他项目文件

压缩包会生成在 `/mnt/data/guoxiaoyu/RoboTwin/policy/` 目录下。

## 传输到目标服务器

```bash
# 使用 scp 传输（示例）
scp /mnt/data/guoxiaoyu/RoboTwin/policy/pi05_packaged_*.tar.gz user@target-server:/path/to/destination/

# 或使用 rsync（推荐，支持断点续传）
rsync -avzP /mnt/data/guoxiaoyu/RoboTwin/policy/pi05_packaged_*.tar.gz user@target-server:/path/to/destination/
```

## 在目标服务器上解压和修复

```bash
# 1. 解压
cd /path/to/destination
tar -xzf pi05_packaged_*.tar.gz

# 2. 进入解压后的目录
cd pi05

# 3. 运行路径修复脚本
chmod +x fix_venv_paths.sh
./fix_venv_paths.sh

# 4. 激活虚拟环境测试
source .venv/bin/activate

# 5. 验证环境
python --version
pip --version
pip list | head -20
```

## 注意事项

1. **Python 版本要求**: 目标服务器需要 Python >= 3.11（与源服务器相同或兼容的版本）

2. **系统依赖**: 某些包（如 cv2, sapien 等）可能需要系统级的依赖库，如果在新服务器上运行出错，可能需要：
   ```bash
   # Ubuntu/Debian 示例
   sudo apt-get update
   sudo apt-get install -y python3-dev build-essential
   # 根据具体错误信息安装其他依赖
   ```

3. **CUDA/GPU**: 如果项目使用 GPU，确保目标服务器有相应的 CUDA 环境

4. **路径修复**: `fix_venv_paths.sh` 会自动更新：
   - `pyvenv.cfg` 中的 Python home 路径
   - `bin/` 目录下脚本的 shebang
   - `activate` 脚本中的虚拟环境路径
   - 清理 Python 缓存

5. **如果路径修复后仍有问题**，可以尝试重新安装关键包：
   ```bash
   source .venv/bin/activate
   pip install --force-reinstall --no-deps <problematic-package>
   ```

