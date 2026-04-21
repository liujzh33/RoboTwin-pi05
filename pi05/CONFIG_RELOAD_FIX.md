# 配置更新后需要重新启动训练

## 问题

配置已经更新为 `max_token_len=300`，但训练时仍然显示 `max length (200)`。

## 原因

**Python 模块缓存**：如果训练进程在配置修改之前就已经启动，它会使用已经加载到内存中的旧配置。

## 解决方案

### 1. 停止当前训练进程

如果训练正在运行，需要：
```bash
# 找到训练进程
ps aux | grep train.py

# 停止进程（使用进程ID）
kill <PID>
```

### 2. 清除 Python 缓存（可选但推荐）

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete
```

### 3. 重新启动训练

```bash
# 确保使用正确的配置名称
python scripts/train.py --config-name pi05_aloha_full_base_multiframe_beat

# 或者 blocks 配置
python scripts/train.py --config-name pi05_aloha_full_base_multiframe_blocks
```

### 4. 验证配置是否正确加载

在训练开始时，检查日志输出，应该能看到：
- 模型配置信息
- `max_token_len=300` 应该出现在配置中

或者添加临时调试代码来验证：

```python
# 在 train.py 的 main 函数开始处添加
print(f"max_token_len: {config.model.max_token_len}")
```

## 当前配置状态

✅ **已更新的配置**：
- `pi05_aloha_full_base_multiframe_beat`: `max_token_len=300`
- `pi05_aloha_full_base_multiframe_blocks`: `max_token_len=300`

## 验证方法

### 方法 1：检查配置文件

```bash
grep -A 2 "pi05_aloha_full_base_multiframe_beat" src/openpi/training/config.py | grep max_token_len
```

应该看到：`max_token_len=300`

### 方法 2：通过 Python 直接检查

```python
from openpi.training.config import get_config

config = get_config("pi05_aloha_full_base_multiframe_beat")
print(f"max_token_len: {config.model.max_token_len}")
# 应该输出: max_token_len: 300
```

### 方法 3：运行时检查

训练启动后，警告应该显示：
```
WARNING:root:Token length (205) exceeds max length (300), truncating.
```

如果还是显示 `max length (200)`，说明：
1. ❌ 训练进程使用的是旧配置（需要重启）
2. ❌ 或者使用了不同的配置名称

## 常见问题

### Q: 为什么修改配置后还是显示旧值？

A: Python 会缓存已导入的模块。如果训练进程在配置修改前启动，它会使用内存中的旧配置。

### Q: 如何确认使用的是哪个配置？

A: 检查训练命令中的 `--config-name` 参数，确保使用的是正确的配置名称。

### Q: 可以通过命令行覆盖配置吗？

A: 可以！使用 tyro 的参数覆盖：

```bash
python scripts/train.py \
    --config-name pi05_aloha_full_base_multiframe_beat \
    --model.max_token_len 300
```

## 总结

1. ✅ **配置已更新**：两个多帧配置都已设置为 `max_token_len=300`
2. ⚠️ **需要重启**：如果训练正在运行，需要停止并重新启动
3. ✅ **验证方法**：检查训练日志，应该显示 `max length (300)` 而不是 `max length (200)`
