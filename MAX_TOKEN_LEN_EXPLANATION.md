# max_token_len 说明

## 什么是 max_token_len？

`max_token_len` 是模型配置中的一个参数，用于限制**语言 prompt（文本指令）**的最大 token 长度。

### 作用范围
- ✅ **限制**：文本 prompt 的 token 长度
- ❌ **不限制**：图像、状态、动作等其他输入

### 默认值
- **pi05 模型**：默认 `max_token_len = 200`
- **pi0 模型**：默认 `max_token_len = 48`

## 为什么会出现警告？

当数据集中某些 prompt 的 token 长度超过 `max_token_len` 时，系统会：
1. **自动截断**：只保留前 `max_token_len` 个 tokens
2. **发出警告**：提示用户考虑增加 `max_token_len`

### 示例警告
```
WARNING:root:Token length (205) exceeds max length (200), truncating.
WARNING:root:Token length (250) exceeds max length (200), truncating.
```

这表示：
- 某些 prompt 有 205 个 tokens，被截断到 200
- 某些 prompt 有 250 个 tokens，被截断到 200

## 是否需要增加？

**建议增加**，原因：

1. **信息丢失**：截断会丢失 prompt 的部分信息，可能影响模型性能
2. **频繁警告**：如果警告频繁出现，说明很多 prompt 被截断
3. **实际需求**：从警告看，prompt 长度在 205-250 之间，建议至少设为 256 或 300

## 如何修改？

### 方法 1：在训练配置中修改（推荐）

在 `config.py` 中的 `TrainConfig` 里设置：

```python
TrainConfig(
    name="pi05_aloha_full_base_multiframe_beat",
    model=pi0_config.Pi0Config(
        pi05=True, 
        n_obs_steps=4, 
        max_token_len=300  # 增加到 300
    ),
    # ... 其他配置
)
```

### 方法 2：通过命令行参数覆盖

如果使用 `tyro` CLI，可以通过命令行参数覆盖：

```bash
python scripts/train.py \
    --config-name pi05_aloha_full_base_multiframe_beat \
    --model.max_token_len 300
```

## 内存影响

增加 `max_token_len` 会增加内存使用，因为：
- Token 序列长度增加
- Transformer 的注意力矩阵大小增加（O(n²)）

**建议值**：
- **256**：适合大多数情况，内存开销较小
- **300**：适合较长的 prompt，内存开销中等
- **400+**：适合非常长的 prompt，但内存开销较大

## 当前配置

你的配置 `pi05_aloha_full_base_multiframe_beat` 和 `pi05_aloha_full_base_multiframe_blocks` 已经更新为 `max_token_len=300`。

## 注意事项

1. **需要重新训练**：如果修改了 `max_token_len`，需要重新训练模型（因为模型架构会改变）
2. **检查数据集**：如果 prompt 都很短，不需要增加 `max_token_len`
3. **平衡考虑**：在内存和性能之间找到平衡点

## 总结

- ✅ **这是警告，不是错误**：训练可以继续，但 prompt 会被截断
- ✅ **建议增加**：从 200 增加到 256 或 300
- ✅ **已更新配置**：两个多帧配置已更新为 `max_token_len=300`
