# 为什么现在会出现截断警告，而之前单帧训练时没有？

## 重要结论

**prompt 截断与单帧/多帧无关**。prompt 是文本指令，与图像帧数没有关系。

## 配置对比

### 单帧配置 (`pi05_aloha_full_base`)
```python
model=pi0_config.Pi0Config(pi05=True)  # max_token_len 默认 200
data=LeRobotAlohaDataConfig(
    repo_id="beat_hammer_pi05_200",
    prompt_from_task=True,  # 从任务名称生成 prompt
)
```

### 多帧配置 (`pi05_aloha_full_base_multiframe_beat`)
```python
model=pi0_config.Pi0Config(pi05=True, n_obs_steps=4)  # max_token_len 默认 200
data=LeRobotAlohaDataConfig(
    repo_id="beat_hammer_pi05_200",  # 相同的数据集
    prompt_from_task=True,  # 相同的 prompt 来源
)
```

**关键发现**：
- ✅ 使用**相同的数据集** (`beat_hammer_pi05_200`)
- ✅ 使用**相同的 prompt 来源** (`prompt_from_task=True`)
- ✅ 使用**相同的 max_token_len** (默认 200)
- ❌ **唯一的区别**：`n_obs_steps` (1 vs 4)

## 为什么之前没有出现警告？

### 可能原因 1：之前没有注意到（最可能）

这些是 **WARNING**，不是 **ERROR**：
- ✅ 训练会**正常继续**
- ✅ 只是 prompt 会被**自动截断**
- ⚠️ 警告可能被其他日志信息淹没
- ⚠️ 可能之前训练时日志级别不同，没有显示警告

### 可能原因 2：数据集中的 prompt 长度分布

即使使用相同的数据集，不同训练批次可能采样到不同的样本：
- 如果之前训练时**恰好采样到的都是短 prompt**，就不会有警告
- 如果这次训练时**采样到了更多长 prompt**，就会出现警告

### 可能原因 3：训练规模不同

- **单帧配置**：`batch_size=64`, `num_train_steps=3001`
- **多帧配置**：`batch_size=128`, `num_train_steps=30001`

多帧训练：
- Batch size 更大（128 vs 64）
- 训练步数更多（30001 vs 3001）
- **遇到长 prompt 样本的概率更高**

### 可能原因 4：日志输出更详细

可能这次训练时：
- 日志级别设置为显示 WARNING
- 或者使用了不同的日志配置
- 所以能看到这些警告

## 验证方法

如果想确认之前单帧训练是否也有这个问题，可以：

1. **检查之前的训练日志**：
   ```bash
   grep -i "token length.*exceeds" <之前的训练日志文件>
   ```

2. **重新运行单帧训练**（使用相同的配置）：
   ```bash
   # 使用单帧配置重新训练一小段时间
   # 观察是否出现相同的警告
   ```

## 结论

**prompt 截断与单帧/多帧无关**，因为：
- Prompt 是文本指令，与图像帧数无关
- 单帧和多帧配置使用相同的数据集和 prompt 来源
- 唯一的区别是 `n_obs_steps`，但这不影响 prompt 长度

**最可能的原因**：
1. 之前训练时**没有注意到**这些警告（警告不是错误，训练会继续）
2. 这次训练时**采样到了更多长 prompt 的样本**
3. 或者**日志配置不同**，现在能看到警告了

## 解决方案

无论原因如何，**建议增加 `max_token_len`**：
- 从警告看，prompt 长度在 205-250 之间
- 建议设置为 256 或 300
- 已经更新配置为 `max_token_len=300`

这样可以：
- ✅ 避免信息丢失
- ✅ 减少警告
- ✅ 提高模型性能（不会截断 prompt）

## 总结

| 项目 | 单帧训练 | 多帧训练 | 是否相关 |
|------|---------|---------|---------|
| Prompt 来源 | `prompt_from_task=True` | `prompt_from_task=True` | ✅ 相同 |
| 数据集 | `beat_hammer_pi05_200` | `beat_hammer_pi05_200` | ✅ 相同 |
| max_token_len | 200 (默认) | 200 (默认) | ✅ 相同 |
| 图像帧数 | 1 | 4 | ❌ **无关** |
| 出现警告 | 可能也有，但没注意到 | 现在看到了 | - |

**结论**：prompt 截断与单帧/多帧无关，可能是之前没有注意到，或者这次采样到了更多长 prompt 的样本。
