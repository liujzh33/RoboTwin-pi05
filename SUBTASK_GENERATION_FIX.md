# Pi0.5 子任务生成修复说明

## 问题诊断

### 症状
使用相同的权重，之前运行 `bash eval.sh beat_block_hammer demo_randomized pi05_aloha_full_base demo_randomized_beat_block_hammer 0 7 26000` 时会输出子任务（如 "raise the arm."），但现在没有子任务输出了。

### 根本原因
你的项目代码在**训练时**已经实现了子任务生成的损失函数（`subtask_generation_loss`），但在**推理时**缺少了 `sample_low_level_task()` 方法来实际生成子任务。

对比参考实现 `/mnt/data1/liujingzhi/openpi_subtask_generation-main/src/openpi/models/pi05.py`：
- ✅ 训练时：通过 `compute_loss()` 计算子任务生成损失
- ✅ 推理时：通过 `sample_low_level_task()` 自回归生成子任务，然后在 `sample_actions()` 中使用生成的子任务的 KV cache

你的原始代码：
- ✅ 训练时：有 `compute_loss()` 计算子任务生成损失
- ❌ 推理时：`sample_actions()` 直接使用 prefix tokens，**没有调用子任务生成**

## 修复内容

### 文件：`/mnt/data1/liujingzhi/RoboTwin/policy/pi05/src/openpi/models/pi05.py`

#### 1. 添加辅助函数（第33-60行）
```python
@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # 用于子任务生成时对齐输入序列
    ...

def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    # 用于在指定位置插入生成的token
    ...
```

#### 2. 添加 `sample_low_level_task()` 方法（第333-438行）
新增完整的子任务生成方法，包括：
- 自回归token生成（使用贪婪解码或温度采样）
- KV cache管理
- EOS token早停机制
- 生成的子任务文本日志输出

```python
@override
def sample_low_level_task(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    max_decoding_steps: int = 20,
    PALIGEMMA_EOS_TOKEN: int = 1,
    temperature: float = 0.0,
) -> tuple[...]:
    """
    Generate low-level subtask tokens autoregressively.
    
    Returns:
        output_tokens: Generated subtask tokens [batch, max_decoding_steps]
        kv_cache: KV cache for subsequent action generation
        mask: Token mask [batch, prefix_len + max_decoding_steps]
        ar_mask: Autoregressive mask [prefix_len + max_decoding_steps]
    """
    # 实现自回归生成逻辑
    ...
```

**关键特性：**
- 使用 `left_to_right_align` 对齐输入序列
- 使用 KV cache 加速生成
- 每生成一个token都会打印日志：`logger.info(f"[Pi0.5 Subtask]: {subtask_text}")`
- 支持温度采样（temperature > 0）或贪婪解码（temperature = 0）

#### 3. 修改 `sample_actions()` 方法（第440-505行）
**关键改动：**

**之前（错误）：**
```python
# 直接使用 prefix tokens
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
positions = jnp.cumsum(prefix_mask, axis=1) - 1
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)
```

**现在（正确）：**
```python
# 先生成子任务，获取包含子任务的 KV cache
output_tokens, kv_cache, prefix_mask, prefix_ar_mask = self.sample_low_level_task(
    rng, observation, max_decoding_steps=20, PALIGEMMA_EOS_TOKEN=1, temperature=0.0
)
```

这样在后续的 flow matching 步骤中，模型能够基于生成的子任务（存储在 KV cache 中）来预测动作。

## 预期效果

修复后，运行 eval 时应该能看到类似的输出：

```
[Pi0.5 Subtask]: raise the arm.
----------------------------------------------------------------------------------------------------
step: 180 / 200
[Debug Tokens]: [13535, 573, 7762, 4894, 108, 4022, 235292, 235248, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[Pi0.5 Subtask]: raise the arm.;
Action:
```

## 技术细节

### 子任务生成流程
1. **Embed prefix**: 图像 + 高级提示 → prefix tokens
2. **Auto-regressive generation**: 使用 while_loop 生成子任务 tokens
3. **KV cache**: 保存图像 + 高级提示 + 生成的子任务的 KV cache
4. **Action generation**: 基于包含子任务的 KV cache 进行 flow matching

### 训练 vs 推理对比

| 阶段 | 子任务来源 | 损失计算 |
|------|-----------|---------|
| **训练** | Ground truth（数据集提供） | Cross-entropy loss on subtask tokens |
| **推理** | 自回归生成（`sample_low_level_task`） | 无损失计算 |

### 为什么之前有输出？

可能的原因：
1. 之前使用的是包含 `sample_low_level_task` 的代码版本
2. 或者权重本身包含了某种默认子任务信息（不太可能）

## 测试方法

重新运行之前的 eval 命令：
```bash
bash eval.sh beat_block_hammer demo_randomized pi05_aloha_full_base demo_randomized_beat_block_hammer 0 7 26000
```

应该能在终端看到 `[Pi0.5 Subtask]:` 开头的日志输出。

## 注意事项

1. **Batch size 限制**: `sample_actions` 中添加了 `assert batch_size == 1`，因为不同样本的子任务长度可能不同
2. **max_decoding_steps**: 默认设置为 20，可根据实际子任务长度调整
3. **Temperature**: 默认为 0.0（贪婪解码），可以设置 > 0 进行采样
4. **日志**: 每次推理都会通过 `logger.info` 输出生成的子任务文本

## 参考实现

完整参考实现位于：
- `/mnt/data1/liujingzhi/openpi_subtask_generation-main/src/openpi/models/pi05.py`
  - `sample_low_level_task()`: 第272-357行
  - `sample_actions()`: 第359-423行
