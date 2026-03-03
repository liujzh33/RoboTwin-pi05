# PI0.5 子任务训练实现完成总结

## ✅ 已完成的修改

### 1. **模型代码修改**

#### `src/openpi/models/tokenizer.py`
- ✅ 添加 `tokenize_high_low_prompt()` 方法
  - 输入：高级任务和子任务文本
  - 输出：tokens, mask, ar_mask, loss_mask
  - 格式：`"Task: {high_prompt}. Subtask: {low_prompt};\nAction: "`
- ✅ 添加 `detokenize()` 方法
  - 将 token IDs 解码为文本

#### `src/openpi/models/pi05.py` (新建)
- ✅ 创建 `Pi05` 类（基于 `Pi0`，但支持子任务生成）
- ✅ 修改 `embed_prefix()`: 语言 tokens 使用自回归注意力（`ar_mask=True`）
- ✅ 修改 `compute_loss()`: 添加子任务生成损失
  - 子任务生成损失（Cross-Entropy）
  - Flow Matching 损失（MSE）
  - 总损失 = 子任务损失 + 动作损失

#### `src/openpi/models/gemma.py`
- ✅ 添加 `deembed()` 方法
  - 将隐藏状态解码为词汇表 logits

#### `src/openpi/models/pi0_config.py`
- ✅ 修改 `create()`: 当 `pi05=True` 时创建 `Pi05`
- ✅ 修改 `inputs_spec()`: 添加 `token_ar_mask` 和 `token_loss_mask` 字段

### 2. **数据转换代码修改**

#### `src/openpi/transforms.py`
- ✅ 添加 `TokenizeHighLowPrompt` transform
  - 使用 `tokenizer.tokenize_high_low_prompt()` tokenize 高级任务和子任务
- ✅ 添加 `LoadSubtaskFromInstructions` transform
  - 从 `instructions.json` 读取 `instructions` 和 `subtasks`
  - 随机选择或使用第一个 instruction/subtask 对

### 3. **文档和示例**

- ✅ `SUBTASK_TRAINING_GUIDE.md`: 详细的训练指南
- ✅ `scripts/load_subtask_data_example.py`: 数据加载示例

## 🎯 子任务预测详解

### **子任务预测在哪里？**

**答案：在 `prefix_out` 解码后的 `logits` 中，需要解码为文本！**

#### 训练时

```python
# 1. 前向传播
(prefix_out, _), kv_cache = self.PaliGemma.llm(
    [prefix_token_embeddings, None], 
    mask=prefix_attn_mask, 
    positions=prefix_positions, 
    adarms_cond=[None, None],
    kv_cache=None
)

# 2. 解码为 logits（这就是子任务预测！）
logits = self.PaliGemma.llm(prefix_out[:, -targets.shape[1] :], method='deembed')
# logits shape: [B, seq_len, vocab_size]
# 每个位置预测下一个 token 的概率分布

# 3. 计算损失（只在 loss_mask=True 的位置）
loss_mask = observation.token_loss_mask[:, 1:]  # 子任务部分为 True
subtask_loss = -jnp.sum(token_pplx * loss_mask, axis=-1)
```

#### 推理时（需要添加 `sample_low_level_task` 方法）

```python
# 1. 自回归生成子任务 tokens
output_tokens = model.sample_low_level_task(observation, max_decoding_steps=25)
# output_tokens shape: [B, max_decoding_steps]

# 2. 解码为文本
subtask_text = tokenizer.detokenize(output_tokens[0])
# 输出: "Grab the hammer."
```

### **是否需要解码？**

**是的，需要解码！**

- **训练时**：`logits` 是概率分布，通过 `argmax` 得到 token IDs，然后使用 `detokenize()` 解码为文本
- **推理时**：`output_tokens` 是 token IDs，直接使用 `detokenize()` 解码为文本

### **子任务预测参数**

| 参数 | 位置 | 格式 | 解码方式 |
|------|------|------|----------|
| **训练时** | `logits` (从 `prefix_out` 解码) | `[B, seq_len, vocab_size]` | `argmax` → `detokenize()` |
| **推理时** | `output_tokens` (从 `sample_low_level_task`) | `[B, max_decoding_steps]` | `detokenize()` |

## 📊 数据流程

### 训练数据流程

```
instructions.json
  ↓
LoadSubtaskFromInstructions
  ↓ (添加 high_prompt, low_prompt)
TokenizeHighLowPrompt
  ↓ (添加 tokenized_prompt, token_ar_mask, token_loss_mask)
Observation.from_dict()
  ↓
model.compute_loss()
  ↓
subtask_generation_loss + flow_matching_loss
```

### 推理数据流程

```
高级任务文本
  ↓
TokenizeHighLevelPrompt (需要添加)
  ↓
Observation
  ↓
model.sample_low_level_task()
  ↓
output_tokens
  ↓
tokenizer.detokenize()
  ↓
子任务文本
```

## 🔧 使用步骤

### Step 1: 确保数据包含子任务

```bash
# 检查 instructions.json 是否包含 subtasks
cat processed_data/.../episode_0/instructions.json | python -m json.tool | grep subtasks
```

### Step 2: 修改训练配置

在训练配置中使用 `TokenizeHighLowPrompt` 而不是 `TokenizePrompt`：

```python
# 在 config.py 的 ModelTransformFactory 中
if model_config.pi05 and use_subtasks:
    model_transforms = [
        _transforms.LoadSubtaskFromInstructions(),
        _transforms.TokenizeHighLowPrompt(tokenizer=tokenizer),
        ...
    ]
```

### Step 3: 训练模型

```bash
# 使用 pi05=True 的配置
python scripts/train.py --config-name your_pi05_config
```

## ⚠️ 重要注意事项

### 1. **AR Mask 变化**

- **原始 pi0**: `ar_mask = [False]` (双向注意力)
- **pi05 子任务**: `ar_mask = [True]` (自回归，用于生成子任务)

这是子任务生成的关键！

### 2. **Loss Mask 的作用**

- `token_loss_mask`: 指示哪些位置需要计算损失
  - `True`: 子任务部分（需要预测）
  - `False`: 高级任务部分（输入，不需要预测）

### 3. **数据格式要求**

训练数据必须包含：
- `instructions`: 高级任务列表
- `subtasks`: 子任务列表（与 instructions 一一对应）

### 4. **模型配置**

确保 `pi0_config` 中 `pi05=True`：

```python
config = Pi0Config(
    pi05=True,  # 必须为 True
    action_dim=32,
    action_horizon=50,
    max_token_len=200,  # 子任务需要更长的序列
    ...
)
```

## 🔍 验证检查

### 检查模型是否正确创建

```python
from openpi.models.pi0_config import Pi0Config
from openpi.models.model import ModelType

config = Pi0Config(pi05=True, ...)
assert config.model_type == ModelType.PI05

model = config.create(rng)
assert isinstance(model, Pi05)  # 应该是 Pi05 而不是 Pi0
```

### 检查数据格式

```python
observation = ...
assert observation.token_loss_mask is not None
assert observation.token_ar_mask is not None

# 检查 loss_mask
print(f"Loss mask: {observation.token_loss_mask[0]}")
# 应该看到：高级任务部分为 False，子任务部分为 True
```

### 检查损失计算

```python
loss = model.compute_loss(rng, observation, actions, train=True)
print(f"Total loss: {loss}")
# 应该包含子任务损失和动作损失
```

## 📝 待完成的工作

### 1. 添加 `sample_low_level_task()` 方法

需要在 `pi05.py` 中添加 `sample_low_level_task()` 方法用于推理时生成子任务。参考 `openpi_subtask_generation-main/src/openpi/models/pi05.py` 的实现。

### 2. 修改训练配置

需要在 `src/openpi/training/config.py` 中修改 `ModelTransformFactory`，使其在 `pi05=True` 时使用 `TokenizeHighLowPrompt`。

### 3. 数据加载器修改

需要修改数据加载器，使其能够从 `processed_data` 目录读取 `instructions.json` 并提取 `subtasks`。

## 📚 相关文件

- **模型**: `src/openpi/models/pi05.py`
- **Tokenizer**: `src/openpi/models/tokenizer.py`
- **Transforms**: `src/openpi/transforms.py`
- **配置**: `src/openpi/models/pi0_config.py`
- **Gemma**: `src/openpi/models/gemma.py` (添加了 `deembed` 方法)
- **文档**: `SUBTASK_TRAINING_GUIDE.md`
- **示例**: `scripts/load_subtask_data_example.py`

## 🎓 总结

### 子任务预测位置

1. **训练时**: `prefix_out` → `logits` (通过 `deembed`)
2. **推理时**: `sample_low_level_task()` → `output_tokens`
3. **都需要解码**: 使用 `tokenizer.detokenize()` 将 token IDs 转换为文本

### 关键修改

1. ✅ Tokenizer: 添加 `tokenize_high_low_prompt()` 和 `detokenize()`
2. ✅ 模型: 创建 `Pi05` 类，添加子任务生成损失
3. ✅ Transforms: 添加 `TokenizeHighLowPrompt` 和 `LoadSubtaskFromInstructions`
4. ✅ 配置: 修改 `pi0_config` 以支持创建 `Pi05`

### 下一步

1. 添加 `sample_low_level_task()` 方法
2. 修改训练配置以使用子任务数据
3. 修改数据加载器以读取子任务数据
4. 开始训练！

---

**关键要点：子任务预测在 `prefix_out` 解码后的 `logits` 中，需要解码为文本！**

