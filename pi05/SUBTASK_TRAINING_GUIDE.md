# PI0.5 子任务训练实现指南

## 📋 概述

本文档说明如何修改 PI0.5 代码以支持使用子任务数据进行训练。

## 🔑 关键修改总结

### 1. **子任务预测位置**

**子任务预测在 `prefix_out` 中，需要解码！**

在 `pi05.py` 的 `compute_loss()` 方法中：

```python
# 1. 前向传播生成 prefix 输出
(prefix_out, _), kv_cache = self.PaliGemma.llm(
    [prefix_token_embeddings, None], 
    mask=prefix_attn_mask, 
    positions=prefix_positions, 
    adarms_cond=[None, None],
    kv_cache=None
)

# 2. 解码为 logits（这里就是子任务预测的位置）
logits = self.PaliGemma.llm(
    prefix_out[:, -targets.shape[1] :], method='deembed'
)

# 3. 计算 Cross-Entropy Loss（只在 loss_mask=True 的位置）
loss_mask = observation.token_loss_mask[:, 1:]  # 子任务部分为 True
token_pplx = jnp.sum(targets * logp, axis=-1)
subtask_generation_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / ...
```

**关键点：**
- `prefix_out` 是模型对 prefix tokens 的隐藏状态输出
- 通过 `method='deembed'` 将隐藏状态解码为词汇表 logits
- `logits` 就是子任务的预测（每个位置预测下一个 token）
- **需要解码**：使用 `tokenizer.detokenize()` 将 tokens 解码为文本

### 2. **推理时生成子任务**

在 `sample_low_level_task()` 方法中（需要添加）：

```python
def sample_low_level_task(self, ...):
    # 1. 生成子任务 tokens（自回归解码）
    output_tokens = ...  # [B, max_decoding_steps]
    
    # 2. 解码为文本
    subtask_text = tokenizer.detokenize(output_tokens[0])
    return subtask_text
```

**子任务预测参数：**
- **位置**：`prefix_out` → `logits` (通过 `deembed`)
- **格式**：Token IDs（需要解码为文本）
- **解码方式**：使用 `PaligemmaTokenizer.detokenize()`

## 📝 已修改的文件

### 1. `src/openpi/models/tokenizer.py`

**添加的方法：**
- `tokenize_high_low_prompt()`: Tokenize 高级任务和子任务
- `detokenize()`: 将 tokens 解码为文本

**关键代码：**
```python
def tokenize_high_low_prompt(self, high_prompt: str, low_prompt: str):
    # 格式: "Task: {high_prompt}. Subtask: {low_prompt};\nAction: "
    # 返回: tokens, mask, ar_mask, loss_mask
    # loss_mask: True 只在子任务部分（用于计算损失）
```

### 2. `src/openpi/models/pi05.py` (新建)

**关键功能：**
- `compute_loss()`: 添加子任务生成损失
- `embed_prefix()`: 语言 tokens 使用自回归注意力（`ar_mask=True`）

**损失计算：**
```python
total_loss = subtask_generation_loss + flow_matching_loss
```

### 3. `src/openpi/transforms.py`

**添加的 Transform：**
- `TokenizeHighLowPrompt`: Tokenize 高级任务和子任务
- `LoadSubtaskFromInstructions`: 从 instructions.json 加载子任务数据

### 4. `src/openpi/models/pi0_config.py`

**修改：**
- `create()`: 当 `pi05=True` 时创建 `Pi05` 而不是 `Pi0`
- `inputs_spec()`: 添加 `token_ar_mask` 和 `token_loss_mask` 字段

## 🔧 数据加载流程

### Step 1: 读取 instructions.json

```python
import json

with open('episode_0/instructions.json', 'r') as f:
    data = json.load(f)

# data 包含:
# - instructions: 高级任务列表
# - subtasks: 子任务列表（每个对应一个 instructions）
# - phase_info: 阶段信息
```

### Step 2: 使用 Transform 加载子任务

```python
from openpi.transforms import LoadSubtaskFromInstructions, TokenizeHighLowPrompt
from openpi.models.tokenizer import PaligemmaTokenizer

# 1. 加载子任务数据
load_subtask = LoadSubtaskFromInstructions(use_first_subtask=False)
data = load_subtask(data)  # 添加 high_prompt 和 low_prompt

# 2. Tokenize
tokenizer = PaligemmaTokenizer(max_len=200)
tokenize_transform = TokenizeHighLowPrompt(tokenizer=tokenizer)
data = tokenize_transform(data)  # 添加 tokenized_prompt, token_ar_mask, token_loss_mask
```

### Step 3: 构建 Observation

```python
from openpi.models.model import Observation

observation = Observation.from_dict({
    'image': images,
    'image_mask': image_masks,
    'state': state,
    'tokenized_prompt': data['tokenized_prompt'],
    'tokenized_prompt_mask': data['tokenized_prompt_mask'],
    'token_ar_mask': data['token_ar_mask'],
    'token_loss_mask': data['token_loss_mask'],
})
```

## 🎯 训练配置修改

### 修改训练配置以使用子任务数据

在 `src/openpi/training/config.py` 中，需要修改 `ModelTransformFactory` 以使用 `TokenizeHighLowPrompt` 而不是 `TokenizePrompt`。

**示例配置修改：**

```python
# 在 ModelTransformFactory 中
if model_config.pi05 and use_subtasks:
    # 使用子任务 tokenize
    model_transforms = [
        _transforms.LoadSubtaskFromInstructions(),
        _transforms.TokenizeHighLowPrompt(tokenizer=tokenizer),
        ...
    ]
else:
    # 使用普通 tokenize
    model_transforms = [
        _transforms.TokenizePrompt(tokenizer=tokenizer, discrete_state_input=...),
        ...
    ]
```

## 📊 子任务预测详解

### 预测流程

1. **输入**：
   - 图像 tokens
   - 高级任务 tokens: `"Task: {high_prompt}. Subtask: "`
   - 子任务 tokens: `"{low_prompt};\nAction: "`（训练时作为 ground truth）

2. **模型处理**：
   - `embed_prefix()`: 编码所有 prefix tokens
   - `PaliGemma.llm()`: 前向传播，生成 `prefix_out`（隐藏状态）

3. **解码预测**：
   ```python
   logits = self.PaliGemma.llm(prefix_out, method='deembed')
   # logits shape: [B, seq_len, vocab_size]
   # 这就是子任务的预测！
   ```

4. **计算损失**：
   ```python
   # 只在 loss_mask=True 的位置计算损失（子任务部分）
   loss_mask = observation.token_loss_mask[:, 1:]  # 子任务 tokens
   subtask_loss = -jnp.sum(token_pplx * loss_mask, axis=-1)
   ```

### 推理时生成子任务

```python
# 1. 输入只有高级任务
high_prompt = "Grab the hammer and beat the block"
# Tokenize: "Task: {high_prompt}. Subtask: "

# 2. 模型自回归生成子任务
subtask_tokens = model.sample_low_level_task(observation, max_decoding_steps=25)

# 3. 解码为文本
subtask_text = tokenizer.detokenize(subtask_tokens)
# 输出: "Grab the hammer."
```

## ⚠️ 重要说明

### 1. **子任务预测位置**

- **训练时**：`prefix_out` → `logits` (通过 `deembed`)
- **推理时**：`sample_low_level_task()` 返回的 `output_tokens`
- **需要解码**：使用 `tokenizer.detokenize()` 将 token IDs 转换为文本

### 2. **Loss Mask 的作用**

- `token_loss_mask`: 指示哪些位置需要计算损失
- `True`: 子任务部分（需要预测的部分）
- `False`: 高级任务部分（输入，不需要预测）

### 3. **AR Mask 的变化**

- **原始 pi0**: `ar_mask = [False]` (双向注意力)
- **pi05 子任务**: `ar_mask = [True]` (自回归，用于生成子任务)

### 4. **数据格式要求**

训练数据必须包含：
- `instructions`: 高级任务列表
- `subtasks`: 子任务列表（与 instructions 一一对应）

## 🚀 使用示例

### 训练时

```python
# 数据加载
data = {
    'instructions': ["Grab the hammer and beat the block"],
    'subtasks': [["Grab the hammer", "Strike the block with the hammer"]],
    'images': ...,
    'state': ...,
    'actions': ...,
}

# Transform
data = LoadSubtaskFromInstructions()(data)
data = TokenizeHighLowPrompt(tokenizer)(data)

# 训练
observation = Observation.from_dict(data)
loss = model.compute_loss(rng, observation, actions, train=True)
```

### 推理时

```python
# 1. 生成子任务
subtask_tokens = model.sample_low_level_task(rng, observation)
subtask_text = tokenizer.detokenize(subtask_tokens[0])
print(f"Generated subtask: {subtask_text}")

# 2. 使用子任务生成动作
actions = model.sample_actions(rng, observation)
```

## 📚 相关文件

- **模型代码**: `src/openpi/models/pi05.py`
- **Tokenizer**: `src/openpi/models/tokenizer.py` (添加了 `tokenize_high_low_prompt`)
- **Transforms**: `src/openpi/transforms.py` (添加了 `TokenizeHighLowPrompt`, `LoadSubtaskFromInstructions`)
- **配置**: `src/openpi/models/pi0_config.py` (修改了 `create()` 和 `inputs_spec()`)

## 🔍 调试建议

1. **检查 token_loss_mask**：
   ```python
   print(observation.token_loss_mask)
   # 应该看到：高级任务部分为 False，子任务部分为 True
   ```

2. **检查生成的 logits**：
   ```python
   logits = model.PaliGemma.llm(prefix_out, method='deembed')
   predicted_tokens = jnp.argmax(logits, axis=-1)
   print(tokenizer.detokenize(predicted_tokens[0]))
   ```

3. **验证损失计算**：
   ```python
   loss = model.compute_loss(rng, observation, actions, train=True)
   print(f"Subtask loss: {loss}")
   ```

## ✅ 检查清单

- [ ] `tokenizer.py` 包含 `tokenize_high_low_prompt()` 和 `detokenize()`
- [ ] `pi05.py` 包含子任务生成损失计算
- [ ] `transforms.py` 包含 `TokenizeHighLowPrompt` 和 `LoadSubtaskFromInstructions`
- [ ] `pi0_config.py` 在 `pi05=True` 时创建 `Pi05`
- [ ] 数据加载代码读取 `instructions.json` 中的 `subtasks`
- [ ] 训练配置使用 `TokenizeHighLowPrompt` 而不是 `TokenizePrompt`

---

**关键要点：子任务预测在 `prefix_out` 解码后的 `logits` 中，需要解码为文本！**

