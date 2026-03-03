# PI0.5 子任务训练快速开始

## 🎯 核心问题回答

### Q: 子任务预测在哪个参数里？

**A: 在 `prefix_out` 解码后的 `logits` 中**

```python
# 在 pi05.py 的 compute_loss() 中
prefix_out = ...  # 模型隐藏状态输出
logits = self.PaliGemma.llm(prefix_out, method='deembed')
# logits shape: [B, seq_len, vocab_size]
# 这就是子任务的预测！
```

### Q: 需要解码吗？

**A: 是的，需要解码！**

- **训练时**: `logits` → `argmax` → token IDs → `detokenize()` → 文本
- **推理时**: `output_tokens` → `detokenize()` → 文本

```python
# 解码示例
predicted_tokens = jnp.argmax(logits, axis=-1)  # 获取预测的 token IDs
subtask_text = tokenizer.detokenize(predicted_tokens[0])  # 解码为文本
```

## 📋 已完成的修改清单

### ✅ 核心文件修改

1. **`src/openpi/models/tokenizer.py`**
   - ✅ 添加 `tokenize_high_low_prompt(high_prompt, low_prompt)`
   - ✅ 添加 `detokenize(tokens)`

2. **`src/openpi/models/pi05.py`** (新建)
   - ✅ 创建 `Pi05` 类
   - ✅ 修改 `embed_prefix()`: `ar_mask=True` (自回归)
   - ✅ 修改 `compute_loss()`: 添加子任务生成损失

3. **`src/openpi/models/gemma.py`**
   - ✅ 添加 `deembed()` 方法

4. **`src/openpi/models/pi0_config.py`**
   - ✅ 修改 `create()`: `pi05=True` 时创建 `Pi05`
   - ✅ 修改 `inputs_spec()`: 添加 `token_ar_mask` 和 `token_loss_mask`

5. **`src/openpi/transforms.py`**
   - ✅ 添加 `TokenizeHighLowPrompt` transform
   - ✅ 添加 `LoadSubtaskFromInstructions` transform

## 🚀 使用步骤

### Step 1: 验证数据格式

```bash
# 检查数据是否包含子任务
cat processed_data/beat_block_hammer-.../episode_0/instructions.json | \
    python -c "import json, sys; d=json.load(sys.stdin); print('Has subtasks:', 'subtasks' in d)"
```

### Step 2: 测试数据加载

```python
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.transforms import LoadSubtaskFromInstructions, TokenizeHighLowPrompt

tokenizer = PaligemmaTokenizer(max_len=200)

# 模拟数据
data = {
    'instructions': ["Grab the hammer and beat the block"],
    'subtasks': [["Grab the hammer", "Strike the block with the hammer"]],
}

# 加载子任务
data = LoadSubtaskFromInstructions()(data)
print(f"High prompt: {data['high_prompt']}")
print(f"Low prompt: {data['low_prompt']}")

# Tokenize
data = TokenizeHighLowPrompt(tokenizer=tokenizer)(data)
print(f"Tokenized prompt shape: {data['tokenized_prompt'].shape}")
print(f"Loss mask (first 50): {data['token_loss_mask'][:50]}")
```

### Step 3: 修改训练配置

在 `src/openpi/training/config.py` 中，找到 `ModelTransformFactory`，修改为：

```python
# 当使用子任务时
if model_config.pi05 and use_subtasks:
    model_transforms = [
        _transforms.LoadSubtaskFromInstructions(),
        _transforms.TokenizeHighLowPrompt(
            tokenizer=_tokenizer.PaligemmaTokenizer(max_len=model_config.max_token_len)
        ),
        ...
    ]
```

### Step 4: 开始训练

```bash
# 使用 pi05=True 的配置
python scripts/train.py --config-name your_pi05_config_with_subtasks
```

## 🔍 验证子任务预测

### 检查损失计算

```python
import jax
from openpi.models.pi0_config import Pi0Config
from openpi.models.model import Observation

# 创建模型
config = Pi0Config(pi05=True, action_dim=32, action_horizon=50, max_token_len=200)
model = config.create(jax.random.key(0))

# 创建测试数据
observation = config.fake_obs()
# 需要添加 token_loss_mask
observation = observation.replace(
    token_loss_mask=jnp.ones((1, 200), dtype=bool)  # 简化：所有位置都计算损失
)

# 计算损失
actions = config.fake_act()
loss = model.compute_loss(jax.random.key(0), observation, actions, train=True)
print(f"Loss: {loss}")
```

### 检查子任务预测

```python
# 在 compute_loss 中添加调试代码
logits = self.PaliGemma.llm(prefix_out[:, -targets.shape[1] :], method='deembed')
predicted_tokens = jnp.argmax(logits, axis=-1)

# 解码查看预测的子任务
from openpi.models.tokenizer import PaligemmaTokenizer
tokenizer = PaligemmaTokenizer(max_len=200)
predicted_text = tokenizer.detokenize(predicted_tokens[0])
print(f"Predicted subtask: {predicted_text}")
```

## 📊 数据格式

### 输入数据格式

```json
{
  "instructions": ["Grab the hammer and beat the block"],
  "subtasks": [["Grab the hammer", "Strike the block with the hammer"]],
  "phase_info": {
    "grasp_end_idx": 58,
    "total_steps": 114,
    "phase1_steps": 58,
    "phase2_steps": 56
  }
}
```

### 模型输入格式

```python
Observation(
    images={...},
    state=...,
    tokenized_prompt=tokens,  # "Task: {high}. Subtask: {low};\nAction: "
    tokenized_prompt_mask=mask,
    token_ar_mask=ar_mask,    # 所有语言 tokens 为 True (自回归)
    token_loss_mask=loss_mask, # 子任务部分为 True
)
```

## ⚠️ 重要提示

1. **AR Mask 必须为 True**: 语言 tokens 的 `ar_mask` 必须为 `True`，否则无法生成子任务
2. **Loss Mask 正确设置**: `token_loss_mask` 只在子任务部分为 `True`
3. **需要解码**: 子任务预测是 token IDs，需要使用 `detokenize()` 解码为文本
4. **数据格式**: 确保 `instructions.json` 包含 `subtasks` 字段

## 🔗 参考实现

完整的参考实现（包括 `sample_low_level_task` 方法）在：
- `/mnt/data1/liujingzhi/openpi_subtask_generation-main/src/openpi/models/pi05.py`

如果需要添加推理时的子任务生成功能，可以参考该实现。

---

**快速总结：子任务预测在 `logits` 中，需要解码！**

