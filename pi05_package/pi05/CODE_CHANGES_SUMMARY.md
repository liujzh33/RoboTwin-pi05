# PI0.5 子任务训练代码修改总结

## 📁 修改的文件列表

### 1. **核心模型文件**

#### ✅ `src/openpi/models/tokenizer.py`
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/src/openpi/models/tokenizer.py`

**修改内容**:
- 在 `PaligemmaTokenizer` 类中添加了 `tokenize_high_low_prompt()` 方法（第 51-120 行）
- 添加了 `detokenize()` 方法（第 122-126 行）

**关键代码**:
```python
def tokenize_high_low_prompt(self, high_prompt: str, low_prompt: str):
    # 格式: "Task: {high_prompt}. Subtask: {low_prompt};\nAction: "
    # 返回: tokens, mask, ar_mask, loss_mask
```

---

#### ✅ `src/openpi/models/pi05.py` (新建文件)
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/src/openpi/models/pi05.py`

**修改内容**:
- 创建了新的 `Pi05` 类（基于 `Pi0`，但支持子任务生成）
- 修改了 `embed_prefix()` 方法：语言 tokens 使用自回归注意力（`ar_mask=True`，第 127-137 行）
- 修改了 `compute_loss()` 方法：添加子任务生成损失（Cross-Entropy Loss，第 188-269 行）

**关键代码**:
```python
# 子任务生成损失
logits = self.PaliGemma.llm(prefix_out, method='deembed')
subtask_generation_loss = -jnp.sum(token_pplx * loss_mask, axis=-1)

# 总损失
return subtask_generation_loss + jnp.mean(flow_loss, axis=-1)
```

---

#### ✅ `src/openpi/models/gemma.py`
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/src/openpi/models/gemma.py`

**修改内容**:
- 在 `Module` 类中添加了 `deembed()` 方法（第 388-391 行）

**关键代码**:
```python
def deembed(self, embeddings):
    """Decode embeddings back to vocabulary logits."""
    return self.embedder.decode(embeddings)
```

---

#### ✅ `src/openpi/models/pi0_config.py`
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/src/openpi/models/pi0_config.py`

**修改内容**:
- 修改了 `create()` 方法：当 `pi05=True` 时创建 `Pi05` 而不是 `Pi0`（第 48-54 行）
- 修改了 `inputs_spec()` 方法：添加 `token_ar_mask` 和 `token_loss_mask` 字段（第 72-73 行）
- 添加了类型导入（第 15 行）

**关键代码**:
```python
def create(self, rng):
    if self.pi05:
        from openpi.models.pi05 import Pi05
        return Pi05(self, rngs=nnx.Rngs(rng))
    else:
        from openpi.models.pi0 import Pi0
        return Pi0(self, rngs=nnx.Rngs(rng))
```

---

### 2. **数据转换文件**

#### ✅ `src/openpi/transforms.py`
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/src/openpi/transforms.py`

**修改内容**:
- 添加了 `TokenizeHighLowPrompt` transform 类（第 269-290 行）
- 添加了 `LoadSubtaskFromInstructions` transform 类（第 327-365 行）

**关键代码**:
```python
@dataclasses.dataclass(frozen=True)
class TokenizeHighLowPrompt(DataTransformFn):
    """Tokenize high-level prompt and low-level (subtask) prompt."""
    tokenizer: _tokenizer.PaligemmaTokenizer
    
    def __call__(self, data):
        tokens, mask, ar_mask, loss_mask = self.tokenizer.tokenize_high_low_prompt(
            data['high_prompt'], data['low_prompt']
        )
        return {**data, "tokenized_prompt": tokens, ...}

@dataclasses.dataclass(frozen=True)
class LoadSubtaskFromInstructions(DataTransformFn):
    """Load subtasks from instructions.json."""
    def __call__(self, data):
        # 从 instructions 和 subtasks 中随机选择
        # 返回 high_prompt 和 low_prompt
```

---

### 3. **新增脚本文件**

#### ✅ `scripts/detect_gripper_phases.py` (新建)
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/scripts/detect_gripper_phases.py`

**功能**: 检测 HDF5 文件中的夹爪开闭状态，自动识别抓取阶段结束点，并可视化

---

#### ✅ `scripts/add_subtasks_to_data.py` (新建)
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/scripts/add_subtasks_to_data.py`

**功能**: 为已处理的 `processed_data` 添加子任务信息，更新 `instructions.json`

---

#### ✅ `scripts/process_data_with_subtasks.py` (新建)
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/scripts/process_data_with_subtasks.py`

**功能**: 在数据转换时自动检测夹爪状态并添加子任务（`process_data.py` 的增强版本）

---

#### ✅ `scripts/load_subtask_data_example.py` (新建)
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/scripts/load_subtask_data_example.py`

**功能**: 展示如何加载包含子任务的数据用于训练

---

### 4. **文档文件**

#### ✅ `SUBTASK_TRAINING_GUIDE.md` (新建)
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/SUBTASK_TRAINING_GUIDE.md`

**内容**: 详细的子任务训练指南

---

#### ✅ `SUBTASK_QUICK_START.md` (新建)
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/SUBTASK_QUICK_START.md`

**内容**: 快速开始指南

---

#### ✅ `SUBTASK_IMPLEMENTATION_COMPLETE.md` (新建)
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/SUBTASK_IMPLEMENTATION_COMPLETE.md`

**内容**: 完整实现总结

---

#### ✅ `SUBTASK_DATA_GUIDE.md` (新建)
**路径**: `/mnt/data1/liujingzhi/RoboTwin/policy/pi05/SUBTASK_DATA_GUIDE.md`

**内容**: 数据生成指南

---

## 📊 修改统计

### 修改的文件（5个）
1. `src/openpi/models/tokenizer.py` - 添加 2 个方法
2. `src/openpi/models/pi05.py` - 新建文件（299 行）
3. `src/openpi/models/gemma.py` - 添加 1 个方法
4. `src/openpi/models/pi0_config.py` - 修改 2 个方法
5. `src/openpi/transforms.py` - 添加 2 个 transform 类

### 新建的脚本文件（4个）
1. `scripts/detect_gripper_phases.py` - 夹爪状态检测
2. `scripts/add_subtasks_to_data.py` - 添加子任务数据
3. `scripts/process_data_with_subtasks.py` - 数据处理（带子任务）
4. `scripts/load_subtask_data_example.py` - 数据加载示例

### 新建的文档文件（4个）
1. `SUBTASK_TRAINING_GUIDE.md`
2. `SUBTASK_QUICK_START.md`
3. `SUBTASK_IMPLEMENTATION_COMPLETE.md`
4. `SUBTASK_DATA_GUIDE.md`

---

## 🔑 核心修改点

### 1. **子任务 Tokenize** (`tokenizer.py`)
- 位置: `PaligemmaTokenizer.tokenize_high_low_prompt()`
- 功能: 将高级任务和子任务 tokenize 为模型输入格式

### 2. **子任务生成损失** (`pi05.py`)
- 位置: `Pi05.compute_loss()` 方法
- 功能: 计算子任务生成的 Cross-Entropy Loss

### 3. **自回归注意力** (`pi05.py`)
- 位置: `Pi05.embed_prefix()` 方法
- 修改: `ar_mask += [True]` (原来是 `[False]`)

### 4. **数据加载** (`transforms.py`)
- 位置: `LoadSubtaskFromInstructions` 和 `TokenizeHighLowPrompt`
- 功能: 从 `instructions.json` 加载子任务并 tokenize

### 5. **模型创建** (`pi0_config.py`)
- 位置: `Pi0Config.create()` 方法
- 修改: `pi05=True` 时创建 `Pi05` 而不是 `Pi0`

---

## 📝 文件路径总览

```
/mnt/data1/liujingzhi/RoboTwin/policy/pi05/
├── src/openpi/models/
│   ├── tokenizer.py          ✅ 修改（添加 tokenize_high_low_prompt, detokenize）
│   ├── pi05.py               ✅ 新建（子任务生成模型）
│   ├── gemma.py              ✅ 修改（添加 deembed 方法）
│   └── pi0_config.py         ✅ 修改（支持创建 Pi05）
├── src/openpi/transforms.py  ✅ 修改（添加 TokenizeHighLowPrompt, LoadSubtaskFromInstructions）
└── scripts/
    ├── detect_gripper_phases.py        ✅ 新建
    ├── add_subtasks_to_data.py         ✅ 新建
    ├── process_data_with_subtasks.py    ✅ 新建
    └── load_subtask_data_example.py    ✅ 新建
```

---

## 🎯 关键修改说明

### 1. 子任务预测位置

**文件**: `src/openpi/models/pi05.py`  
**方法**: `compute_loss()`  
**位置**: 第 200 行

```python
logits = self.PaliGemma.llm(prefix_out[:, -targets.shape[1] :], method='deembed')
# logits 就是子任务的预测！
# 需要解码: argmax → detokenize()
```

### 2. 子任务 Tokenize

**文件**: `src/openpi/models/tokenizer.py`  
**方法**: `tokenize_high_low_prompt()`  
**位置**: 第 51-120 行

### 3. 数据加载

**文件**: `src/openpi/transforms.py`  
**类**: `LoadSubtaskFromInstructions`, `TokenizeHighLowPrompt`  
**位置**: 第 269-365 行

---

**所有修改已完成，代码可以直接使用！**

