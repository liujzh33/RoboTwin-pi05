# PI0.5 快速开始指南

## 🎯 5分钟快速理解

### 核心概念

PI0.5 是一个**视觉语言动作模型**，输入图像+语言指令，输出机器人动作序列。

```
图像 + 语言指令 + 机器人状态 → [PI0.5模型] → 动作序列
```

### 三个核心组件

1. **模型** (`models_pytorch/pi0_pytorch.py`)
   - 双专家架构：PaliGemma（视觉语言）+ Expert Gemma（动作生成）
   - 使用扩散模型生成动作

2. **策略** (`policies/policy.py`)
   - 封装模型，提供 `infer(obs)` 接口
   - 处理数据转换和归一化

3. **配置** (`training/config.py`)
   - 定义模型、数据、训练参数
   - 通过名称加载预定义配置

## 📖 代码阅读路径（推荐顺序）

### 第1步：看接口（15分钟）

**文件：** `src/openpi/policies/policy.py`

```python
# 核心接口
policy = Policy(model, transforms=..., output_transforms=...)
actions = policy.infer(observation)  # 这就是你主要使用的接口
```

**关键点：**
- `Policy.infer()` 是主要接口
- 输入是字典格式的观测
- 输出包含 `actions` 字段

### 第2步：看示例（20分钟）

**文件：** `examples/simple_client/main.py`

这个文件展示了：
1. 如何加载模型
2. 如何准备输入数据
3. 如何调用推理接口

**文件：** `pi_model.py`

这是更简化的封装：
```python
pi0 = PI0(train_config_name, model_name, checkpoint_id, pi0_step)
pi0.set_language("pick up the cup")
pi0.update_observation_window(images, state)
actions = pi0.get_action()
```

### 第3步：看模型架构（30分钟）

**文件：** `src/openpi/models_pytorch/pi0_pytorch.py`

**重点方法：**

1. `__init__()` (第 85-124 行)
   - 初始化 PaliGemma 和 Expert Gemma
   - 设置投影层

2. `forward()` (第 316-373 行)
   - 训练时使用
   - 输入：observation + actions
   - 输出：loss（MSE loss for noise prediction）

3. `sample_actions()` (第 375-419 行) ⭐ **最重要**
   - 推理时使用
   - 实现扩散去噪过程
   - 从噪声开始，逐步去噪得到动作

4. `embed_prefix()` (第 186-235 行)
   - 编码图像和语言提示

5. `embed_suffix()` (第 237-314 行)
   - 编码状态和动作序列

### 第4步：看数据流（20分钟）

**文件：** `src/openpi/models/model.py`

理解数据结构：
- `Observation`: 包含 images, state, tokenized_prompt 等
- `Actions`: 动作序列 (batch, action_horizon, action_dim)

**文件：** `src/openpi/transforms.py`

理解数据转换管道：
```
原始数据 → 归一化 → 模型输入格式
```

## 🔧 实际使用流程

### 场景1：使用已有模型进行推理

```python
from openpi.policies import policy_config
from openpi.training import config as _config

# 1. 加载配置
train_config = _config.get_config("pi05_base")

# 2. 加载模型
policy = policy_config.create_trained_policy(
    train_config,
    checkpoint_dir="path/to/checkpoint"
)

# 3. 准备观测
observation = {
    "state": np.array([...]),  # 机器人状态
    "images": {
        "base_0_rgb": np.array([...]),  # 图像 (H, W, 3)
        "left_wrist_0_rgb": np.array([...]),
        "right_wrist_0_rgb": np.array([...]),
    },
    "prompt": "pick up the cup"  # 语言指令
}

# 4. 推理
result = policy.infer(observation)
actions = result["actions"]  # (action_horizon, action_dim)
```

### 场景2：训练新模型

```python
# 1. 准备数据（LeRobot 格式）
# 2. 运行训练脚本
python scripts/train.py --config pi05_base --data_dir ...
```

### 场景3：使用简化接口

```python
from pi_model import PI0

pi0 = PI0(
    train_config_name="pi05_base",
    model_name="pi05",
    checkpoint_id="checkpoint_id",
    pi0_step=10
)

pi0.set_language("pick up the cup")
pi0.update_observation_window(images, state)
actions = pi0.get_action()
```

## 🗂️ 目录功能速查

| 目录 | 功能 | 何时查看 |
|------|------|----------|
| `models_pytorch/` | PyTorch 模型实现 | 理解模型架构时 |
| `policies/` | 策略接口 | 使用模型时 |
| `training/` | 训练相关 | 训练模型时 |
| `examples/` | 示例代码 | 学习如何使用 |
| `scripts/` | 脚本文件 | 运行训练/评估 |
| `checkpoints/` | 模型权重 | 加载模型时 |

## 🎓 学习建议

### 如果你要**使用模型**：
1. 先看 `examples/simple_client/main.py`
2. 再看 `pi_model.py`（更简单的接口）
3. 理解 `Policy.infer()` 的输入输出格式

### 如果你要**理解模型**：
1. 看 `pi0_pytorch.py` 的 `sample_actions()` 方法
2. 理解扩散去噪过程
3. 看 `gemma_pytorch.py` 理解双专家架构

### 如果你要**训练模型**：
1. 看 `scripts/train.py`
2. 看 `training/config.py` 理解配置
3. 看 `training/data_loader.py` 理解数据格式

### 如果你要**修改模型**：
1. 先完全理解 `pi0_pytorch.py`
2. 理解 `gemma_pytorch.py` 的双专家架构
3. 注意保持接口兼容性

## ⚡ 关键代码片段

### 模型推理核心流程

```python
# 在 pi0_pytorch.py 的 sample_actions() 中：

# 1. 编码图像和语言（前缀）
prefix_embs = self.embed_prefix(images, lang_tokens)

# 2. 计算 KV cache（加速推理）
_, past_key_values = self.paligemma_with_expert.forward(
    inputs_embeds=[prefix_embs, None],
    use_cache=True
)

# 3. 扩散去噪过程
x_t = noise  # 从噪声开始
time = 1.0
while time >= 0:
    # 去噪一步
    v_t = self.denoise_step(state, past_key_values, x_t, time)
    # Euler 步进
    x_t = x_t + dt * v_t
    time += dt

return x_t  # 最终动作
```

### 数据格式

```python
# 输入格式（Observation）
{
    "state": np.array([14]),  # 机器人状态
    "images": {
        "base_0_rgb": np.array([224, 224, 3]),  # RGB 图像
        "left_wrist_0_rgb": np.array([224, 224, 3]),
        "right_wrist_0_rgb": np.array([224, 224, 3]),
    },
    "prompt": "instruction text"  # 语言指令
}

# 输出格式
{
    "actions": np.array([action_horizon, action_dim]),  # 动作序列
    "state": np.array([14]),  # 原始状态（透传）
    "policy_timing": {"infer_ms": 123.4}  # 推理时间
}
```

## 🐛 常见问题快速解决

**Q: 如何加载模型？**
```python
from openpi.policies import policy_config
from openpi.training import config as _config

train_config = _config.get_config("pi05_base")
policy = policy_config.create_trained_policy(
    train_config,
    checkpoint_dir="path/to/checkpoint"
)
```

**Q: 输入图像格式是什么？**
- RGB 图像，形状 `(H, W, 3)`，值域 `[0, 255]` 或 `[-1, 1]`
- 默认分辨率 `224x224`

**Q: 如何设置语言指令？**
- 在 `observation["prompt"]` 中提供字符串
- 或使用 `transforms.InjectDefaultPrompt()`

**Q: 动作序列长度是多少？**
- 由 `config.action_horizon` 决定（通常是 8 或 16）
- 输出是 `(action_horizon, action_dim)` 的形状

## 📚 下一步

1. ✅ 运行 `examples/simple_client/main.py` 验证环境
2. ✅ 阅读 `pi0_pytorch.py` 的 `sample_actions()` 方法
3. ✅ 尝试修改输入，观察输出变化
4. ✅ 查看训练脚本，理解训练流程
5. ✅ 阅读完整架构文档 `ARCHITECTURE.md`

