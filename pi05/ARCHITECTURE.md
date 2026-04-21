# PI0.5 项目架构文档

## 📁 项目目录结构

```
pi05/
├── src/openpi/                    # 核心源代码
│   ├── models/                    # JAX 版本的模型实现
│   │   ├── pi0.py                 # PI0 模型（JAX版本）
│   │   ├── pi0_fast.py            # PI0 Fast 模型
│   │   ├── gemma.py                # Gemma 语言模型
│   │   ├── siglip.py               # SigLIP 视觉编码器
│   │   └── model.py                # 模型基类和接口定义
│   ├── models_pytorch/             # PyTorch 版本的模型实现 ⭐
│   │   ├── pi0_pytorch.py          # PI0.5 PyTorch 模型（核心）⭐
│   │   ├── gemma_pytorch.py        # PaliGemma + Expert Gemma 模型
│   │   ├── preprocessing_pytorch.py # 数据预处理
│   │   └── transformers_replace/   # Transformers 库的修改版本
│   ├── policies/                   # 策略接口和实现
│   │   ├── policy.py               # Policy 基类（核心接口）⭐
│   │   ├── policy_config.py        # 策略配置和加载
│   │   ├── aloha_policy.py         # ALOHA 机器人策略
│   │   ├── droid_policy.py         # DROID 数据集策略
│   │   └── libero_policy.py        # LIBERO 策略
│   ├── training/                   # 训练相关
│   │   ├── config.py               # 训练配置（核心）⭐
│   │   ├── data_loader.py          # 数据加载器
│   │   ├── optimizer.py            # 优化器配置
│   │   ├── checkpoints.py          # 检查点管理
│   │   └── weight_loaders.py       # 权重加载器
│   ├── shared/                     # 共享工具
│   │   ├── normalize.py            # 数据归一化
│   │   ├── image_tools.py          # 图像处理工具
│   │   └── download.py             # 模型/数据下载
│   ├── transforms.py               # 数据转换管道
│   └── serving/                    # 服务相关
├── scripts/                        # 脚本文件
│   ├── train.py                    # 训练脚本 ⭐
│   ├── process_data.py             # 数据处理脚本
│   └── serve_policy.py             # 策略服务脚本
├── examples/                       # 示例代码 ⭐
│   ├── aloha_real/                 # ALOHA 真实机器人示例
│   ├── aloha_sim/                  # ALOHA 仿真示例
│   ├── droid/                      # DROID 数据集示例
│   ├── libero/                     # LIBERO 示例
│   └── simple_client/              # 简单客户端示例
├── checkpoints/                    # 模型检查点
├── training_data/                  # 训练数据
├── processed_data/                 # 处理后的数据
├── pi_model.py                     # 模型封装类（用于部署）
├── deploy_policy.py                # 部署脚本
└── lerobot/                        # LeRobot 数据集工具
```

## 🏗️ 核心架构

### 1. 模型架构（Model Architecture）

PI0.5 采用**双专家模型架构**：

```
输入: 图像 + 语言提示 + 状态
  ↓
[PaliGemma] - 视觉语言模型（VLM）
  ├── SigLIP 视觉编码器（处理图像）
  └── Gemma 语言模型（处理文本）
  ↓
[Expert Gemma] - 动作专家模型
  ├── 接收 PaliGemma 的输出
  ├── 处理状态和动作序列
  └── 使用 AdaRMS（PI0.5 特有）
  ↓
输出: 动作序列 (action_horizon × action_dim)
```

**关键文件：**
- `src/openpi/models_pytorch/pi0_pytorch.py` - PI0.5 主模型类
- `src/openpi/models_pytorch/gemma_pytorch.py` - PaliGemma + Expert 模型

**核心组件：**
- `PI0Pytorch` 类：主模型类
  - `embed_prefix()`: 编码图像和语言提示
  - `embed_suffix()`: 编码状态和动作序列
  - `forward()`: 训练前向传播
  - `sample_actions()`: 推理时采样动作（使用扩散去噪）

### 2. 策略接口（Policy Interface）

**关键文件：** `src/openpi/policies/policy.py`

```python
Policy 类：
  - infer(obs) -> dict: 核心推理接口
    ├── 输入转换 (transforms)
    ├── 模型推理 (model.sample_actions)
    └── 输出转换 (output_transforms)
```

**数据流：**
```
原始观测 → 输入转换 → Observation 对象 → 模型 → 动作 → 输出转换 → 最终动作
```

### 3. 训练流程（Training Pipeline）

**关键文件：** `scripts/train.py`

训练流程：
1. 加载配置 (`training/config.py`)
2. 初始化模型和优化器
3. 数据加载 (`training/data_loader.py`)
4. 训练循环：
   - 前向传播：`model.forward(observation, actions)`
   - 计算损失：MSE loss (noise prediction)
   - 反向传播和优化

### 4. 数据管道（Data Pipeline）

**关键文件：** `src/openpi/transforms.py`

数据转换链：
```
原始数据 → RepackTransforms → DataTransforms → Normalize → ModelTransforms → 模型输入
```

**支持的机器人平台：**
- ALOHA (`policies/aloha_policy.py`)
- DROID (`policies/droid_policy.py`)
- LIBERO (`policies/libero_policy.py`)

## 🚀 快速上手指南

### 第一步：理解核心接口

1. **模型接口** (`src/openpi/models/model.py`)
   - `BaseModel`: 所有模型的基类
   - `Observation`: 观测数据结构
   - `Actions`: 动作数据结构

2. **策略接口** (`src/openpi/policies/policy.py`)
   - `Policy.infer(obs)`: 推理接口
   - 这是你与模型交互的主要接口

### 第二步：查看示例代码

**推荐阅读顺序：**

1. **简单示例** (`examples/simple_client/main.py`)
   - 最基础的推理示例
   - 展示如何加载模型和使用 Policy

2. **ALOHA 示例** (`examples/aloha_real/main.py` 或 `examples/aloha_sim/main.py`)
   - 完整的机器人控制流程
   - 展示如何与机器人交互

3. **模型封装** (`pi_model.py`)
   - 简化的模型封装类
   - 适合快速部署

### 第三步：理解模型架构

**重点文件：**

1. `src/openpi/models_pytorch/pi0_pytorch.py` (第 84-462 行)
   - `PI0Pytorch.__init__()`: 模型初始化
   - `PI0Pytorch.forward()`: 训练前向传播
   - `PI0Pytorch.sample_actions()`: 推理采样（扩散去噪过程）

2. `src/openpi/models_pytorch/gemma_pytorch.py`
   - `PaliGemmaWithExpertModel`: 双专家模型架构

### 第四步：理解配置系统

**关键文件：** `src/openpi/training/config.py`

配置层次：
- `TrainConfig`: 训练配置
  - `model`: 模型配置
  - `data`: 数据配置
  - `optimizer`: 优化器配置

**预定义配置：**
- 在 `config.py` 的 `_CONFIGS` 字典中
- 通过 `get_config(name)` 获取

### 第五步：运行示例

```bash
# 1. 查看训练脚本
cat scripts/train.py

# 2. 查看推理示例
cat examples/simple_client/main.py

# 3. 查看部署脚本
cat deploy_policy.py
```

## 🔑 关键概念

### 1. 扩散模型（Diffusion Model）
- PI0.5 使用扩散模型生成动作序列
- `sample_actions()` 实现去噪过程（Euler 步进）
- 时间步从 1.0 到 0.0

### 2. 注意力机制（Attention Masks）
- `make_att_2d_masks()`: 创建注意力掩码
- 图像和语言可以互相注意
- 动作序列只能注意前缀（图像+语言）

### 3. AdaRMS（PI0.5 特有）
- 自适应 RMS 归一化
- 在 Expert Gemma 中使用
- 通过时间嵌入条件化

### 4. 数据归一化
- 训练时计算归一化统计量
- 存储在 `checkpoints/.../assets/` 中
- 推理时自动加载和应用

## 📝 代码阅读建议

### 入门路径（按优先级）

1. **理解接口** (30分钟)
   - `src/openpi/policies/policy.py` - Policy 类
   - `src/openpi/models/model.py` - Observation/Actions 数据结构
   - `examples/simple_client/main.py` - 简单示例

2. **理解模型** (1-2小时)
   - `src/openpi/models_pytorch/pi0_pytorch.py` - 主模型
   - `src/openpi/models_pytorch/gemma_pytorch.py` - 双专家架构
   - 重点关注 `forward()` 和 `sample_actions()` 方法

3. **理解训练** (1小时)
   - `scripts/train.py` - 训练脚本
   - `src/openpi/training/config.py` - 配置系统
   - `src/openpi/training/data_loader.py` - 数据加载

4. **理解部署** (30分钟)
   - `pi_model.py` - 模型封装
   - `deploy_policy.py` - 部署脚本
   - `examples/aloha_real/main.py` - 真实机器人示例

### 调试建议

1. **从简单开始**：先运行 `examples/simple_client/main.py`
2. **打印中间结果**：在 `Policy.infer()` 中添加打印
3. **检查数据格式**：确保输入符合 `Observation` 格式
4. **查看配置**：检查 `checkpoints/.../config.json`

## 🔍 关键文件速查

| 文件 | 作用 | 重要性 |
|------|------|--------|
| `models_pytorch/pi0_pytorch.py` | PI0.5 主模型 | ⭐⭐⭐ |
| `policies/policy.py` | 策略接口 | ⭐⭐⭐ |
| `policies/policy_config.py` | 模型加载 | ⭐⭐⭐ |
| `training/config.py` | 配置系统 | ⭐⭐ |
| `models/model.py` | 数据结构定义 | ⭐⭐ |
| `scripts/train.py` | 训练脚本 | ⭐⭐ |
| `examples/simple_client/main.py` | 简单示例 | ⭐⭐ |
| `pi_model.py` | 模型封装 | ⭐ |

## 💡 常见问题

1. **JAX vs PyTorch？**
   - PI0.5 使用 PyTorch 实现（`models_pytorch/`）
   - 旧版本 PI0 使用 JAX（`models/`）

2. **如何加载模型？**
   - 使用 `policy_config.create_trained_policy()`
   - 需要提供 checkpoint 路径和配置名称

3. **如何添加新的机器人平台？**
   - 在 `policies/` 下创建新的 policy 文件
   - 实现数据转换逻辑

4. **如何修改模型架构？**
   - 修改 `models_pytorch/pi0_pytorch.py`
   - 注意保持接口兼容性

## 📚 下一步

1. 运行一个简单示例，验证环境
2. 阅读 `pi0_pytorch.py` 的 `forward()` 方法，理解训练流程
3. 阅读 `sample_actions()` 方法，理解推理流程
4. 查看训练脚本，理解如何训练新模型
5. 尝试修改配置，训练自己的模型

