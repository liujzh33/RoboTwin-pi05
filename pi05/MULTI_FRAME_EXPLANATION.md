# 多帧堆叠机制详解

本文档详细解释pi05模型中单帧和多帧模式下observation的处理方式。

## 1. 单帧模式（n_obs_steps=1，默认）

### 输入格式
```python
obs.images = {
    "base_0_rgb": [batch, 224, 224, 3],      # 顶部相机
    "left_wrist_0_rgb": [batch, 224, 224, 3],  # 左腕相机
    "right_wrist_0_rgb": [batch, 224, 224, 3] # 右腕相机
}
```

### 处理流程

#### Step 1: 图像编码（分别处理每个视角）
```python
# 代码位置: pi05.py embed_prefix() 第145-157行
for name in obs.images:
    # 对每个视角的图像分别调用图像编码器
    image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
    # image_tokens shape: [batch, num_tokens_per_image, emb_dim]
    # 对于224x224图像，patch_size=16，num_tokens_per_image = (224/16)^2 = 196
    tokens.append(image_tokens)
```

**图像编码器（SigLIP ViT）的工作原理**：
- 将图像分成16x16的patches
- 每个patch通过卷积层转换为一个token
- 224x224图像 → 14x14 = 196个tokens
- 每个token的维度 = emb_dim (通常是768或1024)

#### Step 2: 多视角tokens拼接
```python
# 代码位置: pi05.py embed_prefix() 第166行
tokens = jnp.concatenate(tokens, axis=1)
# 最终shape: [batch, 3*196, emb_dim] = [batch, 588, emb_dim]
```

**拼接顺序**：
```
[base_0_rgb的196个tokens] + [left_wrist_0_rgb的196个tokens] + [right_wrist_0_rgb的196个tokens]
```

#### Step 3: 添加语言tokens
```python
# 代码位置: pi05.py embed_prefix() 第160-165行
tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
tokens.append(tokenized_inputs)
tokens = jnp.concatenate(tokens, axis=1)
# 最终shape: [batch, 588 + language_tokens, emb_dim]
```

### 最终token序列结构
```
[图像tokens (588个)] + [语言tokens (N个)]
├─ base_0_rgb: 196 tokens
├─ left_wrist_0_rgb: 196 tokens  
├─ right_wrist_0_rgb: 196 tokens
└─ language: N tokens
```

---

## 2. 多帧模式（n_obs_steps > 1）

### 输入格式
```python
obs.images = {
    "base_0_rgb": [batch, n_obs_steps, 224, 224, 3],      # 例如 n_obs_steps=3
    "left_wrist_0_rgb": [batch, n_obs_steps, 224, 224, 3],
    "right_wrist_0_rgb": [batch, n_obs_steps, 224, 224, 3]
}
```

### 处理流程

#### Step 1: 展平时间维度，批量处理所有帧
```python
# 代码位置: pi05.py embed_prefix() 第113-120行
for name in obs.images:
    image = obs.images[name]  # [batch, n_obs_steps, h, w, c]
    
    # 将时间维度展平到batch维度
    image_flat = einops.rearrange(image, "b t h w c -> (b t) h w c")
    # image_flat shape: [batch * n_obs_steps, h, w, c]
    # 例如: [batch*3, 224, 224, 3]
    
    # 一次性处理所有帧（并行）
    image_tokens_flat, _ = self.PaliGemma.img(image_flat, train=False)
    # image_tokens_flat shape: [batch * n_obs_steps, 196, emb_dim]
    # 例如: [batch*3, 196, emb_dim]
```

**关键点**：
- 所有历史帧的图像**并行**通过编码器
- 不是相加，而是分别编码后拼接

#### Step 2: 重组为时间序列tokens
```python
# 代码位置: pi05.py embed_prefix() 第124-130行
num_tokens_per_image = image_tokens_flat.shape[1]  # 196
emb_dim = image_tokens_flat.shape[2]  # 768或1024

image_tokens = einops.rearrange(
    image_tokens_flat, 
    "(b t) s d -> b (t s) d", 
    b=batch_size, 
    t=num_frames,  # n_obs_steps
    s=num_tokens_per_image  # 196
)
# image_tokens shape: [batch, n_obs_steps * 196, emb_dim]
# 例如: [batch, 3*196, emb_dim] = [batch, 588, emb_dim]
```

**重组逻辑**：
```
输入: [batch*3, 196, emb_dim]
      ├─ frame_0: 196 tokens
      ├─ frame_1: 196 tokens  
      └─ frame_2: 196 tokens

输出: [batch, 588, emb_dim]
      ├─ frame_0的所有196个tokens
      ├─ frame_1的所有196个tokens
      └─ frame_2的所有196个tokens
```

#### Step 3: 多视角拼接（与单帧模式相同）
```python
tokens.append(image_tokens)  # 每个视角的tokens
# 最终拼接所有视角
tokens = jnp.concatenate(tokens, axis=1)
```

### 最终token序列结构（n_obs_steps=3示例）
```
[图像tokens (3*588=1764个)] + [语言tokens (N个)]
├─ base_0_rgb: 588 tokens (3帧 × 196 tokens/帧)
│  ├─ frame_t-2: 196 tokens
│  ├─ frame_t-1: 196 tokens
│  └─ frame_t: 196 tokens
├─ left_wrist_0_rgb: 588 tokens (3帧 × 196 tokens/帧)
│  ├─ frame_t-2: 196 tokens
│  ├─ frame_t-1: 196 tokens
│  └─ frame_t: 196 tokens
├─ right_wrist_0_rgb: 588 tokens (3帧 × 196 tokens/帧)
│  ├─ frame_t-2: 196 tokens
│  ├─ frame_t-1: 196 tokens
│  └─ frame_t: 196 tokens
└─ language: N tokens
```

---

## 3. 关键区别总结

| 特性 | 单帧模式 | 多帧模式 |
|------|---------|---------|
| **输入shape** | `[batch, h, w, c]` | `[batch, n_obs_steps, h, w, c]` |
| **处理方式** | 每个视角分别编码 | 所有帧并行编码后重组 |
| **每个视角tokens数** | 196 | `n_obs_steps * 196` |
| **总图像tokens数** | 588 (3视角×196) | `3 * n_obs_steps * 196` |
| **时间信息** | 无 | 通过token顺序体现 |

## 4. Attention机制

所有tokens（图像+语言）都会进入Transformer的attention层：
- **图像tokens之间**：可以互相attention（ar_mask=False）
- **历史帧tokens之间**：可以互相attention，模型可以学习时序关系
- **图像与语言tokens**：可以互相attention

## 5. 代码位置

- **单帧处理**: `src/openpi/models/pi05.py` 第142-157行
- **多帧处理**: `src/openpi/models/pi05.py` 第108-141行
- **图像编码器**: `src/openpi/models/siglip.py` 第188-290行
- **Token拼接**: `src/openpi/models/pi05.py` 第166行

---

## 6. 与其他方法的对比

### 6.1 我们实现的方式（Token Concatenation）

**特点**：
- 每个历史帧分别编码成tokens
- 在sequence维度拼接所有帧的tokens
- 通过Transformer的attention机制学习时序关系

**优点**：
- ✅ 保留完整的空间-时间信息（每个帧的每个patch都有独立的token）
- ✅ Transformer可以灵活学习帧间关系（通过attention）
- ✅ 适合视觉-语言模型架构（如PaliGemma）
- ✅ 可以处理变长历史（理论上）

**缺点**：
- ❌ Token数量线性增长：`n_obs_steps × num_tokens_per_image`
- ❌ 计算和内存开销较大（attention复杂度O(n²)）
- ❌ 对于长历史（>5帧）可能效率较低

### 6.2 Diffusion Policy的方式（Feature Concatenation）

**特点**：
- 每个历史帧通过CNN/ViT编码成固定维度的特征向量
- 在特征维度拼接：`[feat_t-2, feat_t-1, feat_t]`
- 最终flatten成单一特征向量

**代码示例**（来自diffusion policy）：
```python
# 每个帧编码成特征
img_features = self.rgb_encoder(images)  # [B, n_obs_steps, num_cameras, H, W, C]
# -> [B, n_obs_steps, feat_dim]

# 拼接并flatten
global_cond = torch.cat([state_feat, img_feat], dim=-1).flatten(start_dim=1)
# -> [B, n_obs_steps * (state_dim + feat_dim)]
```

**优点**：
- ✅ 特征维度固定，不随历史长度线性增长太多
- ✅ 计算效率较高（CNN编码 + 简单拼接）
- ✅ 适合CNN-based架构

**缺点**：
- ❌ 丢失了空间细节（每个帧压缩成单个特征向量）
- ❌ 时序关系需要模型显式学习（没有attention机制）
- ❌ 不适合需要细粒度空间信息的任务

### 6.3 其他方法

**1. Temporal Convolution (TCN)**：
- 使用1D卷积在时间维度上滑动
- 适合固定长度的短历史（2-5帧）

**2. LSTM/GRU**：
- 递归处理历史帧
- 适合长历史，但难以并行化

**3. Past-Token Prediction (2024新方法)**：
- 显式监督模型学习历史动作依赖
- 通过预测过去和未来动作来正则化

### 6.4 方法选择建议

| 方法 | 适用场景 | 历史长度 | 计算开销 |
|------|---------|---------|---------|
| **Token Concatenation (我们的)** | 视觉-语言模型，需要细粒度空间信息 | 2-5帧 | 高 |
| **Feature Concatenation** | CNN-based策略，快速推理 | 2-10帧 | 中 |
| **Temporal Convolution** | 固定短历史，实时控制 | 2-5帧 | 低 |
| **LSTM/GRU** | 长历史依赖，离线训练 | 10+帧 | 中 |

### 6.5 我们的方式是否常用？

**答案：是的，这是当前视觉-语言-动作模型的主流方式**

**原因**：
1. **Transformer架构的天然优势**：Vision-Language模型（如PaliGemma、CLIP）都使用Transformer，token concatenation是最自然的扩展方式
2. **细粒度信息保留**：相比feature concatenation，保留了每个patch的空间信息
3. **灵活的时间建模**：通过attention机制可以学习复杂的时序关系
4. **与单帧模式一致**：多帧模式是单帧模式的直接扩展，代码简洁

**类似实现**：
- **OpenVLA**: 使用类似的token拼接方式处理多帧
- **RT-1/RT-2**: 使用sequence-to-sequence架构，历史帧作为sequence输入
- **ACT**: 虽然使用CNN，但也将历史帧在时间维度堆叠

**限制**：
- 主要限制是计算和内存开销
- 通常用于2-5帧的历史（我们默认n_obs_steps=1，可扩展到2-3帧）
- 对于更长历史（>10帧），可能需要其他方法或优化

### 6.6 实际应用建议

1. **从单帧开始**（n_obs_steps=1）：验证模型基本功能
2. **逐步增加**（n_obs_steps=2-3）：如果任务需要时序信息
3. **监控性能**：注意内存和推理时间
4. **考虑混合方法**：对于长历史，可以先用CNN压缩再拼接
