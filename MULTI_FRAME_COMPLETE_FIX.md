# 多帧训练完整修复文档

## 问题概述

在实现多帧历史帧堆叠训练时，需要修改整个训练流程中的多个组件，使其能够处理多帧输入。本文档详细说明了所有需要修改的地方。

## 训练流程概览

```
LeRobotDataset (with delta_timestamps)
  ↓
RepackTransform (映射键名)
  ↓
FrameStack Transform (堆叠历史帧)
  ↓
Data Transforms (AlohaInputs, DeltaActions等)
  ↓
Normalize Transform
  ↓
Model Transforms
  ↓
Model Forward (embed_prefix)
```

## 修复清单

### 1. 数据加载器 (`data_loader.py`)

#### 1.1 `create_torch_dataset`
**修改内容：**
- 检测 `n_obs_steps > 1`
- 为所有相机键和 state 添加 `delta_timestamps`
- 使用负时间戳获取历史帧：`[-(n_obs_steps-1), ..., -1, 0] / fps`

**关键代码：**
```python
if n_obs_steps > 1:
    camera_keys = dataset_meta.camera_keys
    for cam_key in camera_keys:
        delta_timestamps[cam_key] = [-(n_obs_steps - 1 - t) / dataset_meta.fps for t in range(n_obs_steps)]
    delta_timestamps["observation.state"] = [-(n_obs_steps - 1 - t) / dataset_meta.fps for t in range(n_obs_steps)]
```

#### 1.2 `transform_dataset`
**修改内容：**
- 添加 `model_config` 参数
- 在 repack 之后添加 `FrameStack` transform
- 从 repack transform 结构中提取图像键

### 2. FrameStack Transform (`transforms.py`)

**新增 Transform：**
- 处理从 `delta_timestamps` 获取的历史帧
- 将图像从 `(n_frames, c, h, w)` 转换为 `(n_frames, h, w, c)`
- 从 padding 标志创建图像 mask
- 确保 state 形状为 `(n_frames, dim)`

**关键功能：**
- 处理 repack 后的嵌套 `images` 字典
- 处理 padding 信息（`*_is_pad` 键）
- 创建 `image_masks` 字典

### 3. DeltaActions Transform (`transforms.py`)

**修改内容：**
- 支持多帧 state 输入 `(n_frames, dim)`
- 使用最后一帧（当前帧）计算 delta actions
- 修复广播形状问题

**关键代码：**
```python
if state.ndim == 2:
    # Multi-frame: use last frame
    current_state = state[-1]  # Shape: (dim,)
else:
    # Single-frame
    current_state = state

state_delta = np.where(mask, current_state[:dims], 0)
actions[..., :dims] -= state_delta  # Correct broadcasting
```

### 4. AbsoluteActions Transform (`transforms.py`)

**修改内容：**
- 与 `DeltaActions` 相同的多帧支持
- 使用最后一帧计算绝对 actions

### 5. AlohaInputs Transform (`aloha_policy.py`)

#### 5.1 `_decode_state`
**修改内容：**
- 支持单帧 `(dim,)` 和多帧 `(n_frames, dim)`
- 多帧时对每一帧应用相同的解码逻辑

**关键代码：**
```python
if state.ndim == 2:
    # Multi-frame: apply to each frame
    decoded_state = np.zeros_like(state)
    for i in range(state.shape[0]):
        decoded_state[i] = decode_single_frame(state[i])
    return decoded_state
else:
    # Single-frame
    return decode_single_frame(state)
```

#### 5.2 `_decode_aloha` 中的 `convert_image`
**修改内容：**
- 支持单帧和多帧图像
- 自动检测格式并转换
- 处理 `(n_frames, c, h, w)` 和 `(n_frames, h, w, c)` 格式

### 6. 模型 Forward (`pi05.py`)

#### 6.1 `embed_prefix`
**修改内容：**
- 添加健壮性检查，处理 4D 和 5D 输入
- 如果检测到 4D 输入但处于多帧模式，自动扩展为 5D

**关键代码：**
```python
if image.ndim == 4:
    # Single-frame input in multi-frame mode: expand to 5D
    image = image[:, np.newaxis, ...]  # [batch, 1, h, w, c]
    image_mask = image_mask[:, np.newaxis]  # [batch, 1]
```

## 数据形状变化

### 单帧模式 (n_obs_steps=1)
- Images: `(batch, h, w, c)`
- State: `(batch, dim)`
- Actions: `(batch, action_horizon, action_dim)`

### 多帧模式 (n_obs_steps>1)
- Images: `(batch, n_obs_steps, h, w, c)`
- State: `(batch, n_obs_steps, dim)` (但某些 transforms 只使用最后一帧)
- Actions: `(batch, action_horizon, action_dim)` (不变)

## Transform 执行顺序

1. **RepackTransform**: 映射数据集键到模型键
2. **FrameStack**: 堆叠历史帧（仅在 `n_obs_steps > 1` 时）
3. **AlohaInputs**: 解码 Aloha 格式数据
4. **DeltaActions**: 将绝对 actions 转换为 delta（使用最后一帧 state）
5. **Normalize**: 归一化数据
6. **Model Transforms**: 模型特定的 transforms

## 关键设计决策

### 1. 为什么 DeltaActions 使用最后一帧？
Delta actions 是相对于**当前状态**的，所以应该使用最后一帧（当前帧）的 state 来计算。

### 2. 为什么 FrameStack 在 Repack 之后？
Repack 将数据集键映射到模型键，FrameStack 需要访问映射后的 `images` 字典结构。

### 3. 为什么需要处理多种图像格式？
不同阶段的数据可能有不同的格式：
- LeRobotDataset: `(n_frames, c, h, w)` (PyTorch 格式)
- FrameStack 后: `(n_frames, h, w, c)` (模型格式)
- AlohaInputs 需要处理两种格式

## 测试建议

1. **单帧模式测试** (`n_obs_steps=1`):
   - 确保所有 transforms 仍然正常工作
   - 验证向后兼容性

2. **多帧模式测试** (`n_obs_steps=2`):
   - 检查数据形状是否正确
   - 验证 delta actions 计算是否正确
   - 确认模型能正确处理多帧输入

3. **边界情况测试**:
   - Episode 开始时的 padding 处理
   - 不同 batch size 的处理
   - 不同 action horizon 的处理

## 常见问题

### Q: 为什么 state 是多帧但 actions 不是？
A: Actions 是未来动作序列，不需要历史帧。只有 observations（images 和 state）需要历史帧来提供时间上下文。

### Q: 如何处理 episode 开始时的 padding？
A: FrameStack transform 会从 `*_is_pad` 键创建 mask，标记哪些帧是有效的。模型会使用这些 mask 来忽略 padding 的帧。

### Q: 多帧会增加多少内存？
A: 对于 `n_obs_steps=2`，图像内存会增加约 2 倍。但这是必要的，因为模型需要历史信息来做出更好的决策。

## 总结

多帧训练的实现涉及整个训练流程的修改：
1. 数据加载：获取历史帧
2. Transforms：处理多帧数据
3. 模型：处理多帧输入

所有修改都保持了向后兼容性（`n_obs_steps=1` 时行为不变），并且遵循了常见的最佳实践（使用最后一帧计算 delta actions）。
