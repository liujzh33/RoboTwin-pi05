# Multi-Frame Training Fix

## Problem

When training with multi-frame stacking (`n_obs_steps > 1`), the training data loader was not providing historical frames. The model's `embed_prefix` method expected 5D tensors `[batch, n_obs_steps, h, w, c]` but received 4D tensors `[batch, h, w, c]`, causing an `EinopsError`.

## Solution

### 1. Data Loader Changes (`data_loader.py`)

**Modified `create_torch_dataset`:**
- Added logic to detect `n_obs_steps > 1` from model config
- When multi-frame is enabled, adds `delta_timestamps` for:
  - All camera keys from dataset metadata
  - State (`observation.state`)
- Uses negative timestamps to fetch past frames: `[-(n_obs_steps-1), ..., -1, 0] / fps`

**Modified `transform_dataset`:**
- Added `model_config` parameter
- After repack transforms, adds `FrameStack` transform if `n_obs_steps > 1`
- Extracts image keys from repack transform structure

### 2. FrameStack Transform (`transforms.py`)

**New `FrameStack` transform:**
- Processes data AFTER repack transform
- Converts images from `(n_frames, c, h, w)` to `(n_frames, h, w, c)`
- Creates image masks from padding flags (`*_is_pad` keys)
- Handles nested `images` dictionary structure
- Ensures state has shape `(n_frames, dim)`

### 3. Model Changes (`pi05.py`)

**Modified `embed_prefix`:**
- Added robustness to handle both 4D and 5D inputs
- If 4D input detected in multi-frame mode, expands to 5D automatically
- This handles edge cases where data might not be properly stacked

## How It Works

1. **Data Loading:**
   - `LeRobotDataset` with `delta_timestamps` loads multiple frames per sample
   - Returns tensors with shape `(n_frames, ...)` for images and state

2. **Repack Transform:**
   - Maps dataset keys to model keys: `observation.images.*` → `images.*`

3. **FrameStack Transform:**
   - Converts PyTorch format `(n_frames, c, h, w)` to model format `(n_frames, h, w, c)`
   - Creates masks from padding information
   - Stores masks in `image_masks` dictionary

4. **Data Transforms:**
   - Normalize, resize, etc. are applied
   - Final data has shape `[batch, n_obs_steps, h, w, c]` for images

5. **Model Forward:**
   - `embed_prefix` receives 5D tensors `[batch, n_obs_steps, h, w, c]`
   - Flattens to `[batch * n_obs_steps, h, w, c]` for image encoder
   - Reshapes tokens to `[batch, n_obs_steps * num_tokens, emb_dim]`
   - Concatenates tokens from all frames and cameras

## Usage

To enable multi-frame training, set `n_obs_steps` in your model config:

```python
config = pi0_config.Pi0Config(pi05=True, n_obs_steps=2)  # 2-frame history
```

The data loader will automatically:
- Fetch historical frames from the dataset
- Stack them into the correct format
- Create appropriate masks

## Common Issues and Solutions

### Issue: `EinopsError: Wrong shape: expected 5 dims. Received 4-dim tensor`

**Cause:** Data loader not providing multi-frame data, or FrameStack transform not applied.

**Solution:** 
- Ensure `n_obs_steps > 1` in model config
- Check that `delta_timestamps` are set in `create_torch_dataset`
- Verify FrameStack transform is in the transform pipeline

### Issue: Padding masks not working

**Cause:** Padding keys (`*_is_pad`) might have different names after repack.

**Solution:** FrameStack checks multiple key formats:
- `observation.images.{key}_is_pad`
- `{key}_is_pad`

If padding info is missing, all frames are assumed valid.

## Testing

After making these changes, training should work with multi-frame inputs. The model will:
1. Receive properly stacked historical frames
2. Process them through the image encoder
3. Concatenate tokens from all frames
4. Use self-attention to model temporal relationships

## Notes

- The FrameStack transform must run AFTER repack transforms (to access nested `images` dict)
- FrameStack must run BEFORE data transforms (normalization, etc.)
- The transform pipeline order is: repack → FrameStack → data transforms → normalize → model transforms
