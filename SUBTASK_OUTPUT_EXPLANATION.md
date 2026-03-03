# Eval 时子任务（Subtask）输出说明

## 直接使用 JAX 权重即可

Eval 时**无需**换成 PyTorch 格式，**直接使用 JAX checkpoint**（仅含 `params/`、`train_state/` 的目录）即可。

- **JAX 路径**：`src/openpi/models/pi05.py` 中的 `sample_subtask()` 会在推理时对 “Subtask: ” 后的 token 做自回归解码。
- **PyTorch 路径**：若使用带 `model.safetensors` 的目录，则由 `models_pytorch/pi0_pytorch.py` 的 `sample_subtask()` 解码。

两种格式都会在 `policy.infer()` 中返回 `subtask_text` 和 `debug_tokens`，`deploy_policy.eval()` 会打印 `[Pi0.5 Subtask]: ...` 和 `[Debug Tokens]: ...`。

## 相关代码位置

- `tokenizer.py`：`tokenize_subtask_prefix`
- `transforms.py`：TokenizePrompt 输出 subtask 前缀
- `policy.py`：根据是否为 PyTorch 调用 `model.sample_subtask(device, obs, ...)` 或 `model.sample_subtask(obs, ...)`，并 detokenize
- `policy_config.py`：在 metadata 中注入 tokenizer
- `deploy_policy.py` / `pi_model.py`：打印 `_last_infer` 中的 subtask 与 debug_tokens

## 若仍无子任务输出

请确认：

1. **用的是本地代码**：`pi_model.py` 已把 `policy/pi05/src` 加入 `sys.path`，保证 openpi 来自 `policy/pi05/src`。
2. **config 带 state**：当前 train config（如 `pi05_aloha_full_base_multiframe_beat`）使用 `discrete_state_input=True`，TokenizePrompt 会产出 `tokenized_prompt_subtask_prefix`。
3. **eval 命令**：例如  
   `bash eval.sh beat_block_hammer demo_randomized pi05_aloha_full_base_multiframe_beat beat_block_hammer_multiframe_2f 0 <gpu_id> <checkpoint_id>`  
   其中 `checkpoint_id` 例如 `30001`，对应目录  
   `policy/pi05/checkpoints/pi05_aloha_full_base_multiframe_beat/beat_block_hammer_multiframe_2f/30001`（JAX 或 PyTorch 均可）。

若仍无 subtask 行，deploy 会打印：  
`[Pi0.5 Subtask]: (use PyTorch checkpoint with model.safetensors to enable)`  
仅当模型未提供 `sample_subtask`（例如旧版代码）时会出现；当前 JAX 与 PyTorch 均已支持。
