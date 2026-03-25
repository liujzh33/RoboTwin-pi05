# RvS 权重评估说明

## 架构变化

- RvS 权重是在 **recovery2 的 openpi** 上训练的，与 pi05 的 openpi 有差异：
  - **models_pytorch**：`pi0_pytorch.py` 增加 `reward_mlp_in` / `reward_mlp_out`，推理时必须提供 **reward** 输入。
  - **models_pytorch**：`preprocessing_pytorch.py` 中 observation 支持 `reward=getattr(observation, "reward", None)`。
  - **models/model.py**：`Observation` 增加字段 `reward: at.Float[ArrayT, "*b 1"] | None = None`。

因此不能再用「只换权重路径、继续用 pi05 的 openpi」的方式做 eval，必须：

1. **用 recovery2 的 openpi**（通过 `PYTHONPATH` 指向 `policy/recovery2/openpi-main/src`）。
2. **用 recovery2 的 norm_stats**：`policy/recovery2/openpi-main/norm_stats.json`。
3. **在推理时给 observation 提供 reward**：eval 时无真实 reward，统一传 `0.0`。

## 权重的读取方式

- 权重目录：`/data2/liangxiwen/zkd/cry/RoboTwin/policy/recovery2/openpi-main/RvS/10000`（内含 `model.safetensors`、`metadata.pt` 等）。
- 仍通过 **PyTorch checkpoint** 方式加载（`create_trained_policy` 检测到 `model.safetensors` 后走 `load_pytorch`）。
- 使用 **recovery2 的** `policy_config.create_trained_policy` 和 `train_config_name="pi05_rl_aloha_reward"`，这样会实例化带 reward 分支的模型，并与 RvS 权重结构一致。

## Eval 的修改

1. **新 policy 包 `policy/pi05_rvs`**
   - `get_model`：用 recovery2 的 openpi 创建 policy（需 `torch_checkpoint_dir`、`norm_stats_path`，且 `norm_stats` 必传，因 RvS 目录下无 `assets/`）。
   - `pi_model_rvs.PI0_RvS`：在 `update_observation_window` 里给 `observation_window["reward"] = np.array(0.0, dtype=np.float32)`，满足推理接口。
   - `eval` / `reset_model`：与 pi05 相同，只是 model 换成了 `PI0_RvS`。

2. **运行方式**
   - 必须设置 **PYTHONPATH**，让本次进程内 `openpi` 来自 recovery2：
     ```bash
     export PYTHONPATH=/data2/liangxiwen/zkd/cry/RoboTwin/policy/recovery2/openpi-main/src:$PYTHONPATH
     ```
   - 使用脚本（内部已设 `PYTHONPATH` 和上述参数）：
     ```bash
     bash /data2/liangxiwen/zkd/cry/RoboTwin/script/run_eval_rvs.sh [任务名] [环境] [GPU] [条数]
     ```
   - 或手动调用 `eval_policy_initial_motus.py` 时传入：
     - `--policy_name pi05_rvs`
     - `--train_config_name pi05_rl_aloha_reward`
     - `--torch_checkpoint_dir .../policy/recovery2/openpi-main/RvS/10000`
     - `--norm_stats_path .../policy/recovery2/openpi-main/norm_stats.json`
     - 并保证同一进程内 `PYTHONPATH` 已包含 recovery2 的 `src`。

3. **结果目录**
   - 使用 `ckpt_setting=rvs_10000`，结果写入：`eval_result/<task_name>/pi05_rvs/<task_config>/rvs_10000/<时间>/`。

## 小结

| 项目           | pi05 原版 eval                    | RvS eval（当前实现）                          |
|----------------|-----------------------------------|-----------------------------------------------|
| openpi 代码    | `policy/pi05/src/openpi`         | `policy/recovery2/openpi-main/src/openpi`     |
| 模型结构       | 无 reward 输入                    | 有 reward 输入（reward_mlp）                  |
| 权重目录       | 如 `checkpoint_torch/motus` 等    | `policy/recovery2/openpi-main/RvS/10000`      |
| norm_stats     | pi05 的 assets 或 norm_stats.json | `policy/recovery2/openpi-main/norm_stats.json`|
| policy 包      | `pi05`                            | `pi05_rvs`                                    |
| train_config   | `pi05_aloha_full_base_all`        | `pi05_rl_aloha_reward`                        |
| observation    | state, images, prompt             | 同上，且 **必须带 reward**（eval 时传 0.0）   |

所以：**架构改了，权重的读取方式要配合 recovery2 的 openpi 和 config；eval 也要改**——用 `pi05_rvs` + `run_eval_rvs.sh`（或等价参数 + PYTHONPATH）即可。
