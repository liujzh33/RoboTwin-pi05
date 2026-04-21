# 使用多帧（历史帧）权重的 Eval 命令说明

训练多帧模型后，eval 时需要改用**多帧对应的 train_config_name 和 model_name**，这样会：
1. 加载正确的模型配置（`n_obs_steps=4`）
2. 从多帧 checkpoint 目录加载权重
3. 在推理时使用 policy 的历史帧 buffer

## beat_block_hammer 任务（多帧权重）

**原单帧命令：**
```bash
bash eval.sh beat_block_hammer demo_randomized pi05_aloha_full_base beat_hammer_pi05_200_initial 0 4 3001
```

**改为多帧命令：**
```bash
bash eval.sh beat_block_hammer demo_randomized pi05_aloha_full_base_multiframe_beat beat_block_hammer_multiframe_2f 0 4 30001
```

参数说明：
- `train_config_name`: `pi05_aloha_full_base_multiframe_beat`（多帧训练配置）
- `model_name`: `beat_block_hammer_multiframe_2f`（多帧实验目录名）
- `checkpoint_id`: `30001`（你当前多帧 checkpoint 的步数，可按需改为 26000/27000/28000/29000）

## blocks_ranking_rgb 任务（多帧权重）

```bash
bash eval.sh blocks_ranking_rgb demo_randomized pi05_aloha_full_base_multiframe_blocks blocks_ranking_rgb_multiframe_2f 0 4 30001
```

- `train_config_name`: `pi05_aloha_full_base_multiframe_blocks`
- `model_name`: `blocks_ranking_rgb_multiframe_2f`
- `checkpoint_id`: `30001`（或 24000/25000/.../30001 中任意已有步数）

## 路径与 config 对应关系

- Eval 从 **RoboTwin 项目根目录** 运行（`eval.sh` 里会 `cd ../..`）。
- 权重路径：`policy/pi05/checkpoints/{train_config_name}/{model_name}/{checkpoint_id}/`
- 配置通过 `train_config_name` 从 `openpi.training.config.get_config(...)` 加载，因此使用 `pi05_aloha_full_base_multiframe_beat` / `pi05_aloha_full_base_multiframe_blocks` 会自动启用 `n_obs_steps=4` 与历史帧推理。

## 使用其他 checkpoint 步数

若想用非最新的 checkpoint，把最后一个参数改为对应步数即可，例如：
- beat: `26000` / `27000` / `28000` / `29000` / `30001`
- blocks: `24000` / `25000` / `26000` / `27000` / `28000` / `29000` / `30001`
