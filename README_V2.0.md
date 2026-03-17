# PI05 策略 — 分支 v2.0

本分支（v2.0）在 v1.0 基础上做了以下更新：

- **数据**：每个任务同时使用 **50 条干净数据**（aloha-agilex_clean_50）和 **500 条随机化数据**（aloha-agilex_randomized_500），共 10 组 processed 数据参与训练。
- **子任务划分**：子任务划分更加细致（如 beat_block_hammer 4 阶段，blocks_ranking 13 阶段，click 类 3 阶段），并配合语言丰富化（CycleVLA 风格、每阶段多种表述）。
- **任务与训练**：在 **5 个任务**（beat_block_hammer、blocks_ranking_rgb、blocks_ranking_size、click_alarmclock、click_bell）上联合训练，数据集为 `pi05_multi_task_5_v1.0`，配置见 `pi05_aloha_full_base_multi_task_5_v1_0`。

本仓库不包含超过 100MB 的单个文件（如 checkpoint、缓存、训练数据等），请自行在本地或其它存储准备。
