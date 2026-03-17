# RoboTwin-pi05 v2.5

只实现了加入进度检测的训练（使用非错误恢复数据的），但是还没有实现整体的训练框架。

## 主要更新

- **进度检测 (Progress Estimation)**：在 Pi0.5 模型中新增 `progress_mlp_in` 和 `progress_mlp_out` 层，用于预测任务执行进度
- **训练配置**：`pi05_aloha_full_base_multi_task_5_v1_0_progress` 使用 `ComputeProgressLabel` 动态生成进度标签
- **数据**：当前使用非错误恢复数据（pi05_multi_task_5_v1.0）进行训练

## 待完成

- 整体训练框架（含错误恢复数据）
- 完整的 recovery 流程集成
