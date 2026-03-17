# 批量评估 Motus 权重脚本使用说明

## 概述

这个脚本用于批量运行50个任务的评估，使用 motus 权重，每个任务测试50次。

## 文件说明

1. **`eval_policy_initial_motus.py`**: 评估脚本（已修改支持 `torch_checkpoint_dir` 和 `test_num` 参数）
2. **`batch_eval_motus_50_tasks.sh`**: 单GPU批量运行脚本（遍历50个任务，处理错误）
3. **`batch_eval_motus_parallel.sh`**: 并行批量运行脚本（支持指定GPU和任务范围）
4. **`run_batch_eval_motus.sh`**: 单GPU快速启动脚本（使用 nohup 后台运行）
5. **`run_batch_eval_motus_parallel.sh`**: 并行快速启动脚本（同时启动GPU 0和GPU 1）

## 使用方法

### ⚡ 方法1：并行运行（推荐，使用 GPU 0 和 GPU 1，提速2倍）

```bash
cd /data2/liangxiwen/zkd/cry/RoboTwin
bash script/run_batch_eval_motus_parallel.sh
```

这会：
- **GPU 0**: 运行任务 0-24 (25个任务)
- **GPU 1**: 运行任务 25-49 (25个任务)
- 两个GPU并行运行，提速约2倍
- 在后台启动，输出两个进程的 PID

### 方法2：手动指定GPU和任务范围

```bash
# GPU 0 运行任务 0-24
bash script/batch_eval_motus_parallel.sh 0 0 24

# GPU 1 运行任务 25-49
bash script/batch_eval_motus_parallel.sh 1 25 49
```

### 方法3：单GPU顺序运行（不推荐，较慢）

```bash
# 使用单个GPU运行所有50个任务
cd /data2/liangxiwen/zkd/cry/RoboTwin
bash script/batch_eval_motus_50_tasks.sh
```

### 方法4：单GPU后台运行

```bash
cd /data2/liangxiwen/zkd/cry/RoboTwin
bash script/run_batch_eval_motus.sh
```

## 配置说明

### 权重路径
- 默认路径：`/data2/liangxiwen/zkd/cry/RoboTwin/policy/pi05/checkpoint_torch/motus`
- 如需修改，编辑 `batch_eval_motus_50_tasks.sh` 中的 `MOTUS_CHECKPOINT_DIR` 变量

### 测试次数
- 每个任务默认测试 **50次**
- 如需修改，编辑 `batch_eval_motus_50_tasks.sh` 中的 `TEST_NUM` 变量

### GPU 设备
- 默认使用 `CUDA_VISIBLE_DEVICES=1`
- 如需修改，编辑 `batch_eval_motus_50_tasks.sh` 中的运行命令

## 50个任务列表

1. adjust_bottle
2. beat_block_hammer
3. blocks_ranking_rgb
4. blocks_ranking_size
5. click_alarmclock
6. click_bell
7. dump_bin_bigbin
8. grab_roller
9. handover_block
10. handover_mic
11. hanging_mug
12. lift_pot
13. move_can_pot
14. move_pillbottle_pad
15. move_playingcard_away
16. move_stapler_pad
17. open_laptop
18. open_microwave
19. pick_diverse_bottles
20. pick_dual_bottles
21. place_a2b_left
22. place_a2b_right
23. place_bread_basket
24. place_bread_skillet
25. place_burger_fries
26. place_can_basket
27. place_cans_plasticbox
28. place_container_plate
29. place_dual_shoes
30. place_empty_cup
31. place_fan
32. place_mouse_pad
33. place_object_basket
34. place_object_scale
35. place_object_stand
36. place_phone_stand
37. place_shoe
38. press_stapler
39. put_bottles_dustbin
40. put_object_cabinet
41. rotate_qrcode
42. scan_object
43. shake_bottle_horizontally
44. shake_bottle
45. stack_blocks_three
46. stack_blocks_two
47. stack_bowls_three
48. stack_bowls_two
49. stamp_seal
50. turn_switch

## 日志文件

### 日志位置
- 主日志目录：`/data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs/`
- 主日志文件：`batch_eval_YYYYMMDD_HHMMSS.log`（包含所有任务的总结）
- 任务日志文件：`{task_name}_YYYYMMDD_HHMMSS.log`（每个任务的详细日志）

### 查看日志

```bash
# 并行运行的主日志（实时）
tail -f /data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs/batch_eval_parallel_*.log

# GPU 0 的日志
tail -f /data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs/gpu0_*.log

# GPU 1 的日志
tail -f /data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs/gpu1_*.log

# 查看特定任务的日志
tail -f /data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs/gpu*_{task_name}_*.log

# 查看所有失败的任务
grep "❌ 任务失败" /data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs/batch_eval_*.log
```

## 错误处理

- **如果任务出错**：脚本会：
  1. 停止当前任务
  2. 输出错误信息到主日志
  3. 输出错误日志的最后50行
  4. 记录失败的任务名称
  5. **继续下一个任务**（不会中断整个批量运行）

- **最终总结**：所有任务完成后，会输出：
  - 总任务数
  - 成功任务数
  - 失败任务数
  - 成功率
  - 失败任务列表

## 监控运行状态

```bash
# 查看进程状态（并行运行会有两个进程）
ps aux | grep batch_eval_motus_parallel

# 查看GPU使用情况（推荐）
watch -n 1 nvidia-smi

# 查看进程资源使用
top -p $(pgrep -f batch_eval_motus_parallel | tr '\n' ',')

# 查看日志文件大小（了解进度）
ls -lh /data2/liangxiwen/zkd/cry/RoboTwin/batch_eval_motus_logs/ | grep -E "gpu[01]_|batch_eval"
```

## 停止任务

```bash
# 查找并行运行的进程 PID
ps aux | grep batch_eval_motus_parallel | grep -v grep

# 停止所有并行任务
pkill -f batch_eval_motus_parallel

# 或者分别停止（如果知道PID）
kill <PID_GPU0> <PID_GPU1>

# 强制停止
pkill -9 -f batch_eval_motus_parallel
```

## 注意事项

1. **权重路径检查**：确保 `/data2/liangxiwen/zkd/cry/RoboTwin/policy/pi05/checkpoint_torch/motus` 存在且完整
2. **磁盘空间**：确保有足够的磁盘空间存储日志和评估结果
3. **运行时间**：50个任务，每个任务50次测试，预计需要很长时间（可能数天）
4. **GPU 内存**：确保 GPU 有足够内存
5. **网络连接**：如果使用远程模型服务器，确保网络连接稳定

## 修改说明

### 已修改的文件

1. **`eval_policy_initial_motus.py`**:
   - 修改 `test_num` 从固定30改为可通过参数设置（默认50）
   - 支持 `torch_checkpoint_dir` 参数（通过 `usr_args` 传递）

2. **`batch_eval_motus_50_tasks.sh`**:
   - 添加了50个任务列表
   - 实现了错误处理和继续机制
   - 添加了统计和总结功能

## 示例输出

```
==========================================
批量评估开始时间: 2026-01-25 23:00:00
总任务数: 50
每个任务测试次数: 50
权重路径: /data2/liangxiwen/zkd/cry/RoboTwin/policy/pi05/checkpoint_torch/motus
==========================================

[1/50] 开始任务: adjust_bottle
开始时间: 2026-01-25 23:00:01
...
[1/50] ✅ 任务完成: adjust_bottle
完成时间: 2026-01-25 23:30:00

[2/50] 开始任务: beat_block_hammer
...
```

