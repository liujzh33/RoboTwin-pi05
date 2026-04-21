# 5 个任务的语言丰富化指令（clean / random 共用一套脚本）

每个任务用**同一套** `PHASE_VARIANTS`，clean_50 和 randomized_500 只是 `--data_dir` 不同。在 **pi05** 项目根目录下执行（不要进 Qwen3-VL 再执行，以便 data_dir 路径一致）。

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
source .venv/bin/activate
```

---

## 1. beat_block_hammer（4 阶段）

**clean_50：**
```bash
python Qwen3-VL/enrich_beat_block_hammer.py --data_dir processed_data/beat_block_hammer-aloha-agilex_clean_50-50
```

**randomized_500：**
```bash
python Qwen3-VL/enrich_beat_block_hammer.py --data_dir processed_data/beat_block_hammer-aloha-agilex_randomized_500-500
```

---

## 2. blocks_ranking_rgb（13 阶段，v1）

**clean_50：**
```bash
python Qwen3-VL/enrich_blocks_ranking.py --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_clean_50-50
```

**randomized_500：**
```bash
python Qwen3-VL/enrich_blocks_ranking.py --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_randomized_500-500
```

---

## 3. blocks_ranking_size（13 阶段，v1）

**clean_50：**
```bash
python Qwen3-VL/enrich_blocks_ranking_size.py --data_dir processed_data/blocks_ranking_size-aloha-agilex_clean_50-50
```

**randomized_500：**
```bash
python Qwen3-VL/enrich_blocks_ranking_size.py --data_dir processed_data/blocks_ranking_size-aloha-agilex_randomized_500-500
```

---

## 4. click_alarmclock（3 阶段）

**clean_50：**
```bash
python Qwen3-VL/enrich_click_alarmclock.py --data_dir processed_data/click_alarmclock-aloha-agilex_clean_50-50
```

**randomized_500：**
```bash
python Qwen3-VL/enrich_click_alarmclock.py --data_dir processed_data/click_alarmclock-aloha-agilex_randomized_500-500
```

---

## 5. click_bell（3 阶段）

**clean_50：**
```bash
python Qwen3-VL/enrich_click_bell.py --data_dir processed_data/click_bell-aloha-agilex_clean_50-50
```

**randomized_500：**
```bash
python Qwen3-VL/enrich_click_bell.py --data_dir processed_data/click_bell-aloha-agilex_randomized_500-500
```

---

## 一键跑满 5 任务 × 2 种数据（共 10 次）

```bash
cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
source .venv/bin/activate

# beat_block_hammer
python Qwen3-VL/enrich_beat_block_hammer.py --data_dir processed_data/beat_block_hammer-aloha-agilex_clean_50-50
python Qwen3-VL/enrich_beat_block_hammer.py --data_dir processed_data/beat_block_hammer-aloha-agilex_randomized_500-500

# blocks_ranking_rgb
python Qwen3-VL/enrich_blocks_ranking.py --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_clean_50-50
python Qwen3-VL/enrich_blocks_ranking.py --data_dir processed_data/blocks_ranking_rgb-aloha-agilex_randomized_500-500

# blocks_ranking_size
python Qwen3-VL/enrich_blocks_ranking_size.py --data_dir processed_data/blocks_ranking_size-aloha-agilex_clean_50-50
python Qwen3-VL/enrich_blocks_ranking_size.py --data_dir processed_data/blocks_ranking_size-aloha-agilex_randomized_500-500

# click_alarmclock
python Qwen3-VL/enrich_click_alarmclock.py --data_dir processed_data/click_alarmclock-aloha-agilex_clean_50-50
python Qwen3-VL/enrich_click_alarmclock.py --data_dir processed_data/click_alarmclock-aloha-agilex_randomized_500-500

# click_bell
python Qwen3-VL/enrich_click_bell.py --data_dir processed_data/click_bell-aloha-agilex_clean_50-50
python Qwen3-VL/enrich_click_bell.py --data_dir processed_data/click_bell-aloha-agilex_randomized_500-500
```

说明：不传 `--data_dir` 时，各脚本使用其内默认的 `DATA_ROOT`（多为 randomized_500 路径）。
