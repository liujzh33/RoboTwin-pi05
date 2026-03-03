import pandas as pd
import os
import json

# 请替换为您的实际路径
path = "/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache/huggingface/lerobot/blocks_ranking_rgb_pi05_200/data/chunk-000/episode_000000.parquet"
df = pd.read_parquet(path)

print(df.head())
print(df.columns)
print(df.subtasks)