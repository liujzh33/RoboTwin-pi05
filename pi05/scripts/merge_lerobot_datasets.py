"""
Merge multiple LeRobot datasets (by repo_id) into a single dataset.

Use this to build a multi-task dataset from existing single-task LeRobot repos
without re-running generate.sh on the combined processed_data. Result is equivalent
to running generate.sh on training_data/pi05_multi_task_5 (same schema, same episode
order: first repo's episodes, then second, etc.).

Usage:
  cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05 && source .venv/bin/activate
  export XDG_CACHE_HOME=/mnt/data1/liujingzhi/RoboTwin/policy/pi05/.cache
  uv run scripts/merge_lerobot_datasets.py \\
    --source_repos beat_hammer_pi05_200 blocks_ranking_rgb_pi05_200 blocks_ranking_size_pi05_200 click_alarmclock_pi05_200 click_bell_pi05_200 \\
    --output_repo pi05_multi_task_5
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# Allow importing lerobot from project
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_episode_data_index, load_episodes, load_info


def _ensure_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "__array__"):
        return np.array(x)
    return x


def merge_datasets(
    source_repos: list[str],
    output_repo: str,
    cache_root: Path,
) -> None:
    cache_root = Path(cache_root)
    if not cache_root.exists():
        raise FileNotFoundError(f"Cache root not found: {cache_root}")

    # Load first source to get features and fps
    first_root = cache_root / source_repos[0]
    if not first_root.exists():
        raise FileNotFoundError(f"Source repo not found: {first_root}")
    info = load_info(first_root)
    features = info["features"]
    fps = info["fps"]
    robot_type = info["robot_type"]

    # Create merged dataset (same features, no videos); root = full path to repo dir
    output_path = cache_root / output_repo
    if output_path.exists():
        shutil.rmtree(output_path)

    merged = LeRobotDataset.create(
        repo_id=output_repo,
        fps=fps,
        root=output_path,
        robot_type=robot_type,
        features=features,
        use_videos=False,
    )

    # add_frame 会自动设置 timestamp, frame_index, episode_index, index, task_index，不要传入
    skip_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    feature_keys = set(features.keys()) - skip_keys

    for repo_id in source_repos:
        repo_path = cache_root / repo_id
        if not repo_path.exists():
            raise FileNotFoundError(f"Source repo not found: {repo_path}")

        ds = LeRobotDataset(repo_id, root=repo_path)
        episodes = load_episodes(ds.root)
        ep_data_index = get_episode_data_index(episodes)

        n_episodes = ds.meta.total_episodes
        for ep_idx in range(n_episodes):
            length = episodes[ep_idx]["length"]
            start = int(ep_data_index["from"][ep_idx].item())
            for i in range(length):
                idx = start + i
                item = ds[idx]
                # Build frame dict: only keys in merged features + "task"
                frame = {"task": item["task"]}
                for k in feature_keys:
                    if k not in item:
                        continue
                    v = item[k]
                    if k in ("instructions", "subtasks", "phase_info") and isinstance(v, (list, dict)):
                        v = json.dumps(v, ensure_ascii=False)
                    v = _ensure_numpy(v)
                    # frame_idx 要求 dtype=int32, shape=(1,)
                    if k == "frame_idx":
                        v = np.array([int(v)], dtype=np.int32) if np.isscalar(v) or getattr(v, "ndim", 0) == 0 else np.asarray(v, dtype=np.int32).reshape(1)
                    frame[k] = v
                merged.add_frame(frame)
            merged.save_episode()

    print(f"Merged {len(source_repos)} repos -> {output_repo}")
    print(f"  total_episodes: {merged.meta.total_episodes}, total_frames: {merged.meta.total_frames}")


def main():
    parser = argparse.ArgumentParser(description="Merge LeRobot datasets into one.")
    parser.add_argument(
        "--source_repos",
        nargs="+",
        required=True,
        help="Repo IDs to merge (order preserved)",
    )
    parser.add_argument(
        "--output_repo",
        required=True,
        help="Output repo_id for merged dataset",
    )
    parser.add_argument(
        "--cache_root",
        default=None,
        help="Parent of repo dirs (default: XDG_CACHE_HOME/huggingface/lerobot)",
    )
    args = parser.parse_args()

    if args.cache_root is None:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", REPO_ROOT / ".cache")) / "huggingface" / "lerobot"
    else:
        cache_root = Path(args.cache_root)

    merge_datasets(
        source_repos=args.source_repos,
        output_repo=args.output_repo,
        cache_root=cache_root,
    )


if __name__ == "__main__":
    main()
