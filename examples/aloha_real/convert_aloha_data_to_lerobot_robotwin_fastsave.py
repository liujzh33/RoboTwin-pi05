"""
Fast-save wrapper for Robotwin Aloha conversion.

This script does NOT modify the original converter. It monkey-patches
LeRobotDataset.save_episode to skip expensive full-directory scans:
  - list(self.root.rglob("*.mp4"))
  - list(self.root.rglob("*.parquet"))

These scans are useful as safety checks, but on large datasets they can
introduce O(N^2)-like overhead across episodes.
"""

from __future__ import annotations

import importlib.util
import shutil
import argparse
from pathlib import Path

import numpy as np

import lerobot.common.datasets.lerobot_dataset as lds
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def _load_port_aloha():
    src = Path(__file__).with_name("convert_aloha_data_to_lerobot_robotwin.py")
    spec = importlib.util.spec_from_file_location("robotwin_convert_base", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source converter: {src}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.port_aloha


def _fast_save_episode(self: LeRobotDataset, episode_data: dict | None = None) -> None:
    """
    Equivalent to LeRobotDataset.save_episode, but skips global rglob checks.
    """
    if not episode_data:
        episode_buffer = self.episode_buffer
    else:
        episode_buffer = episode_data

    lds.validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

    # size and task are special cases that won't be added to hf_dataset
    episode_length = episode_buffer.pop("size")
    tasks = episode_buffer.pop("task")
    episode_tasks = list(set(tasks))
    episode_index = episode_buffer["episode_index"]

    episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
    episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

    # Add new tasks to the tasks dictionary
    for task in episode_tasks:
        task_index = self.meta.get_task_index(task)
        if task_index is None:
            self.meta.add_task(task)

    # Given tasks in natural language, find their corresponding task indices
    episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

    for key, ft in self.features.items():
        # index, episode_index, task_index are already processed above, and image and video
        # are processed separately by storing image path and frame info as metadata
        if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
            continue
        episode_buffer[key] = np.stack(episode_buffer[key])

    self._wait_image_writer()
    self._save_episode_table(episode_buffer, episode_index)
    ep_stats = lds.compute_episode_stats(episode_buffer, self.features)

    if len(self.meta.video_keys) > 0:
        video_paths = self.encode_episode_videos(episode_index)
        for key in self.meta.video_keys:
            episode_buffer[key] = video_paths[key]

    # meta.save_episode should be executed after encoding videos
    self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

    ep_data_index = lds.get_episode_data_index(self.meta.episodes, [episode_index])
    ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
    lds.check_timestamps_sync(
        episode_buffer["timestamp"],
        episode_buffer["episode_index"],
        ep_data_index_np,
        self.fps,
        self.tolerance_s,
    )

    # delete temporary images
    img_dir = self.root / "images"
    if img_dir.is_dir():
        shutil.rmtree(self.root / "images")

    if not episode_data:  # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()


def _patch_fast_save() -> None:
    LeRobotDataset.save_episode = _fast_save_episode


if __name__ == "__main__":
    _patch_fast_save()
    port_aloha = _load_port_aloha()
    parser = argparse.ArgumentParser(description="Fast-save Robotwin converter")
    parser.add_argument("--raw_dir", required=True, type=Path)
    parser.add_argument("--repo_id", required=True, type=str)
    parser.add_argument("--task", default="DEBUG", type=str)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--is_mobile", action="store_true")
    parser.add_argument("--mode", default="image", choices=["image", "video"])
    args = parser.parse_args()

    # Use defaults from original converter for omitted args (dataset_config, with_subtask_fields, etc.)
    port_aloha(
        raw_dir=args.raw_dir,
        repo_id=args.repo_id,
        task=args.task,
        push_to_hub=args.push_to_hub,
        is_mobile=args.is_mobile,
        mode=args.mode,
    )

