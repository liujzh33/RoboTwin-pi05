#!/usr/bin/env python
"""
Minimal LeRobot v2.1 dataset sanity check (local disk).

What it validates:
1) Required file structure under <dataset_root> (meta/info.json, episodes.jsonl, etc.).
2) info.json parses and declares v2.1, data_path, video_path, features.
3) Parquet files are readable via HuggingFace `datasets` (requires datasets+pyarrow).
4) (Optional) Decode one frame from one mp4 via torchvision/pyav backend (requires av, torchvision).

Example:
  python scripts/test_lerobot21_dataset.py \
    --dataset_root /mnt/a100_1_data3/.../rollout_recovery/hanging_mug/demo_clean
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _read_text_head(path: Path, n_lines: int = 3) -> str:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for _ in range(n_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def _assert_exists(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the LeRobot dataset root folder (contains data/, meta/, videos/).",
    )
    ap.add_argument(
        "--local_lerobot_root",
        type=str,
        default=None,
        help="Optional path to local lerobot repo root (the folder containing `lerobot/`). "
        "If omitted, will try to use RoboTwin/policy/pi05/lerobot relative to this script.",
    )
    ap.add_argument(
        "--try_video_decode",
        action="store_true",
        help="Also try decoding a single frame from a sample mp4 video.",
    )
    ap.add_argument(
        "--max_parquet_files",
        type=int,
        default=3,
        help="How many parquet files to sample-read for sanity check.",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    print(f"[1] dataset_root={dataset_root}")
    _assert_exists(dataset_root, "dataset_root directory")

    meta_dir = dataset_root / "meta"
    data_dir = dataset_root / "data"
    videos_dir = dataset_root / "videos"
    _assert_exists(meta_dir, "meta/ directory")
    _assert_exists(data_dir, "data/ directory")
    _assert_exists(videos_dir, "videos/ directory (can be empty but directory should exist)")

    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    _assert_exists(info_path, "meta/info.json")
    _assert_exists(episodes_path, "meta/episodes.jsonl")
    _assert_exists(tasks_path, "meta/tasks.jsonl")

    print("[2] meta files OK")
    print("    episodes.jsonl head:\n" + _read_text_head(episodes_path, n_lines=3))
    print("    tasks.jsonl head:\n" + _read_text_head(tasks_path, n_lines=3))

    info = json.loads(info_path.read_text(encoding="utf-8"))
    codebase_version = info.get("codebase_version")
    print(f"[3] info.json codebase_version={codebase_version!r}")
    if codebase_version not in ("v2.1", "v2.0"):
        raise ValueError(f"Unexpected codebase_version={codebase_version!r} (expected v2.1/v2.0)")

    for k in ("data_path", "video_path", "features", "fps"):
        if k not in info:
            raise KeyError(f"info.json missing key: {k}")
    features = info["features"]
    video_keys = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
    print(f"[4] declared fps={info.get('fps')}, num_features={len(features)}, video_keys={video_keys}")

    # Resolve local lerobot source tree for optional imports (not strictly needed for parquet-only check).
    if args.local_lerobot_root is not None:
        local_lerobot_root = Path(args.local_lerobot_root).resolve()
    else:
        # scripts/ is at RoboTwin/policy/pi05/scripts; lerobot repo is at RoboTwin/policy/pi05/lerobot
        local_lerobot_root = (Path(__file__).resolve().parents[1] / "lerobot").resolve()
    if (local_lerobot_root / "lerobot").exists():
        sys.path.insert(0, str(local_lerobot_root))
        print(f"[5] using local lerobot source: {local_lerobot_root}")
    else:
        print(f"[5] local lerobot source not found at: {local_lerobot_root} (continuing)")

    # --- Parquet readability check (datasets + pyarrow) ---
    try:
        import datasets  # noqa: F401
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Failed to import `datasets`. Install deps first:\n"
            "  pip install datasets pyarrow\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    parquet_files = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}/chunk-*/episode_*.parquet")

    sample_files = parquet_files[: max(1, args.max_parquet_files)]
    print(f"[6] found {len(parquet_files)} parquet files; sampling {len(sample_files)}")

    for i, pf in enumerate(sample_files):
        ds = load_dataset("parquet", data_files=str(pf), split="train")
        row0 = ds[0]
        print(f"    parquet[{i}]={pf.name} cols={len(ds.column_names)}")
        # Print a few key columns if present
        keys_to_show = [
            "timestamp",
            "frame_index",
            "episode_index",
            "task_index",
            "action",
            "observation.state",
        ]
        present = [k for k in keys_to_show if k in row0]
        small = {k: row0[k] for k in present}
        print(f"      row0 keys(sample)={list(small.keys())}")
        for k, v in small.items():
            # keep output compact
            if isinstance(v, (list, tuple)) and len(v) > 8:
                print(f"      {k}: len={len(v)} head={v[:3]} ... tail={v[-3:]}")
            else:
                print(f"      {k}: {v}")

    print("[7] parquet read OK (datasets+pyarrow working)")

    # --- Optional: decode a single frame from an mp4 ---
    if args.try_video_decode:
        if not video_keys:
            raise RuntimeError("No video keys declared in info.json; cannot test video decode.")
        # The folder name under videos/chunk-000 is typically either the full key or an alias.
        # In your dataset, it's like videos/chunk-000/front_camera/episode_000000.mp4.
        chunk0 = videos_dir / "chunk-000"
        _assert_exists(chunk0, "videos/chunk-000 directory")
        # pick the first subdir that has mp4s
        mp4s = sorted(chunk0.glob("*/*.mp4"))
        if not mp4s:
            raise FileNotFoundError(f"No mp4 files found under: {chunk0}/*/*.mp4")
        mp4 = mp4s[0]
        print(f"[8] trying video decode: {mp4}")
        try:
            from lerobot.common.datasets.video_utils import decode_video_frames
        except Exception as e:
            raise RuntimeError(
                "Failed to import lerobot video_utils. Make sure local lerobot path is correct, and deps exist:\n"
                "  pip install av pyarrow torchvision\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        # decode a single frame at t=0.0
        frames = decode_video_frames(str(mp4), timestamps=[0.0], tolerance_s=1e-3, backend="pyav")
        print(f"    decoded frames shape={tuple(frames.shape)} dtype={frames.dtype}")
        print("[9] video decode OK")

    print("\nSUCCESS: dataset looks compliant with LeRobot v2.1 and is readable locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

