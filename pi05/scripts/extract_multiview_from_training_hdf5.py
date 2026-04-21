#!/usr/bin/env python3
"""
从 training_data/<task>/episode_K/episode_K.hdf5 提取图像并保存。

默认：仅 head 相机 cam_high；导出该 episode「全部帧」、无帧数上限；无损 PNG（compression 0）。
若要求文件也完全不经压缩，使用 --format bmp（体积最大）。
HDF5 内若为 JPEG 字节，会先解码为像素再写出（像素无损）。

解码规则与 examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py 中
load_raw_images_per_camera 一致。

示例：
  cd /mnt/data1/liujingzhi/RoboTwin/policy/pi05
  python scripts/extract_multiview_from_training_hdf5.py \\
    --root training_data/pi05_multi_task_5_v1.0 --episode_id 0

  # 三路手腕+头（旧行为）：
  python scripts/extract_multiview_from_training_hdf5.py ... --all_cameras

  # 只要部分帧：
  python scripts/extract_multiview_from_training_hdf5.py \\
    --root training_data/pi05_multi_task_5_v1.0 --episode_id 0 --frame_indices 0,10,50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

# 与 Aloha / convert_aloha_data_to_lerobot_robotwin 命名一致
ALL_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
HEAD_CAMERA = "cam_high"

# PNG：compression 0 = 无损、最小 zlib（仍为合法 PNG；要零 zlib 请用 --format bmp）
_PNG_PARAMS = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]


def _num_frames(ep: h5py.File, camera: str = HEAD_CAMERA) -> int:
    path = f"/observations/images/{camera}"
    if path not in ep:
        raise KeyError(f"Missing {path} in {ep.file.filename}")
    return int(ep[path].shape[0])


def _decode_frame(ep: h5py.File, camera: str, fi: int) -> np.ndarray | None:
    path = f"/observations/images/{camera}"
    ds = ep[path]
    if ds.ndim == 4:
        return np.asarray(ds[fi])
    raw = ds[fi]
    buf = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/data1/liujingzhi/RoboTwin/policy/pi05/training_data/pi05_multi_task_5_v1.0"),
        help="含若干 task 子目录的根路径",
    )
    parser.add_argument("--episode_id", type=int, default=0, help="episode 编号，例如 0 -> episode_0/")
    parser.add_argument(
        "--frame_indices",
        type=str,
        default=None,
        help="可选。要保存的帧索引，逗号分隔。省略则保存该 episode 全部帧。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出根目录（默认 <root>/../_extracted_head_ep{episode_id}_png_only）",
    )
    parser.add_argument(
        "--format",
        choices=("png", "bmp"),
        default="png",
        help="png=无损 PNG（compression 0）；bmp=无压缩位图，体积更大",
    )
    parser.add_argument(
        "--all_cameras",
        action="store_true",
        help="同时保存 cam_high / cam_left_wrist / cam_right_wrist（默认仅 cam_high）",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    ep_id = args.episode_id
    out_root = args.out
    cameras = list(ALL_CAMERAS) if args.all_cameras else [HEAD_CAMERA]
    if out_root is None:
        fmt_tag = "png_only" if args.format == "png" else "bmp_only"
        cam_tag = "allcams" if args.all_cameras else "head"
        out_root = root.parent / f"_extracted_{cam_tag}_ep{ep_id}_{fmt_tag}"
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not task_dirs:
        raise SystemExit(f"No subdirs under {root}")

    for task_dir in task_dirs:
        ep_dir = task_dir / f"episode_{ep_id}"
        h5_path = ep_dir / f"episode_{ep_id}.hdf5"
        if not h5_path.is_file():
            print(f"[skip] missing {h5_path}")
            continue

        tag = task_dir.name
        save_dir = out_root / tag
        save_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(h5_path, "r") as f:
            n = _num_frames(f)
            if args.frame_indices is None:
                frame_iter = range(n)
            else:
                frame_iter = [int(x.strip()) for x in args.frame_indices.split(",") if x.strip()]

            count = 0
            for fi in frame_iter:
                if fi < 0 or fi >= n:
                    print(f"[warn] {tag}: frame {fi} out of range [0,{n})")
                    continue
                for cam in cameras:
                    bgr = _decode_frame(f, cam, fi)
                    if bgr is None:
                        print(f"[warn] {tag}: decode failed cam={cam} frame={fi}")
                        continue
                    ext = ".png" if args.format == "png" else ".bmp"
                    fname = f"ep{ep_id}_frame{fi:06d}_{cam}{ext}"
                    if args.format == "png":
                        cv2.imwrite(str(save_dir / fname), bgr, _PNG_PARAMS)
                    else:
                        cv2.imwrite(str(save_dir / fname), bgr)
                    count += 1

        which = f"0..{n - 1} (all)" if args.frame_indices is None else args.frame_indices.strip()
        print(f"[ok] {tag} -> {save_dir} (T={n}, frames={which}, wrote {count} files)")


if __name__ == "__main__":
    main()
