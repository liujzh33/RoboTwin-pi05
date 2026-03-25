#!/usr/bin/env python3
"""
从HDF5文件中提取多视角图片并保存为视频（FPS=10）
"""
import sys
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import cv2


def parse_img_array(data):
    """
    将一个字节流数组解码为图像数组。
    
    Args:
        data: np.ndarray of shape (N,), 每个元素要么是 Python bytes，要么是 np.ndarray(dtype=uint8)
    Returns:
        imgs: np.ndarray of shape (N, H, W, C), dtype=uint8 (BGR格式)
    """
    # 确保 data 是可迭代的一维数组
    flat = data.ravel()
    
    imgs = []
    for buf in flat:
        # buf 可能是 bytes，也可能是 np.ndarray(dtype=uint8)
        if isinstance(buf, (bytes, bytearray)):
            arr = np.frombuffer(buf, dtype=np.uint8)
        elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
            arr = buf
        else:
            raise TypeError(f"Unsupported buffer type: {type(buf)}")
        
        # 解码成 BGR 图像
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode 返回 None，说明字节流可能不是有效的图片格式")
        imgs.append(img)
    
    # 将 list 转成形如 (N, H, W, C) 的 ndarray
    return np.stack(imgs, axis=0)


def images_to_video(imgs: np.ndarray, out_path: str, fps: float = 30.0, is_rgb: bool = False) -> None:
    """
    将图像数组保存为视频文件
    
    Args:
        imgs: numpy数组，形状为 (N, H, W, C)，C为3或4
        out_path: 输出视频路径
        fps: 帧率
        is_rgb: 如果True，图像是RGB格式；如果False，图像是BGR格式（OpenCV默认）
    """
    if (not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[3] not in (3, 4)):
        raise ValueError("imgs must be a numpy.ndarray of shape (N, H, W, C), with C equal to 3 or 4.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_frames, H, W, C = imgs.shape
    if C == 3:
        pixel_format = "rgb24" if is_rgb else "bgr24"
    else:
        pixel_format = "rgba"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            pixel_format,
            "-video_size",
            f"{W}x{H}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            f"{out_path}",
        ],
        stdin=subprocess.PIPE,
    )
    ffmpeg.stdin.write(imgs.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise IOError(f"Cannot open ffmpeg. Please check the output path and ensure ffmpeg is supported.")
    
    print(
        f"🎬 Video is saved to `{out_path}`, containing \033[94m{n_frames}\033[0m frames at {W}×{H} resolution and {fps} FPS."
    )

def extract_videos_from_hdf5(hdf5_path, output_dir=None, fps=10.0):
    """
    从HDF5文件中提取所有视角的图像并保存为视频
    
    Args:
        hdf5_path: HDF5文件路径
        output_dir: 输出目录（如果为None，则在HDF5文件同目录下创建videos文件夹）
        fps: 视频帧率，默认10.0
    """
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        print(f"❌ 错误: 文件不存在 {hdf5_path}")
        return
    
    # 设置输出目录
    if output_dir is None:
        output_dir = hdf5_path.parent / "videos"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"📁 HDF5文件: {hdf5_path}")
    print(f"📂 输出目录: {output_dir}")
    print(f"🎬 帧率: {fps} FPS")
    print(f"{'='*80}\n")
    
    # 读取HDF5文件
    with h5py.File(hdf5_path, 'r') as f:
        # 检查是否有observation组
        if 'observation' not in f:
            print("❌ 错误: HDF5文件中没有找到 'observation' 组")
            return
        
        observation_group = f['observation']
        
        # 遍历所有相机
        camera_names = sorted(observation_group.keys())
        if not camera_names:
            print("❌ 错误: 没有找到任何相机数据")
            return
        
        print(f"📹 找到 {len(camera_names)} 个相机视角: {', '.join(camera_names)}\n")
        
        # 处理每个相机
        for cam_name in camera_names:
            cam_group = observation_group[cam_name]
            
            # 检查是否有rgb数据
            if 'rgb' not in cam_group:
                print(f"⚠️  警告: {cam_name} 没有rgb数据，跳过")
                continue
            
            print(f"🔄 处理 {cam_name}...")
            
            try:
                # 读取JPEG编码的图像数据
                rgb_data = cam_group['rgb'][:]
                
                # 解码图像（parse_img_array返回BGR格式）
                images = parse_img_array(rgb_data)
                
                print(f"   ✅ 解码完成: {len(images)} 帧, 尺寸: {images[0].shape}")
                
                # 保存视频（parse_img_array返回BGR，所以is_rgb=False）
                video_path = output_dir / f"{cam_name}.mp4"
                images_to_video(images, str(video_path), fps=fps, is_rgb=False)
                
                print(f"   ✅ 视频已保存: {video_path}\n")
                
            except Exception as e:
                print(f"   ❌ 处理 {cam_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        print(f"🎉 所有视频已保存到: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从HDF5文件提取多视角视频")
    parser.add_argument("--hdf5_path", type=str, required=True, help="HDF5文件路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（可选，默认为HDF5文件同目录下的videos文件夹）")
    parser.add_argument("--fps", type=float, default=10.0, help="视频帧率（默认10.0）")
    
    args = parser.parse_args()
    
    extract_videos_from_hdf5(
        hdf5_path=args.hdf5_path,
        output_dir=args.output_dir,
        fps=args.fps
    )

