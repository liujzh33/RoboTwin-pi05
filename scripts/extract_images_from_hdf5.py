#!/usr/bin/env python3
"""
从 HDF5 文件中提取图片，按视角分类保存

读取 processed_data 中的 HDF5 文件，提取所有视角的图片并保存到不同目录
"""

import h5py
import numpy as np
import cv2
from pathlib import Path
import argparse
import sys


def inspect_hdf5_structure(hdf5_path: Path):
    """
    检查 HDF5 文件结构，找出所有图片数据
    """
    print(f"Inspecting HDF5 file: {hdf5_path}")
    print("=" * 60)
    
    with h5py.File(hdf5_path, "r") as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        
        print("File structure:")
        f.visititems(print_structure)
        print()
        
        # 查找图片数据
        image_keys = []
        
        # 检查 observations/images/ 路径
        if "observations" in f:
            obs = f["observations"]
            if "images" in obs:
                images_group = obs["images"]
                print(f"Found images group: observations/images")
                for key in images_group.keys():
                    dataset = images_group[key]
                    print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                    image_keys.append(("observations/images", key))
        
        # 检查其他可能的路径
        if "images" in f:
            images_group = f["images"]
            print(f"Found images group: images")
            for key in images_group.keys():
                dataset = images_group[key]
                print(f"  - {key}: shape={dataset.shape}, dtype={dataset.dtype}")
                image_keys.append(("images", key))
        
        return image_keys


def extract_images(hdf5_path: Path, output_dir: Path):
    """
    从 HDF5 文件中提取所有视角的图片并保存
    
    Args:
        hdf5_path: HDF5 文件路径
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracting images from: {hdf5_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    with h5py.File(hdf5_path, "r") as f:
        image_groups = []
        
        # 查找图片数据组
        if "observations" in f and "images" in f["observations"]:
            image_groups.append(("observations/images", f["observations"]["images"]))
        elif "images" in f:
            image_groups.append(("images", f["images"]))
        else:
            print("Error: No images found in HDF5 file")
            return
        
        # 处理每个图片组
        for group_path, images_group in image_groups:
            print(f"\nProcessing image group: {group_path}")
            
            for cam_name in images_group.keys():
                cam_dataset = images_group[cam_name]
                print(f"\n  Camera: {cam_name}")
                print(f"    Shape: {cam_dataset.shape}")
                print(f"    Dtype: {cam_dataset.dtype}")
                
                # 创建该视角的输出目录
                cam_output_dir = output_dir / cam_name
                cam_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 读取图片数据
                images = cam_dataset[:]
                
                # 处理不同格式的图片数据
                num_images = images.shape[0]
                print(f"    Total frames: {num_images}")
                
                # 检查数据类型
                if cam_dataset.dtype == 'object' or cam_dataset.dtype.kind == 'S':
                    # 字符串类型（可能是编码后的图片）
                    print(f"    Detected string/encoded format, decoding...")
                    for i in range(num_images):
                        # 尝试解码
                        img_data = images[i]
                        if isinstance(img_data, bytes):
                            # 使用 OpenCV 解码
                            nparr = np.frombuffer(img_data, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img is not None:
                                save_path = cam_output_dir / f"frame_{i:05d}.jpg"
                                cv2.imwrite(str(save_path), img)
                        elif isinstance(img_data, np.ndarray):
                            # 已经是数组
                            img = img_data
                            save_path = cam_output_dir / f"frame_{i:05d}.jpg"
                            cv2.imwrite(str(save_path), img)
                        
                        if (i + 1) % 10 == 0:
                            print(f"      Processed {i + 1}/{num_images} frames...")
                else:
                    # 数组类型（直接是图片数组）
                    print(f"    Detected array format, saving directly...")
                    for i in range(num_images):
                        img = images[i]
                        
                        # 确保是 uint8 格式
                        if img.dtype != np.uint8:
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                        
                        # 确保是 BGR 格式（OpenCV 默认）
                        if len(img.shape) == 3:
                            if img.shape[2] == 3:
                                # 检查是否是 RGB，需要转换为 BGR
                                # 通常 HDF5 存储的是 RGB，OpenCV 需要 BGR
                                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            else:
                                img_bgr = img
                        else:
                            img_bgr = img
                        
                        save_path = cam_output_dir / f"frame_{i:05d}.jpg"
                        cv2.imwrite(str(save_path), img_bgr)
                        
                        if (i + 1) % 10 == 0:
                            print(f"      Processed {i + 1}/{num_images} frames...")
                
                print(f"    ✓ Saved {num_images} images to {cam_output_dir}")
    
    print("\n" + "=" * 60)
    print("✓ Image extraction complete!")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="从 HDF5 文件中提取图片，按视角分类保存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 只检查文件结构
  python extract_images_from_hdf5.py --hdf5_path episode_0.hdf5 --inspect
  
  # 提取图片到指定目录
  python extract_images_from_hdf5.py --hdf5_path episode_0.hdf5 --output_dir extracted_images
        """
    )
    parser.add_argument(
        "--hdf5_path", type=str, required=True,
        help="HDF5 文件路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（如果不指定，会在 HDF5 文件同目录下创建 extracted_images 目录）"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="只检查文件结构，不提取图片"
    )
    
    args = parser.parse_args()
    
    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        print(f"Error: HDF5 file does not exist: {hdf5_path}")
        return 1
    
    # 如果只检查结构
    if args.inspect:
        inspect_hdf5_structure(hdf5_path)
        return 0
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 默认在 HDF5 文件同目录下创建 extracted_images
        output_dir = hdf5_path.parent / "extracted_images"
    
    try:
        extract_images(hdf5_path, output_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

