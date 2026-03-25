#!/usr/bin/env python3
"""
查看HDF5文件的所有键和数据结构
"""
import h5py
import numpy as np
import sys

def print_hdf5_structure(file_path):
    """递归打印HDF5文件的所有键和数据结构"""
    print(f"\n{'='*80}")
    print(f"📁 HDF5文件: {file_path}")
    print(f"{'='*80}\n")
    
    with h5py.File(file_path, 'r') as f:
        def print_group(name, obj, indent=0):
            prefix = "  " * indent
            if isinstance(obj, h5py.Group):
                print(f"{prefix}📂 Group: {name}")
                # 打印属性
                if obj.attrs:
                    print(f"{prefix}   Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{prefix}     {attr_name}: {attr_value}")
                # 递归打印子项
                for key in sorted(obj.keys()):
                    print_group(f"{name}/{key}", obj[key], indent + 1)
            elif isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                print(f"{prefix}📊 Dataset: {name}")
                print(f"{prefix}   Shape: {shape}")
                print(f"{prefix}   Dtype: {dtype}")
                
                # 检查是否是二进制数据（图像、字符串等）
                is_binary = False
                if 'rgb' in name.lower() or 'image' in name.lower():
                    is_binary = True
                    print(f"{prefix}   Type: JPEG encoded image data (binary)")
                elif dtype.kind in ['S', 'U', 'O']:  # 字符串类型
                    is_binary = True
                    if 'rgb' in name.lower():
                        print(f"{prefix}   Type: JPEG encoded image data (binary)")
                    else:
                        print(f"{prefix}   Type: String/Binary data")
                
                # 如果不是二进制数据，才尝试打印数值
                if not is_binary:
                    # 如果是小数组，打印一些示例值
                    if np.prod(shape) <= 20:
                        try:
                            print(f"{prefix}   Data: {obj[:]}")
                        except:
                            print(f"{prefix}   Data: [Cannot display]")
                    elif len(shape) == 1 and shape[0] <= 10:
                        try:
                            print(f"{prefix}   Data: {obj[:]}")
                        except:
                            print(f"{prefix}   Data: [Cannot display]")
                    else:
                        try:
                            # 只读取第一个元素作为示例
                            if len(shape) > 0:
                                sample_idx = tuple([0] * len(shape))
                                sample = obj[sample_idx]
                                print(f"{prefix}   Data: [Sample value: {sample}...]")
                            else:
                                print(f"{prefix}   Data: [Scalar value]")
                        except:
                            print(f"{prefix}   Data: [Cannot display]")
                
                # 打印属性
                if obj.attrs:
                    print(f"{prefix}   Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{prefix}     {attr_name}: {attr_value}")
        
        # 打印根目录下的所有键
        print("📋 文件结构:\n")
        for key in sorted(f.keys()):
            print_group(key, f[key])
        
        # 打印文件级别的属性
        if f.attrs:
            print(f"\n📝 文件级别属性:")
            for attr_name, attr_value in f.attrs.items():
                print(f"  {attr_name}: {attr_value}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python view_hdf5_keys.py <hdf5_file_path>")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    print_hdf5_structure(hdf5_path)

