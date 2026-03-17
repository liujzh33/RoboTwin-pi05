"""
错误恢复工具函数 - 支持多任务扩展
"""

import os
import json
import numpy as np
import inspect
import ast
from typing import Dict, List, Any, Optional


def load_task_frame_threshold(task_name: str, threshold_type: str = "max") -> Optional[int]:
    """
    加载任务的帧数阈值
    
    优先从task_lim目录读取，如果不存在则从eval_result目录查找
    
    Args:
        task_name: 任务名称
        threshold_type: 阈值类型 ("max", "p99", "p95", "p90", "mean")
    
    Returns:
        阈值帧数，如果找不到返回None
    """
    # 优先从task_lim目录读取（统一存放位置）
    task_lim_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "task_lim")
    threshold_file = os.path.join(task_lim_dir, f"{task_name}.json")
    
    if os.path.exists(threshold_file):
        try:
            with open(threshold_file, 'r') as f:
                data = json.load(f)
            
            thresholds = data.get("thresholds", {})
            if threshold_type == "max":
                return thresholds.get("max")
            elif threshold_type == "p99":
                return thresholds.get("p99")
            elif threshold_type == "p95":
                return thresholds.get("p95")
            elif threshold_type == "p90":
                return thresholds.get("p90")
            elif threshold_type == "mean":
                return thresholds.get("mean")
        except Exception as e:
            print(f"Warning: Failed to load threshold from {threshold_file}: {e}")
    
    # 如果task_lim目录不存在，尝试从eval_result目录查找（向后兼容）
    base_dir = "eval_result"
    possible_paths = [
        f"{base_dir}/{task_name}/pi05/demo_clean/pi05_ft_4a100",
    ]
    
    for base_path in possible_paths:
        if not os.path.exists(base_path):
            continue
        
        # 查找所有子目录中的统计文件
        for root, dirs, files in os.walk(base_path):
            stats_file = os.path.join(root, "success_video_stats.json")
            if os.path.exists(stats_file):
                try:
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                    
                    if threshold_type == "max":
                        return stats.get("max")
                    elif threshold_type == "p99":
                        return stats.get("p99")
                    elif threshold_type == "p95":
                        return stats.get("p95")
                    elif threshold_type == "p90":
                        return stats.get("p90")
                    elif threshold_type == "mean":
                        return int(stats.get("mean", 0))
                except Exception as e:
                    print(f"Warning: Failed to load stats from {stats_file}: {e}")
                    continue
    
    return None


def count_play_once_stages(task_class) -> int:
    """
    自动检测play_once方法的阶段数
    
    通过分析play_once方法中的self.move()调用次数来确定阶段数
    每个self.move()调用代表一个阶段
    
    Args:
        task_class: 任务类
    
    Returns:
        阶段数（至少为1）
    """
    if not hasattr(task_class, 'play_once'):
        return 1
    
    try:
        source = inspect.getsource(task_class.play_once)
        
        # 统计self.move()调用次数（不包括注释和字符串中的）
        move_count = 0
        try:
            # 尝试解析AST，如果失败则使用字符串匹配
            # 注意：inspect.getsource可能返回的代码有缩进问题，需要处理
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Attribute) and 
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id == 'self' and
                        node.func.attr == 'move'):
                        move_count += 1
        except (SyntaxError, IndentationError) as e:
            # AST解析失败（通常是缩进问题），使用字符串匹配作为fallback
            # 这是正常的，因为inspect.getsource可能返回的代码缩进不完整
            move_count = source.count('self.move(')
        except Exception as e:
            # 其他异常也使用fallback
            move_count = source.count('self.move(')
        
        # 每个self.move()调用代表一个阶段
        # 但需要排除self.info["info"]相关的代码
        # 实际上，self.move()的数量就是阶段数
        stages = max(1, move_count)
        
        return stages
        
    except Exception as e:
        print(f"Warning: Failed to analyze play_once stages for {task_class.__name__}: {e}")
        # 默认返回3（大多数任务有2-3个阶段）
        return 3


def get_all_actor_poses(env) -> Dict[str, Any]:
    """
    自动获取环境中所有actor的位置
    
    Args:
        env: 环境实例
    
    Returns:
        包含所有actor位置的字典
    """
    poses = {}
    
    # 常见的actor命名模式
    common_patterns = [
        'block1', 'block2', 'block3', 'block4',
        'bottle', 'bottles',
        'hammer', 'tray', 'pot', 'microwave',
        'deskbin', 'bigbin',
        'microphone', 'mic',
        'roller', 'screwdriver', 'dumbbell',
        'phone', 'can', 'stapler', 'pad',
        'bowl', 'basket', 'container', 'plate',
        'bread', 'skillet', 'mouse', 'pad',
        'laptop', 'bell', 'alarmclock',
        'mug', 'pillbottle', 'shoe', 'shoes',
        'burger', 'fries', 'fan', 'playingcard',
        'qrcode', 'switch', 'seal',
        'box', 'target_box',  # handover_block任务
    ]
    
    # 遍历所有可能的actor
    for pattern in common_patterns:
        if hasattr(env, pattern):
            actor = getattr(env, pattern)
            if hasattr(actor, 'get_pose'):
                try:
                    poses[f"{pattern}_pose"] = actor.get_pose()
                except:
                    pass
    
    # 也尝试通过scene获取所有actor（更全面的检测）
    if hasattr(env, 'scene'):
        try:
            # 获取场景中的所有实体
            for entity in env.scene.get_all_entities():
                if hasattr(entity, 'get_pose'):
                    name = entity.get_name() if hasattr(entity, 'get_name') else None
                    if name and name not in ['table', 'ground', 'robot']:
                        # 保存所有非基础物体的位置（不仅仅是block/bottle）
                        # 这样可以捕获所有物体，包括box, target_box等
                        name_key = f"{name}_pose"
                        if name_key not in poses:  # 避免重复
                            try:
                                poses[name_key] = entity.get_pose()
                            except:
                                pass
        except:
            pass
    
    # 最后，尝试通过env的所有属性来查找actor（兜底方案）
    # 遍历env的所有属性，查找可能是actor的对象
    try:
        for attr_name in dir(env):
            if attr_name.startswith('_') or attr_name in ['scene', 'robot', 'table', 'ground']:
                continue
            if attr_name not in common_patterns:  # 避免重复检查
                try:
                    attr_obj = getattr(env, attr_name)
                    if hasattr(attr_obj, 'get_pose') and hasattr(attr_obj, 'actor'):
                        # 这看起来是一个actor对象
                        pose_key = f"{attr_name}_pose"
                        if pose_key not in poses:
                            poses[pose_key] = attr_obj.get_pose()
                except:
                    pass
    except:
        pass
    
    return poses


def restore_all_actor_poses(env, poses: Dict[str, Any]):
    """
    恢复所有actor的位置
    
    Args:
        env: 环境实例
        poses: 包含actor位置的字典
    """
    restored_count = 0
    for key, pose in poses.items():
        if key.endswith('_pose'):
            actor_name = key[:-5]  # 移除'_pose'后缀
            
            # 尝试直接获取actor
            if hasattr(env, actor_name):
                actor = getattr(env, actor_name)
                restored = False
                if hasattr(actor, 'actor') and hasattr(actor.actor, 'set_pose'):
                    try:
                        actor.actor.set_pose(pose)
                        restored = True
                        restored_count += 1
                    except Exception as e:
                        pass
                elif hasattr(actor, 'set_pose'):
                    try:
                        actor.set_pose(pose)
                        restored = True
                        restored_count += 1
                    except Exception as e:
                        pass
                
                if restored:
                    print(f"\033[94m[Recovery] Restored {actor_name} pose\033[0m")
    
    # 也尝试通过scene恢复（如果直接恢复失败）
    if restored_count == 0 and hasattr(env, 'scene'):
        try:
            for key, pose in poses.items():
                if key.endswith('_pose'):
                    entity_name = key[:-5]  # 移除'_pose'后缀
                    for entity in env.scene.get_all_entities():
                        if hasattr(entity, 'get_name'):
                            name = entity.get_name()
                            if name == entity_name or name.replace('_', '') == entity_name.replace('_', ''):
                                if hasattr(entity, 'set_pose'):
                                    try:
                                        entity.set_pose(pose)
                                        print(f"\033[94m[Recovery] Restored {entity_name} pose via scene\033[0m")
                                        restored_count += 1
                                    except:
                                        pass
        except:
            pass
    
    if restored_count > 0:
        print(f"\033[94m[Recovery] Successfully restored {restored_count} actor pose(s)\033[0m")
    else:
        print(f"\033[93m[Recovery] Warning: Failed to restore any actor poses from snapshot\033[0m")

