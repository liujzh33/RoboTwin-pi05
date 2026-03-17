#!/usr/bin/env python3
"""删除 eval_result 下每个任务中的 pi05/demo_clean/pi05_ft_4a100 文件夹。"""

import os
import shutil
from pathlib import Path

EVAL_RESULT_ROOT = Path("/data2/liangxiwen/zkd/cry/RoboTwin/eval_result")
TARGET_SUBDIR = "pi05/demo_clean/pi05_ft_4a100"


def main():
    if not EVAL_RESULT_ROOT.exists():
        print(f"错误: {EVAL_RESULT_ROOT} 不存在")
        return

    deleted = []
    skipped = []

    for task_dir in sorted(EVAL_RESULT_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue
        target = task_dir / TARGET_SUBDIR
        if target.exists() and target.is_dir():
            try:
                shutil.rmtree(target)
                deleted.append(str(target))
                print(f"已删除: {target}")
            except Exception as e:
                print(f"删除失败 {target}: {e}")
        else:
            skipped.append(task_dir.name)

    print(f"\n共删除 {len(deleted)} 个文件夹")
    if skipped:
        print(f"以下 {len(skipped)} 个任务下无 {TARGET_SUBDIR}，已跳过: {', '.join(skipped[:10])}{'...' if len(skipped) > 10 else ''}")


if __name__ == "__main__":
    main()
