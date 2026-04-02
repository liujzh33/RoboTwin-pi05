#!/usr/bin/env python3
"""
从 training_data/pi05_multi_task_recovery_3 下全部子目录，为每个 episode 生成 1 条 Judge 训练样本：

- 含 error_attempt_range（recovery 数据）：start/end 取 error_attempt 起止帧的三视角图（与 pi05 一致）。
- 否则（正确演示数据）：start=0，end 从 phase_info.checkpoints 中按 extract_judge_success_samples 规则选取。

输出 ShareGPT 格式 JSON + images/，供 LLaMA-Factory 微调 Qwen2-VL。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_judge_dataset_beat_block_hammer import (  # noqa: E402
    IMAGE_KEYS,
    build_sharegpt_item,
    load_episode_meta,
    save_images_for_episode,
)
from extract_judge_success_samples_beat_block_hammer import (  # noqa: E402
    SUCCESS_VARIANTS,
    build_success_sharegpt_item,
    choose_end_frame,
    load_episode_instructions,
    save_success_frames,
)

# 与 pi05_multi_task_recovery_3 目录名前缀一致（长前缀优先）
TASK_PREFIXES: tuple[str, ...] = (
    "blocks_ranking_size",
    "blocks_ranking_rgb",
    "beat_block_hammer",
)

DATASET_ROOT = Path("/mnt/data1/liujingzhi/dataset")
DATASET_RECOVERY_ROOT = Path("/mnt/data1/liujingzhi/dataset_recovery")


def parse_training_folder(name: str) -> tuple[str, str]:
    for prefix in TASK_PREFIXES:
        if name.startswith(prefix + "-"):
            return prefix, name[len(prefix) + 1 :]
    raise ValueError(f"无法从目录名解析任务: {name}")


def raw_hdf5_dir(task: str, variant_rest: str) -> Path:
    """将 training 子目录名的后缀映射到原始 HDF5 所在 data 目录。"""
    if variant_rest.startswith("aloha-agilex_clean_50"):
        return DATASET_ROOT / task / "aloha-agilex_clean_50" / "data"
    if variant_rest.startswith("aloha-agilex_randomized_500"):
        return DATASET_ROOT / task / "aloha-agilex_randomized_500" / "data"
    if variant_rest.startswith("demo_clean-recovery"):
        return DATASET_RECOVERY_ROOT / task / "demo_clean" / "data"
    if variant_rest.startswith("demo_randomized-recovery"):
        return DATASET_RECOVERY_ROOT / task / "demo_randomized" / "data"
    raise ValueError(f"未知数据后缀，无法映射 HDF5: task={task}, rest={variant_rest}")


def source_tag(folder_name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", folder_name).strip("_")
    return s[:96] if len(s) > 96 else s


def list_episode_ids(training_dir: Path) -> list[int]:
    ids: list[int] = []
    for d in training_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("episode_"):
            continue
        try:
            ids.append(int(d.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build multi-task recovery_3 judge dataset (ShareGPT).")
    parser.add_argument(
        "--training_root",
        type=str,
        default=str(root / "training_data" / "pi05_multi_task_recovery_3"),
        help="pi05_multi_task_recovery_3 根目录",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(root / "judge_dataset" / "recovery_judge_3tasks"),
        help="输出目录（images/ + JSON）",
    )
    parser.add_argument("--seed", type=int, default=42, help="合并打乱随机种子")
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="合并时不打乱顺序（recovery 与 success 按目录、episode 顺序拼接）",
    )
    parser.add_argument(
        "--no_variants",
        action="store_true",
        help="error 样本每类只用词库首句，不随机",
    )
    parser.add_argument(
        "--errors_only",
        action="store_true",
        help="只生成 recovery(错误)样本，不生成正常(success)样本。",
    )
    parser.add_argument(
        "--max_success_total",
        type=int,
        default=0,
        help="限制 success 样本总数（0 表示不限制）。常用：500。会按 6 个 success 数据源（3任务xclean/rand）尽量均分抽样。",
    )
    parser.add_argument(
        "--success_seed",
        type=int,
        default=42,
        help="success 抽样随机种子（与 --seed 分开，便于固定 success 采样）。",
    )
    args = parser.parse_args()

    training_root = Path(args.training_root)
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if not training_root.is_dir():
        raise SystemExit(f"training_root 不存在: {training_root}")

    error_items: list[dict] = []
    success_items: list[dict] = []
    skipped: list[str] = []

    subdirs = sorted([d for d in training_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    rng = random.Random(args.seed)

    # --- success 抽样预算 ---
    # 仅针对正常(success)目录：包含 aloha-agilex_* 且不包含 recovery
    success_subdirs: list[Path] = []
    for d in subdirs:
        n = d.name
        if "recovery" in n:
            continue
        if "aloha-agilex_" in n:
            success_subdirs.append(d)
    success_subdirs = sorted(success_subdirs, key=lambda p: p.name)

    # 每个源可抽取的 episode 数上限（按现有 episode_ 目录）
    success_pool_sizes = {d.name: len(list_episode_ids(d)) for d in success_subdirs}

    def _is_clean_source(dir_name: str) -> bool:
        return "aloha-agilex_clean_50" in dir_name

    def _is_rand_source(dir_name: str) -> bool:
        return "aloha-agilex_randomized_500" in dir_name

    def compute_success_budget() -> dict[str, int]:
        if args.errors_only:
            return {d.name: 0 for d in success_subdirs}
        if args.max_success_total is None or args.max_success_total <= 0:
            # 不限制：允许抽全部
            return {d.name: success_pool_sizes.get(d.name, 0) for d in success_subdirs}
        # 规则：先把 clean 的 3 个源全部抽完（理论上 3*50=150），剩下的从 3 个 randomized 源补齐
        total = int(args.max_success_total)
        budgets: dict[str, int] = {d.name: 0 for d in success_subdirs}
        if not success_subdirs:
            return budgets

        clean_dirs = [d for d in success_subdirs if _is_clean_source(d.name)]
        rand_dirs = [d for d in success_subdirs if _is_rand_source(d.name)]

        # 1) clean：全取（受 pool size 限制）
        clean_total = 0
        for d in clean_dirs:
            take = success_pool_sizes.get(d.name, 0)
            budgets[d.name] = take
            clean_total += take

        remaining = max(0, total - clean_total)
        if remaining <= 0 or not rand_dirs:
            return budgets

        # 2) randomized：均分 remaining，最后把余数从前往后补 1
        base = remaining // len(rand_dirs)
        rem = remaining % len(rand_dirs)
        for i, d in enumerate(rand_dirs):
            want = base + (1 if i < rem else 0)
            cap = success_pool_sizes.get(d.name, 0)
            budgets[d.name] = min(want, cap)

        # 3) 若某个 randomized 源 cap 不够，尝试把缺口继续分配到其他 randomized 源
        allocated = sum(budgets[d.name] for d in rand_dirs)
        short = remaining - allocated
        if short > 0:
            for d in rand_dirs:
                cap = success_pool_sizes.get(d.name, 0)
                extra_room = cap - budgets[d.name]
                if extra_room <= 0:
                    continue
                add = min(extra_room, short)
                budgets[d.name] += add
                short -= add
                if short <= 0:
                    break

        return budgets

    success_budget = compute_success_budget()
    success_rng = random.Random(args.success_seed)
    # 预先为每个 success 源挑选 episode id（避免遍历时“全都生成”）
    success_selected: dict[str, set[int]] = {}
    for d in success_subdirs:
        cap = success_pool_sizes.get(d.name, 0)
        want = min(success_budget.get(d.name, 0), cap)
        if want <= 0:
            success_selected[d.name] = set()
            continue
        eps = list_episode_ids(d)
        # 稳定随机抽样
        success_rng.shuffle(eps)
        success_selected[d.name] = set(eps[:want])

    for sub in subdirs:
        folder_name = sub.name
        try:
            task, variant_rest = parse_training_folder(folder_name)
            raw_dir = raw_hdf5_dir(task, variant_rest)
        except ValueError as e:
            print(f"[skip dir] {folder_name}: {e}")
            continue
        if not raw_dir.is_dir():
            print(f"[skip dir] {folder_name}: raw 不存在 {raw_dir}")
            continue

        tag = source_tag(folder_name)

        # 如果是 success 源且设置了抽样预算，则跳过未被选中的 episode
        is_success_source = (folder_name in success_selected) and ("recovery" not in folder_name)

        for ep_id in list_episode_ids(sub):
            if args.errors_only and ("recovery" not in folder_name):
                continue
            if is_success_source and ep_id not in success_selected[folder_name]:
                continue
            instr_path = sub / f"episode_{ep_id}" / "instructions.json"
            if not instr_path.is_file():
                skipped.append(f"{folder_name}/episode_{ep_id}: no instructions.json")
                continue
            with open(instr_path, "r", encoding="utf-8") as f:
                instr_data = json.load(f)
            phase_info = instr_data.get("phase_info") or {}
            err_range = phase_info.get("error_attempt_range")

            if err_range and len(err_range) >= 2:
                meta = load_episode_meta(sub, ep_id)
                if not meta:
                    skipped.append(f"{folder_name}/episode_{ep_id}: recovery but no valid phase meta")
                    continue

                paths = save_images_for_episode(
                    raw_dir,
                    ep_id,
                    meta["err_start"],
                    meta["err_end"],
                    IMAGE_KEYS,
                    image_dir,
                    source_tag=tag,
                )
                if not paths:
                    skipped.append(f"{folder_name}/episode_{ep_id}: failed read h5 {raw_dir}/episode{ep_id}.hdf5")
                    continue
                start_paths, end_paths = paths
                item = build_sharegpt_item(
                    meta["high_instruction"],
                    start_paths,
                    end_paths,
                    meta["error_type"],
                    use_variants=not args.no_variants,
                )
                error_items.append(item)
            else:
                if args.errors_only:
                    continue
                meta_s = load_episode_instructions(instr_path)
                if not meta_s:
                    skipped.append(f"{folder_name}/episode_{ep_id}: success but no checkpoints/total_steps")
                    continue
                n_ref = len(SUCCESS_VARIANTS["reflection"])
                reflex_idx = rng.randint(0, n_ref - 1)
                reflection = SUCCESS_VARIANTS["reflection"][reflex_idx]
                end_frame = choose_end_frame(
                    meta_s["checkpoints"],
                    meta_s["total_steps"],
                    reflex_idx,
                )
                paths = save_success_frames(
                    raw_dir,
                    ep_id,
                    0,
                    end_frame,
                    image_dir,
                    source_tag=tag,
                )
                if not paths:
                    skipped.append(f"{folder_name}/episode_{ep_id}: failed read h5 {raw_dir}/episode{ep_id}.hdf5")
                    continue
                start_paths, end_paths = paths
                item = build_success_sharegpt_item(
                    meta_s["high_instruction"],
                    start_paths,
                    end_paths,
                    reflection,
                )
                success_items.append(item)

    merged = error_items + success_items
    if not args.no_shuffle:
        rng.shuffle(merged)

    out_dir.mkdir(parents=True, exist_ok=True)
    err_path = out_dir / "recovery_judge_errors.json"
    suc_path = out_dir / "recovery_judge_success.json"
    merged_path = out_dir / "recovery_judge_dataset_merged.json"

    with open(err_path, "w", encoding="utf-8") as f:
        json.dump(error_items, f, indent=2, ensure_ascii=False)
    with open(suc_path, "w", encoding="utf-8") as f:
        json.dump(success_items, f, indent=2, ensure_ascii=False)
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"wrote errors: {len(error_items)} -> {err_path}")
    print(f"wrote success: {len(success_items)} -> {suc_path}")
    print(f"wrote merged: {len(merged)} -> {merged_path}")
    if skipped:
        skip_log = out_dir / "skipped.txt"
        skip_log.write_text("\n".join(skipped), encoding="utf-8")
        print(f"skipped {len(skipped)} episodes, see {skip_log}")


if __name__ == "__main__":
    main()
