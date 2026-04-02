#!/usr/bin/env python3
"""
Parallel multi-task LeRobot generation:
1) Split a multi-task raw dir by immediate child task directories.
2) Convert each task directory to a temporary LeRobot repo in parallel.
3) Merge temporary repos into the final output repo.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_LEROBOT_ROOT = REPO_ROOT / "lerobot"
CONVERT_SCRIPT = REPO_ROOT / "examples" / "aloha_real" / "convert_aloha_data_to_lerobot_robotwin.py"
MERGE_SCRIPT = REPO_ROOT / "scripts" / "merge_lerobot_datasets.py"


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def _contains_hdf5(root: Path) -> bool:
    return any(root.rglob("*.hdf5"))


def _discover_task_dirs(raw_dir: Path) -> list[Path]:
    task_dirs = []
    for child in sorted(raw_dir.iterdir()):
        if child.is_dir() and _contains_hdf5(child):
            task_dirs.append(child)
    if not task_dirs and _contains_hdf5(raw_dir):
        task_dirs = [raw_dir]
    return task_dirs


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    env = dict(os.environ)
    if LOCAL_LEROBOT_ROOT.is_dir():
        # Prepend so it wins over site-packages.
        env["PYTHONPATH"] = f"{LOCAL_LEROBOT_ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(cmd, cwd=str(cwd), check=False, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _convert_one(task_dir: Path, temp_repo_id: str) -> str:
    cmd = [
        "uv",
        "run",
        str(CONVERT_SCRIPT),
        "--raw_dir",
        str(task_dir),
        "--repo_id",
        temp_repo_id,
    ]
    print(f"[convert] {task_dir.name} -> {temp_repo_id}")
    _run_cmd(cmd, REPO_ROOT)
    print(f"[done] {temp_repo_id}")
    return temp_repo_id


def _build_temp_repo_ids(final_repo_id: str, task_dirs: list[Path]) -> list[str]:
    temp_ids = []
    for idx, task_dir in enumerate(task_dirs):
        task_name = _safe_name(task_dir.name)
        temp_ids.append(f"{final_repo_id}__part_{idx:03d}__{task_name}")
    return temp_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel convert + merge for multi-task datasets.")
    parser.add_argument("--raw_dir", required=True, type=Path)
    parser.add_argument("--repo_id", required=True, help="Final merged repo_id")
    parser.add_argument("--workers", type=int, default=3, help="Parallel convert workers")
    parser.add_argument(
        "--keep_temp_repos",
        action="store_true",
        help="Keep temporary repos after merge for debugging",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    task_dirs = _discover_task_dirs(raw_dir)
    if not task_dirs:
        raise RuntimeError(f"No task dirs or hdf5 found under: {raw_dir}")

    print(f"[info] raw_dir={raw_dir}")
    print(f"[info] discovered {len(task_dirs)} task dirs")
    for d in task_dirs:
        print(f"  - {d.name}")

    temp_repo_ids = _build_temp_repo_ids(args.repo_id, task_dirs)

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(task_dirs))) as ex:
        fut_to_repo = {
            ex.submit(_convert_one, task_dir, temp_repo_id): temp_repo_id
            for task_dir, temp_repo_id in zip(task_dirs, temp_repo_ids)
        }
        for fut in as_completed(fut_to_repo):
            repo = fut_to_repo[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{repo}: {exc}")

    if failures:
        print("[error] some conversions failed:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print(f"[merge] {len(temp_repo_ids)} repos -> {args.repo_id}")
    merge_cmd = [
        "uv",
        "run",
        str(MERGE_SCRIPT),
        "--output_repo",
        args.repo_id,
        "--source_repos",
        *temp_repo_ids,
    ]
    _run_cmd(merge_cmd, REPO_ROOT)

    if not args.keep_temp_repos:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", REPO_ROOT / ".cache")) / "huggingface" / "lerobot"
        for rid in temp_repo_ids:
            p = cache_root / rid
            if p.exists():
                print(f"[cleanup] remove {p}")
                shutil.rmtree(p, ignore_errors=True)

    print("[ok] done")


if __name__ == "__main__":
    main()
