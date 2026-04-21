#!/usr/bin/env python3
"""
Inspect recovery metadata for task/settings and report:
1) per-episode error_type(s)
2) error_attempt_start/end and recovery_start/end
3) aggregated counts

Default targets:
  - blocks_ranking_rgb: demo_clean, demo_randomized
  - blocks_ranking_size: demo_clean, demo_randomized
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _episode_id_from_name(path: Path) -> int:
    name = path.stem  # episode123_metadata
    prefix = "episode"
    suffix = "_metadata"
    if name.startswith(prefix) and name.endswith(suffix):
        s = name[len(prefix) : -len(suffix)]
        if s.isdigit():
            return int(s)
    return -1


def inspect_one_dir(meta_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = sorted(meta_dir.glob("episode*_metadata.json"), key=_episode_id_from_name)
    rows: list[dict[str, Any]] = []
    error_counter: Counter[str] = Counter()
    start_counter: Counter[int] = Counter()

    for fp in files:
        data = _read_json(fp)
        summary = data.get("episode_summary") or {}
        phase = data.get("phase_frames") or {}

        episode_id = int(summary.get("episode_id", _episode_id_from_name(fp)))
        error_types = summary.get("error_types") or []
        if not isinstance(error_types, list):
            error_types = [str(error_types)]
        error_types = [str(x) for x in error_types if x is not None]

        err_start = phase.get("error_attempt_start_frame")
        err_end = phase.get("error_attempt_end_frame")
        rec_start = phase.get("recovery_start_frame")
        rec_end = phase.get("recovery_end_frame")

        for et in (error_types or ["<none>"]):
            error_counter[et] += 1
        if isinstance(err_start, int):
            start_counter[err_start] += 1

        rows.append(
            {
                "episode_id": episode_id,
                "error_types": "|".join(error_types) if error_types else "",
                "error_attempt_start_frame": err_start,
                "error_attempt_end_frame": err_end,
                "recovery_start_frame": rec_start,
                "recovery_end_frame": rec_end,
                "total_frames": summary.get("total_frames"),
                "success": summary.get("success"),
            }
        )

    agg = {
        "num_episodes": len(rows),
        "error_type_counts": dict(sorted(error_counter.items(), key=lambda kv: kv[0])),
        "error_attempt_start_counts": dict(sorted(start_counter.items(), key=lambda kv: kv[0])),
    }
    return rows, agg


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect dataset_recovery metadata.")
    parser.add_argument(
        "--dataset_recovery_root",
        type=str,
        default="/mnt/data1/liujingzhi/dataset_recovery",
        help="Root directory of dataset_recovery.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=["blocks_ranking_rgb", "blocks_ranking_size"],
        help="Task names to inspect.",
    )
    parser.add_argument(
        "--settings",
        type=str,
        nargs="*",
        default=["demo_clean", "demo_randomized"],
        help="Task settings to inspect.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis_results/recovery_metadata_inspection",
        help="Output directory for csv/json reports (relative to current working dir).",
    )
    args = parser.parse_args()

    root = Path(args.dataset_recovery_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_summary: dict[str, Any] = defaultdict(dict)

    for task in args.tasks:
        for setting in args.settings:
            meta_dir = root / task / setting / "metadata"
            key = f"{task}/{setting}"
            if not meta_dir.exists():
                print(f"[Skip] {key}: metadata dir not found -> {meta_dir}")
                continue

            rows, agg = inspect_one_dir(meta_dir)
            global_summary[task][setting] = agg

            csv_path = out_dir / f"{task}_{setting}_episodes.csv"
            write_csv(csv_path, rows)

            print(f"\n=== {key} ===")
            print(f"episodes: {agg['num_episodes']}")
            print(f"error_type_counts: {agg['error_type_counts']}")
            print(f"error_attempt_start_counts: {agg['error_attempt_start_counts']}")
            print(f"per-episode csv: {csv_path}")

    json_path = out_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
