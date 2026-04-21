#!/usr/bin/env python3
"""
统计 recovery 数据集中每个 episode 的 error_attempt / recovery / normal 的起止帧，
按四种错误类型汇总，并写入 Markdown 报告。
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def load_phase_frames(metadata_path: Path) -> dict | None:
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    summary = meta.get("episode_summary") or {}
    pf = meta.get("phase_frames") or {}
    total = summary.get("total_frames")
    if total is None:
        total = (pf.get("recovery_end_frame") or 0) + 1
    def _int(v, default=0):
        if v is None:
            return default
        return int(v)

    return {
        "episode_id": summary.get("episode_id"),
        "total_frames": int(total),
        "error_types": summary.get("error_types") or [],
        "error_attempt_start": _int(pf.get("error_attempt_start_frame"), 0),
        "error_attempt_end": _int(pf.get("error_attempt_end_frame"), 0),
        "recovery_start": _int(pf.get("recovery_start_frame"), 0),
        "recovery_end": _int(pf.get("recovery_end_frame"), 0),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze recovery phase frames and write MD report.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/data1/liujingzhi/RoboTwin/policy/pi05/processed_data/beat_block_hammer-demo_clean-recovery-66",
        help="Processed recovery data dir (must contain metadata/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .md path; default: <data_dir>/recovery_phase_frames_report.md",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    metadata_dir = data_dir / "metadata"
    if not metadata_dir.exists():
        print(f"Error: metadata not found: {metadata_dir}")
        return 1

    # 收集所有 episode 的 metadata（按 episode 编号排序）
    def ep_num(p: Path) -> int:
        m = re.search(r"episode(\d+)_metadata\.json", p.name)
        return int(m.group(1)) if m else -1

    meta_files = sorted(metadata_dir.glob("episode*_metadata.json"), key=ep_num)
    rows = []
    by_error_type = defaultdict(list)

    for p in meta_files:
        d = load_phase_frames(p)
        if d is None:
            continue
        ep_id = d["episode_id"]
        total = d["total_frames"]
        err_s, err_e = d["error_attempt_start"], d["error_attempt_end"]
        rec_s, rec_e = d["recovery_start"], d["recovery_end"]
        error_types = d["error_types"] or ["unknown"]

        # normal_pre_error: [0, err_s) 仅当 err_s > 0
        if err_s > 0:
            normal_pre = (0, err_s - 1)
            n_pre_len = err_s
        else:
            normal_pre = None
            n_pre_len = 0

        # error_attempt: [err_s, err_e]
        err_len = max(0, err_e - err_s + 1)

        # recovery: [rec_s, rec_e]
        rec_len = max(0, rec_e - rec_s + 1)

        # normal_post_recovery: (rec_e, total-1]
        if rec_e < total - 1:
            normal_post = (rec_e + 1, total - 1)
            n_post_len = total - 1 - rec_e
        else:
            normal_post = None
            n_post_len = 0

        et = error_types[0] if error_types else "unknown"
        by_error_type[et].append({
            "episode_id": ep_id,
            "total_frames": total,
            "error_attempt": (err_s, err_e, err_len),
            "recovery": (rec_s, rec_e, rec_len),
            "normal_pre": (normal_pre, n_pre_len),
            "normal_post": (normal_post, n_post_len),
        })
        rows.append({
            "episode_id": ep_id,
            "error_type": et,
            "total_frames": total,
            "normal_pre": normal_pre,
            "normal_pre_len": n_pre_len,
            "error_attempt": (err_s, err_e),
            "error_attempt_len": err_len,
            "recovery": (rec_s, rec_e),
            "recovery_len": rec_len,
            "normal_post": normal_post,
            "normal_post_len": n_post_len,
        })

    # 输出 Markdown
    out_path = Path(args.output) if args.output else data_dir / "recovery_phase_frames_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Recovery 阶段帧统计报告")
    lines.append("")
    lines.append(f"数据目录: `{data_dir}`")
    lines.append(f"Episode 数量: {len(rows)}")
    lines.append("")

    # 按错误类型汇总
    lines.append("## 1. 按错误类型汇总")
    lines.append("")
    for et in ["premature_close", "grasp_slip", "grasp_position_offset", "grasp_orientation_mismatch"]:
        if et not in by_error_type:
            continue
        items = by_error_type[et]
        lines.append(f"### {et}")
        lines.append("")
        lines.append(f"- Episode 数: {len(items)}")
        # 平均
        err_starts = [x["error_attempt"][0] for x in items]
        err_ends = [x["error_attempt"][1] for x in items]
        rec_starts = [x["recovery"][0] for x in items]
        rec_ends = [x["recovery"][1] for x in items]
        err_lens = [x["error_attempt"][2] for x in items]
        rec_lens = [x["recovery"][2] for x in items]
        n_pre_lens = [x["normal_pre"][1] for x in items]
        n_post_lens = [x["normal_post"][1] for x in items]
        lines.append(f"- **error_attempt** 平均: 开始帧 {sum(err_starts)/len(err_starts):.1f}, 结束帧 {sum(err_ends)/len(err_ends):.1f}, 长度 {sum(err_lens)/len(err_lens):.1f}")
        lines.append(f"- **recovery** 平均: 开始帧 {sum(rec_starts)/len(rec_starts):.1f}, 结束帧 {sum(rec_ends)/len(rec_ends):.1f}, 长度 {sum(rec_lens)/len(rec_lens):.1f}")
        lines.append(f"- **normal_pre_error** 平均长度: {sum(n_pre_lens)/len(n_pre_lens):.1f}")
        lines.append(f"- **normal_post_recovery** 平均长度: {sum(n_post_lens)/len(n_post_lens):.1f}")
        lines.append("")

    # 全表：每个 episode
    lines.append("## 2. 每个 Episode 的 phase 起止帧")
    lines.append("")
    lines.append("| episode_id | error_type | total_frames | normal_pre (起, 止) | error_attempt (起, 止) | recovery (起, 止) | normal_post (起, 止) |")
    lines.append("|------------|------------|--------------|---------------------|------------------------|-------------------|----------------------|")
    for r in rows:
        np_str = f"{r['normal_pre'][0]}-{r['normal_pre'][1]}" if r["normal_pre"] else "-"
        ea_str = f"{r['error_attempt'][0]}-{r['error_attempt'][1]}"
        rc_str = f"{r['recovery'][0]}-{r['recovery'][1]}"
        npost_str = f"{r['normal_post'][0]}-{r['normal_post'][1]}" if r["normal_post"] else "-"
        lines.append(f"| {r['episode_id']} | {r['error_type']} | {r['total_frames']} | {np_str} | {ea_str} | {rc_str} | {npost_str} |")
    lines.append("")

    # 四种错误类型是否起止不同
    lines.append("## 3. 四种错误类型的起止对比")
    lines.append("")
    lines.append("| error_type | error_attempt 平均(起, 止) | recovery 平均(起, 止) | 说明 |")
    lines.append("|------------|----------------------------|------------------------|------|")
    for et in ["premature_close", "grasp_slip", "grasp_position_offset", "grasp_orientation_mismatch"]:
        if et not in by_error_type:
            continue
        items = by_error_type[et]
        es = sum(x["error_attempt"][0] for x in items) / len(items)
        ee = sum(x["error_attempt"][1] for x in items) / len(items)
        rs = sum(x["recovery"][0] for x in items) / len(items)
        re_avg = sum(x["recovery"][1] for x in items) / len(items)
        lines.append(f"| {et} | ({es:.0f}, {ee:.0f}) | ({rs:.0f}, {re_avg:.0f}) | 各类型起止不同，见上表 |")
    lines.append("")

    lines.append("## 4. 子任务标注与训练设计说明")
    lines.append("")
    lines.append("- **仅对 normal 段和 recovery 段做子任务标注**：error_attempt 段不参与动作监督（用 action_mask 屏蔽）。")
    lines.append("- 标注方式与原有一致：对 normal_pre_error 与 recovery 两段分别用任务处理器（如 BeatBlockHammerProcessor）跑阶段切分，再合并 checkpoints，并写入 `action_mask`。")
    lines.append("- 运行子任务标注（recovery 专用）:")
    lines.append("  ```bash")
    lines.append("  python scripts/process_data_generic_recovery.py --task_name beat_block_hammer --data_dir processed_data/beat_block_hammer-demo_clean-recovery-66")
    lines.append("  ```")
    lines.append("- 画图（用 batch 脚本对整目录画图）:")
    lines.append("  ```bash")
    lines.append('  python "scripts/batch_analyze_trajectories_beat_block_hammer .py" --data_dir processed_data/beat_block_hammer-demo_clean-recovery-66 --output_dir analysis_results/analysis_results_beat_block_hammer_recovery')
    lines.append("  ```")
    lines.append("")
    lines.append("### 训练时的两个问题与处理方式")
    lines.append("")
    lines.append("**问题 A：随机 batch 时，recovery 帧不知道要加哪条 correction**")
    lines.append("")
    lines.append("- 做法：在**构建数据集时**为每个 episode 固定写入该 episode 的 `error_type` 和对应的 `correction` 文本（或从模板根据 error_type 生成）。")
    lines.append("- 每个 (observation, action) 样本带上 `episode_id`（或 segment_id）；对于属于 recovery 段的样本，文本输入 = `[Correction] <该 episode 的 correction> [Subtask] <当前 phase 的 subtask>`。")
    lines.append("- 这样随机 batch 里只要拿到的是 recovery 帧，就通过 episode_id 查表得到 correction，无需同一 batch 内同时出现 error_attempt 与 recovery。")
    lines.append("")
    lines.append("**问题 B：Reflection VLM 需要 error_attempt 的起止帧对作为输入，但 batch 是随机单帧**")
    lines.append("")
    lines.append("- 做法：**Reflection 头单独一个数据集**，不按“随机单帧”采样。")
    lines.append("- 构建一个 **ReflectionDataset**：每个样本 = (image_start, image_end, instruction_context)，标签 = (reflection, correction)；其中 (image_start, image_end) 固定为该 episode 的 error_attempt_start_frame 与 error_attempt_end_frame。")
    lines.append("- 训练时用这个数据集单独训 Reflection 头（或双图 VLM），batch 为随机的 (start, end) 帧对，无需与 Action Expert 的随机帧 batch 一致。")
    lines.append("")
    lines.append("更完整的训练设计说明见: `docs/RECOVERY_TRAINING_DESIGN.md`")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
