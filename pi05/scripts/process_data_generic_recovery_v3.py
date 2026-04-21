#!/usr/bin/env python3
"""
Recovery 逐帧密集标注 v3：按错误类型区分逻辑，并限制 [Correction] 仅在 recovery 段。

1. 非 grasp_slip（premature_close / grasp_position_offset / grasp_orientation_mismatch）：
   error_attempt 从 0 开始，只对 [rec_start, total_steps] 做四阶段划分，correction 仅填到 rec_end。
2. grasp_slip：先对 [0, err_start) 做四阶段但最多保留前 3 阶段（砍掉 T3 避免“敲击”幻觉），
   再对 [rec_start, total_steps] 做四阶段，correction 仅填到 rec_end。
3. 写入 phase_info.error_attempt_range 供画图时淡红背景使用。
"""

import sys
import json
import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, List

import h5py
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from task_definitions.base_task import BaseTaskProcessor

REFLECTION_TEMPLATES = {
    # 错误：位置偏了
    # 物理动作要素：张开夹爪 -> 调整XY位置 -> 重新抓取
    "grasp_position_offset": "[Correction] Open the gripper, adjust XY position, and retry grasping.",
    
    # 错误：姿态/角度不对
    # 物理动作要素：张开夹爪 -> 调整手腕旋转 -> 重新抓取
    "grasp_orientation_mismatch": "[Correction] Open the gripper, adjust wrist rotation, and retry grasping.",
    
    # 错误：在半空中提前闭合
    # 物理动作要素：张开夹爪 -> 向下移动到目标 -> 闭合
    "premature_close": "[Correction] Open the gripper, move down to the target, and close it.",
    
    # 错误：搬运途中打滑掉落
    # 物理动作要素：追踪掉落物体 -> 返回原位 -> 重新抓紧
    "grasp_slip": "[Correction] Track the dropped object, return to it, and regrasp."
}

RECOVERY_RAW_BASE = {
    "beat_block_hammer": "/mnt/data1/liujingzhi/dataset_recovery/beat_block_hammer/demo_clean/data",
    # demo_randomized 与 demo_clean 并列，用于扩充 recovery 训练样本
    "beat_block_hammer_demo_randomized": "/mnt/data1/liujingzhi/dataset_recovery/beat_block_hammer/demo_randomized/data",
}


class H5Slice:
    def __init__(self, h5_file, start: int, end: int):
        self._f = h5_file
        self._start = start
        self._end = end

    def __contains__(self, key):
        return key in self._f

    def __getitem__(self, key):
        arr = self._f[key][()]
        if hasattr(arr, "shape") and len(arr.shape) >= 1 and arr.shape[0] > 0:
            return arr[self._start : self._end]
        return arr


def _int(v, default=0):
    return int(v) if v is not None else default


def load_phase_frames(metadata_path: Path) -> Optional[dict]:
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    summary = meta.get("episode_summary") or {}
    pf = meta.get("phase_frames") or {}
    total_frames = summary.get("total_frames")
    if total_frames is None and pf:
        total_frames = _int(pf.get("recovery_end_frame"), 0) + 1
    return {
        "err_start": _int(pf.get("error_attempt_start_frame"), 0),
        "err_end": _int(pf.get("error_attempt_end_frame"), 0),
        "rec_start": _int(pf.get("recovery_start_frame"), 0),
        "rec_end": _int(pf.get("recovery_end_frame"), 0),
        "total_frames": int(total_frames) if total_frames is not None else None,
        "error_types": summary.get("error_types") or [],
    }


def get_raw_dual_arm_eef_recovery(
    raw_data_dir: Path, ep_num: int, total_steps: int, start: int, end: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    raw_path = raw_data_dir / f"episode{ep_num}.hdf5"
    if not raw_path.exists():
        return None, None
    try:
        with h5py.File(raw_path, "r") as raw_f:
            if "endpose/left_endpose" not in raw_f or "endpose/right_endpose" not in raw_f:
                return None, None
            l_ep = np.asarray(raw_f["endpose/left_endpose"][()])[:, :3][:total_steps][start:end]
            r_ep = np.asarray(raw_f["endpose/right_endpose"][()])[:, :3][:total_steps][start:end]
            return l_ep, r_ep
    except Exception:
        return None, None


def run_length_encode_subtasks_per_frame(spf: List[str]) -> Tuple[List[str], List[int]]:
    """
    从 subtasks_per_frame 做 run-length 编码，返回 (segment 标签列表, checkpoints)。
    - segment_labels: 每个 segment 的标签，与作图脚本逻辑一致（grasp_slip 前3+后4 等）
    - checkpoints: 各 phase 起始帧，phase_info.checkpoints 格式，供训练时 frame_idx 判断当前阶段
      checkpoints[i] = phase i+1 的起始帧；phase 0 为 [0, checkpoints[0])，phase k 为 [checkpoints[k-1], checkpoints[k])
    """
    if not spf:
        return [], []
    segment_labels = []
    segment_starts = []  # 每个 segment 的起始帧
    cur = spf[0]
    start = 0
    for i in range(1, len(spf) + 1):
        if i == len(spf) or spf[i] != cur:
            segment_labels.append(cur)
            segment_starts.append(start)
            if i < len(spf):
                start = i
                cur = spf[i]
    # checkpoints[i] = phase i+1 的起始帧（即 segment i+1 的 start）
    checkpoints = segment_starts[1:] if len(segment_starts) > 1 else []
    return segment_labels, checkpoints


def fill_subtasks_array_smart(
    subtask_array: List[str],
    start_idx: int,
    end_idx: int,
    local_cps: List[int],
    base_descs: List[str],
    prefix: str = "",
    rec_end_idx: int = -1,
) -> None:
    """填充子任务；仅当 global_idx <= rec_end_idx 时加 prefix，否则纯 desc（避免 correction 溢出到 normal_post）。"""
    current_phase = 0
    cp_idx = 0
    for i in range(end_idx - start_idx):
        global_idx = start_idx + i
        if cp_idx < len(local_cps) and i >= local_cps[cp_idx]:
            current_phase += 1
            cp_idx += 1
        desc = base_descs[min(current_phase, len(base_descs) - 1)]
        if prefix and rec_end_idx >= 0 and global_idx <= rec_end_idx:
            final_text = f"{prefix} [Subtask] {desc}"
        else:
            final_text = desc
        subtask_array[global_idx] = final_text


def run_processor_on_slice(
    hdf5_path: Path,
    task_processor: BaseTaskProcessor,
    task_name: str,
    raw_data_dir: Optional[Path],
    ep_num: int,
    total_steps: int,
    start: int,
    end: int,
    active_side: Optional[str] = None,
) -> List[int]:
    if start >= end:
        return []
    with h5py.File(hdf5_path, "r") as f:
        sl = H5Slice(f, start, end)
        if active_side is None and hasattr(task_processor, "analyzer"):
            l_g, r_g = task_processor.analyzer.extract_gripper_states(sl)
            active_side = "left" if (np.max(l_g) - np.min(l_g)) >= (np.max(r_g) - np.min(r_g)) else "right"
        ext_zl = ext_zr = ext_el = ext_er = None
        if raw_data_dir and task_name == "beat_block_hammer":
            ext_el, ext_er = get_raw_dual_arm_eef_recovery(raw_data_dir, ep_num, total_steps, start, end)
            if ext_el is not None:
                ext_zl = ext_el[:, 2]
            if ext_er is not None:
                ext_zr = ext_er[:, 2]
        try:
            cps = task_processor.get_phase_checkpoints(
                sl,
                active_side=active_side,
                external_z_left=ext_zl,
                external_z_right=ext_zr,
                external_eef_xyz_left=ext_el,
                external_eef_xyz_right=ext_er,
            )
        except TypeError:
            try:
                cps = task_processor.get_phase_checkpoints(sl, active_side=active_side)
            except TypeError:
                cps = task_processor.get_phase_checkpoints(sl)
        return task_processor.validate_checkpoints(cps, end - start)


def process_episode_recovery_v3(
    ep_dir: Path,
    meta_dir: Path,
    task_processor: BaseTaskProcessor,
    task_name: str,
    raw_data_dir: Optional[Path],
) -> bool:
    hdf5_files = list(ep_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        return False
    hdf5_path = hdf5_files[0]

    match = re.search(r"episode_(\d+)", ep_dir.name)
    ep_num = int(match.group(1)) if match else 0
    meta = load_phase_frames(meta_dir / f"episode{ep_num}_metadata.json")
    if not meta or meta["total_frames"] is None:
        print(f"  Skip {ep_dir.name}: no phase_frames/total_frames")
        return False

    with h5py.File(hdf5_path, "r") as f:
        total_steps = f["action"].shape[0] if "action" in f else f["observations/qpos"].shape[0]
    total_steps = min(total_steps, meta["total_frames"])

    err_start = min(meta["err_start"], total_steps)
    err_end = min(meta["err_end"], total_steps - 1) if meta["err_end"] >= total_steps else meta["err_end"]
    rec_start = min(meta["rec_start"], total_steps)
    rec_end = min(meta["rec_end"], total_steps - 1) if meta["rec_end"] >= total_steps else meta["rec_end"]
    error_type = meta["error_types"][0] if meta["error_types"] else ""

    action_mask = np.ones(total_steps, dtype=np.int32)
    action_mask[err_start : err_end + 1] = 0
    subtask_per_frame = [""] * total_steps
    base_descs = task_processor.get_subtask_descriptions()
    active_side = None

    is_grasp_slip = error_type == "grasp_slip"

    # 1. 仅 grasp_slip 有 normal_pre：[0, err_start)，最多 3 阶段（砍掉 T3，避免“敲击”幻觉）
    if is_grasp_slip and err_start > 0:
        cps_pre = run_processor_on_slice(
            hdf5_path, task_processor, task_name, raw_data_dir,
            ep_num, total_steps, 0, err_start, active_side,
        )
        cps_pre = cps_pre[:2]
        fill_subtasks_array_smart(subtask_per_frame, 0, err_start, cps_pre, base_descs, prefix="", rec_end_idx=-1)
        with h5py.File(hdf5_path, "r") as f:
            sl = H5Slice(f, 0, err_start)
            l_g, r_g = task_processor.analyzer.extract_gripper_states(sl)
            active_side = "left" if (np.max(l_g) - np.min(l_g)) >= (np.max(r_g) - np.min(r_g)) else "right"

    # 2. Error_Attempt 段
    #    - grasp_slip: 只在 [err_start, rec_start) 标为 MASKED，0->err_start 仍然是正常三阶段
    #    - 非 grasp_slip: 从 0 到 recovery_start 都当作 error_attempt，不做阶段标注
    if is_grasp_slip:
        for i in range(err_start, rec_start):
            subtask_per_frame[i] = "[MASKED] Error Attempt Phase"
    else:
        for i in range(0, rec_start):
            subtask_per_frame[i] = "[MASKED] Error Attempt Phase"

    # 3. Recovery + Normal_Post：[rec_start, total_steps]，四阶段；correction 只到 rec_end
    if rec_start < total_steps:
        cps_rec = run_processor_on_slice(
            hdf5_path, task_processor, task_name, raw_data_dir,
            ep_num, total_steps, rec_start, total_steps, active_side,
        )
        correction_prefix = REFLECTION_TEMPLATES.get(error_type, "")
        fill_subtasks_array_smart(
            subtask_per_frame, rec_start, total_steps, cps_rec, base_descs,
            prefix=correction_prefix, rec_end_idx=rec_end,
        )

    instructions_path = ep_dir / "instructions.json"
    if instructions_path.exists():
        with open(instructions_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    else:
        data = {"instructions": ["Default instruction"]}

    # 从 subtasks_per_frame run-length 编码得到阶段序列，与作图脚本逻辑一致（grasp_slip 前3+后4 等）
    segment_labels, checkpoints = run_length_encode_subtasks_per_frame(subtask_per_frame)
    num_phases = len(segment_labels)
    data["subtasks"] = [list(segment_labels) for _ in range(len(data.get("instructions", [])))]

    data["subtasks_per_frame"] = subtask_per_frame
    data["action_mask"] = action_mask.tolist()
    data["phase_info"] = {
        "total_steps": total_steps,
        "num_phases": num_phases,
        "num_phases_per_segment": len(base_descs),
        "checkpoints": checkpoints,
        "error_attempt_range": [err_start, err_end],
        "recovery_start": rec_start,
        "recovery_end": rec_end,
    }
    with open(instructions_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)

    print(f"  ✓ {ep_dir.name} error_type={error_type}, masked={err_end - err_start + 1}F, normal_post from {rec_end + 1}F")
    return True


def main():
    parser = argparse.ArgumentParser(description="Recovery 逐帧标注 v3（按错误类型分段 + correction 仅 recovery）")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--episode_range", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    metadata_dir = data_dir / "metadata"
    if not data_dir.exists() or not metadata_dir.exists():
        print("Error: data_dir or metadata not found")
        return 1

    raw_data_dir = None
    # demo_randomized 与 demo_clean：按 data_dir 名选 raw 路径
    if "demo_randomized" in data_dir.name and "beat_block_hammer_demo_randomized" in RECOVERY_RAW_BASE:
        raw_data_dir = Path(RECOVERY_RAW_BASE["beat_block_hammer_demo_randomized"])
    elif args.task_name in RECOVERY_RAW_BASE:
        raw_data_dir = Path(RECOVERY_RAW_BASE[args.task_name])
    if raw_data_dir is not None and not raw_data_dir.exists():
        raw_data_dir = None

    try:
        mod = __import__(f"task_definitions.{args.task_name}", fromlist=[None])
        name = args.task_name.title().replace("_", "") + "Processor"
        cls = getattr(mod, name, None)
        if cls is not None:
            task_processor = cls()
        else:
            task_processor = None
            for c in dir(mod):
                cls = getattr(mod, c)
                if isinstance(cls, type) and issubclass(cls, BaseTaskProcessor) and cls != BaseTaskProcessor:
                    task_processor = cls()
                    break
        if task_processor is None:
            print("Error: no task processor found")
            return 1
    except Exception as e:
        print(f"Error loading processor: {e}")
        return 1

    episode_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")],
        key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
    )
    if args.episode_range:
        if "-" in args.episode_range:
            a, b = map(int, args.episode_range.split("-"))
            episode_dirs = [episode_dirs[i] for i in range(a, min(b + 1, len(episode_dirs)))]
        else:
            idx = [int(x) for x in args.episode_range.split(",")]
            episode_dirs = [episode_dirs[i] for i in idx if 0 <= i < len(episode_dirs)]

    ok = 0
    for ep_dir in episode_dirs:
        if process_episode_recovery_v3(ep_dir, metadata_dir, task_processor, args.task_name, raw_data_dir):
            ok += 1
    print(f"\n✓ Done: {ok}/{len(episode_dirs)} episodes (v3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
