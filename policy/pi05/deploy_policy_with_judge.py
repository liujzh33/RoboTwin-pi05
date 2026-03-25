"""
deploy_policy_with_judge.py
Drop-in replacement for deploy_policy.py that adds:
  - Real-time progress monitoring via Pi0.5's progress head
  - Automatic Judge Model triggering on anomaly
  - Correction injection into Pi0.5's language prompt
  - Start/trigger frame image saving for debugging
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import PI0

logger = logging.getLogger("deploy_judge")

# Confirm this module is loaded (not deploy_policy.py)
print("\033[92m[deploy_policy_with_judge] Module loaded successfully\033[0m")

# ─── ANSI color helpers ───
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_RED = "\033[91m"
_RESET = "\033[0m"

# ─── Progress monitor thresholds ───
PROGRESS_DROP_THRESHOLD = 0.15
PROGRESS_STALL_STEPS = 40
PROGRESS_OSCILLATION_RANGE = 0.10
MAX_TOTAL_STEPS = 150

# ─── Judge model default paths ───
DEFAULT_JUDGE_BASE = "/mnt/data1/liujingzhi/Qwen-VL-7B/qwen/Qwen2-VL-7B-Instruct"
DEFAULT_JUDGE_LORA = "/mnt/data1/liujingzhi/LlamaFactory/train_2026-03-17-11-55-59/checkpoint-600"
DEFAULT_JUDGE_SERVER_URL = "http://127.0.0.1:18080"

_judge_client = None


class ProgressMonitor:
    """Track progress predictions and detect anomalies."""

    def __init__(
        self,
        drop_threshold: float = PROGRESS_DROP_THRESHOLD,
        stall_steps: int = PROGRESS_STALL_STEPS,
        oscillation_range: float = PROGRESS_OSCILLATION_RANGE,
        max_steps: int = MAX_TOTAL_STEPS,
    ):
        self.drop_threshold = drop_threshold
        self.stall_steps = stall_steps
        self.oscillation_range = oscillation_range
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.history: list[float] = []
        self.peak: float = 0.0
        self.steps_since_advance: int = 0
        self.triggered: bool = False

    def update(self, progress: float) -> str | None:
        self.history.append(progress)
        step = len(self.history)

        if progress > self.peak + 0.01:
            self.peak = progress
            self.steps_since_advance = 0
        else:
            self.steps_since_advance += 1

        if self.peak - progress > self.drop_threshold:
            self.triggered = True
            return f"progress_drop (peak={self.peak:.3f}, now={progress:.3f})"

        if self.steps_since_advance >= self.stall_steps:
            self.triggered = True
            return f"progress_stall ({self.steps_since_advance} steps without advance)"

        if step >= self.stall_steps:
            recent = self.history[-self.stall_steps:]
            if max(recent) - min(recent) < self.oscillation_range:
                self.triggered = True
                return f"progress_oscillation (range={max(recent)-min(recent):.3f} over {self.stall_steps} steps)"

        if step >= self.max_steps:
            self.triggered = True
            return f"max_steps_exceeded ({step} >= {self.max_steps})"

        return None


def _get_judge_client():
    """Lazy-load judge client, remote service first by default."""
    global _judge_client
    if _judge_client is None:
        use_remote = os.environ.get("JUDGE_USE_REMOTE", "1").strip().lower() not in ("0", "false", "no")
        if use_remote:
            server_url = os.environ.get("JUDGE_SERVER_URL", DEFAULT_JUDGE_SERVER_URL)
            timeout_sec = float(os.environ.get("JUDGE_TIMEOUT_SEC", "60"))
            from judge_service_client import JudgeServiceClient

            _judge_client = JudgeServiceClient(server_url, timeout_sec=timeout_sec)
            health = _judge_client.health()
            print(f"{_GREEN}[Judge] Connected remote service: {server_url}, health={health}{_RESET}")
        else:
            judge_base = os.environ.get("JUDGE_BASE_MODEL", DEFAULT_JUDGE_BASE)
            judge_lora = os.environ.get("JUDGE_LORA_PATH", DEFAULT_JUDGE_LORA)
            judge_device = os.environ.get("JUDGE_DEVICE", "cuda:0")
            print(f"{_YELLOW}[Judge] Loading local Qwen2-VL + LoRA on {judge_device} ...{_RESET}")
            from judge_client import JudgeClient

            _judge_client = JudgeClient(judge_base, judge_lora, device=judge_device)
            print(f"{_GREEN}[Judge] Local model loaded successfully on {judge_device}{_RESET}")
    return _judge_client


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def _rgb_dict_from_obs(observation):
    """Extract {cam_high, cam_left_wrist, cam_right_wrist} as HWC uint8 arrays."""
    return {
        "cam_high": np.asarray(observation["observation"]["head_camera"]["rgb"], dtype=np.uint8),
        "cam_left_wrist": np.asarray(observation["observation"]["left_camera"]["rgb"], dtype=np.uint8),
        "cam_right_wrist": np.asarray(observation["observation"]["right_camera"]["rgb"], dtype=np.uint8),
    }


def _save_images(images: dict, save_dir: Path, prefix: str):
    """Save three-view images to disk for debugging."""
    from PIL import Image
    save_dir.mkdir(parents=True, exist_ok=True)
    for cam_name, img_arr in images.items():
        img = Image.fromarray(img_arr)
        path = save_dir / f"{prefix}_{cam_name}.png"
        img.save(path)
    print(f"  [Save] Images saved to {save_dir}/{prefix}_*.png")


# ─── State kept across eval() calls for one episode ───
_episode_state: dict = {}
_episode_counter: int = 0


def get_model(usr_args):
    train_config_name = usr_args["train_config_name"]
    model_name = usr_args["model_name"]
    checkpoint_id = usr_args["checkpoint_id"]
    pi0_step = usr_args["pi0_step"]
    model = PI0(train_config_name, model_name, checkpoint_id, pi0_step)
    # 预加载 judge（远端服务健康检查或本地模型加载），避免首次触发延迟。
    preload = os.environ.get("JUDGE_PRELOAD_AT_START", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if preload:
        print(
            f"{_YELLOW}[Judge] Preloading at eval start ...{_RESET}"
        )
        _get_judge_client()
        print(f"{_GREEN}[Judge] Preload done — ready for inference{_RESET}")
    return model


def eval(TASK_ENV, model, observation):
    global _episode_state, _episode_counter

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)
        print(f"[Task] {instruction}")

        # Save start frame images to disk
        start_images = _rgb_dict_from_obs(observation)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(parent_directory) / "judge_debug" / f"episode_{_episode_counter}_{timestamp}"
        _save_images(start_images, save_dir, "start")
        (save_dir / "instruction.txt").write_text(instruction, encoding="utf-8")
        print(f"  [Save] instruction.txt -> {save_dir / 'instruction.txt'}")

        _episode_state = {
            "instruction": instruction,
            "start_images": start_images,
            "save_dir": save_dir,
            "monitor": ProgressMonitor(),
            "correction_active": None,
            "oneshot_correction_pending_revert": False,
            "step_count": 0,
            "last_judge_step": -9999,
        }

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # Must run after observation_window is populated (not None)
    if "has_progress" not in _episode_state:
        hp = getattr(model.policy, "_predict_progress", None) is not None
        _episode_state["has_progress"] = hp
        if hp:
            print(f"{_GREEN}  [Progress] Progress head detected — monitoring enabled{_RESET}")
        else:
            print(f"{_YELLOW}  [Progress] Progress head NOT available — using step-count fallback{_RESET}")

    # ── Progress monitoring ──
    monitor: ProgressMonitor = _episode_state["monitor"]
    _episode_state["step_count"] += 1
    step_count = _episode_state["step_count"]

    trigger_reason = None

    if _episode_state.get("has_progress"):
        progress = model.policy.get_progress(model.observation_window)
        if progress is not None:
            trigger_reason = monitor.update(progress)
            if step_count % 10 == 0 or trigger_reason:
                print(f"  [Progress] step={step_count}, p={progress:.3f}, peak={monitor.peak:.3f}")
    else:
        # Fallback: use step count only (trigger at MAX_TOTAL_STEPS)
        if step_count >= MAX_TOTAL_STEPS and not monitor.triggered:
            trigger_reason = f"max_steps_exceeded ({step_count} >= {MAX_TOTAL_STEPS})"
            monitor.triggered = True

    # ── Judge trigger (cooldown avoids repeated calls every step on same stall) ──
    JUDGE_COOLDOWN_STEPS = 50
    can_call_judge = (
        trigger_reason
        and (step_count - _episode_state["last_judge_step"]) >= JUDGE_COOLDOWN_STEPS
    )
    if trigger_reason and not can_call_judge:
        if step_count % 20 == 0:
            print(
                f"  [Judge] trigger active ({trigger_reason}) but cooldown "
                f"({step_count - _episode_state['last_judge_step']}/{JUDGE_COOLDOWN_STEPS} steps since last call)"
            )

    if can_call_judge:
        _episode_state["last_judge_step"] = step_count
        print(f"{_YELLOW}  [TRIGGER] {trigger_reason}{_RESET}")

        # Save current trigger frame images to disk
        current_images = _rgb_dict_from_obs(observation)
        _save_images(current_images, _episode_state["save_dir"], f"trigger_step{step_count}")

        judge = _get_judge_client()
        result = judge.judge(
            _episode_state["start_images"],
            current_images,
            _episode_state["instruction"],
        )

        print(f"{_YELLOW}  [Judge] error_type : {result['error_type']}{_RESET}")
        print(f"{_YELLOW}  [Judge] reflection : {result['reflection']}{_RESET}")
        print(f"{_YELLOW}  [Judge] correction : {result['correction']}{_RESET}")

        if result["correction"] and result["error_type"] and result["error_type"].lower() != "none":
            correction_text = result["correction"]
            new_prompt = f"{_episode_state['instruction']} {correction_text}"
            model.set_language(new_prompt)
            _episode_state["correction_active"] = correction_text
            _episode_state["oneshot_correction_pending_revert"] = True
            print(f"{_GREEN}  [RECOVERY] Injected correction: {correction_text}{_RESET}")
            print(f"{_GREEN}  [RECOVERY] New prompt: {new_prompt}{_RESET}")

            monitor.reset()
            monitor.update(0.0)
            _episode_state["last_judge_step"] = -9999  # allow future triggers if still stuck
        else:
            print(f"{_CYAN}  [Judge] No error detected or no correction needed.{_RESET}")

    # ── Get actions ──
    actions = model.get_action()[:model.pi0_step]

    # One-shot correction: apply only to this inference call, then revert to original instruction.
    if _episode_state.get("oneshot_correction_pending_revert", False):
        model.set_language(_episode_state["instruction"])
        _episode_state["oneshot_correction_pending_revert"] = False
        _episode_state["correction_active"] = None
        print(f"{_CYAN}  [RECOVERY] One-shot correction consumed; prompt reverted to base instruction{_RESET}")

    if getattr(model, "_last_infer", None) is not None:
        last = model._last_infer
        if last.get("subtask_text") is not None:
            print(f"[Pi0.5 Subtask]: {last['subtask_text']}")

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    global _episode_state, _episode_counter
    _episode_counter += 1
    model.reset_obsrvationwindows()
    _episode_state = {}
