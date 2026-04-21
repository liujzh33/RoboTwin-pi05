"""
Judge Model client: calls Qwen2-VL + LoRA to diagnose grasp errors.

Usage:
    from judge_client import JudgeClient
    client = JudgeClient(base_model_path, lora_path, device="cuda:0")
    result = client.judge(start_images, current_images, instruction)
    # result = {"error_type": "premature_close",
    #           "reflection": "...",
    #           "correction": "Release, descend to the correct height, and grasp."}
"""

import re
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger("judge_client")

JUDGE_PROMPT_TEMPLATE = (
    "Task: {instruction}\n\n"
    "Look at the start state (first 3 images) and the current state (last 3 images). "
    "What went wrong? Output exactly in this format:\n"
    "[error_type]: <one of: grasp_position_offset, grasp_orientation_mismatch, premature_close, grasp_slip, None>\n"
    "[reflection]: <short reflection>\n"
    "[correction]: <short correction>"
)


def _parse_judge_output(text: str) -> dict:
    """Parse the three-line output from the judge model."""
    result = {"error_type": None, "reflection": None, "correction": None, "raw": text}

    m_error = re.search(r"\[error_type\]:\s*(.+)", text)
    m_reflect = re.search(r"\[reflection\]:\s*(.+)", text)
    m_correct = re.search(r"\[correction\]:\s*(.+)", text)

    if m_error:
        result["error_type"] = m_error.group(1).strip()
    if m_reflect:
        result["reflection"] = m_reflect.group(1).strip()
    if m_correct:
        raw_correction = m_correct.group(1).strip()
        raw_correction = re.sub(r"^\[Correction\]\s*", "", raw_correction)
        result["correction"] = raw_correction if raw_correction.lower() != "none" else None

    return result


class JudgeClient:
    """Loads Qwen2-VL + LoRA once and provides a .judge() method."""

    def __init__(
        self,
        base_model_path: str,
        lora_path: str | None = None,
        device: str = "cuda:0",
    ):
        logger.info(f"Loading Judge model from {base_model_path} (LoRA: {lora_path})")

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from peft import PeftModel

        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        if lora_path is not None:
            logger.info(f"Merging LoRA adapter from {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()
        self.device = device
        logger.info("Judge model loaded successfully.")

    @torch.inference_mode()
    def judge(
        self,
        start_images: dict[str, np.ndarray],
        current_images: dict[str, np.ndarray],
        instruction: str,
    ) -> dict:
        """
        Args:
            start_images: {"cam_high": HWC_uint8, "cam_left_wrist": ..., "cam_right_wrist": ...}
            current_images: same keys
            instruction: high-level task instruction string

        Returns:
            dict with keys: error_type, reflection, correction, raw
        """
        cam_order = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        pil_images = []
        for key in cam_order:
            pil_images.append(Image.fromarray(start_images[key]))
        for key in cam_order:
            pil_images.append(Image.fromarray(current_images[key]))

        text_prompt = JUDGE_PROMPT_TEMPLATE.format(instruction=instruction)

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text_input],
            images=pil_images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        prompt_len = inputs["input_ids"].shape[1]
        output_text = self.processor.batch_decode(
            generated_ids[:, prompt_len:], skip_special_tokens=True
        )[0]

        logger.info(f"Judge raw output:\n{output_text}")
        return _parse_judge_output(output_text)
