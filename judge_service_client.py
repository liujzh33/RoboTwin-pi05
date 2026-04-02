"""
HTTP client for calling external judge service.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from typing import Any

import numpy as np
from PIL import Image


def _encode_image_to_base64_png(img: np.ndarray) -> str:
    img = np.asarray(img, dtype=np.uint8)
    buff = io.BytesIO()
    Image.fromarray(img).save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


class JudgeServiceClient:
    """Client for a remote judge service process."""

    def __init__(self, base_url: str, timeout_sec: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def health(self) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/health",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def judge(
        self,
        start_images: dict[str, np.ndarray],
        current_images: dict[str, np.ndarray],
        instruction: str,
    ) -> dict[str, Any]:
        payload = {
            "instruction": instruction,
            "start_images": {k: _encode_image_to_base64_png(v) for k, v in start_images.items()},
            "current_images": {k: _encode_image_to_base64_png(v) for k, v in current_images.items()},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/judge",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
