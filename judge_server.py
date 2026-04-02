"""
Run Judge model as an independent HTTP service.

Example:
  python judge_server.py \
    --host 0.0.0.0 --port 18080 \
    --base-model /path/to/Qwen2-VL-7B-Instruct \
    --lora-path /path/to/lora/checkpoint \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from PIL import Image

from judge_client import JudgeClient


def _decode_base64_png_to_numpy(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("utf-8"))
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def build_handler(judge: JudgeClient):
    class JudgeHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_json({"ok": True})
                return
            self._send_json({"ok": False, "error": f"unknown path: {self.path}"}, status=404)

        def do_POST(self):
            if self.path != "/judge":
                self._send_json({"ok": False, "error": f"unknown path: {self.path}"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                data = json.loads(body)

                instruction = data["instruction"]
                start_images = {k: _decode_base64_png_to_numpy(v) for k, v in data["start_images"].items()}
                current_images = {k: _decode_base64_png_to_numpy(v) for k, v in data["current_images"].items()}

                result = judge.judge(start_images, current_images, instruction)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)

        # Keep terminal output clean.
        def log_message(self, format, *args):
            return

    return JudgeHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    judge = JudgeClient(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
    )

    server = ThreadingHTTPServer((args.host, args.port), build_handler(judge))
    print(f"[JudgeServer] Listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
