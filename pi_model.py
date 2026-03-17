#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import os
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


# Resolve project root (RoboTwin) based on this file's location, so that
# checkpoint paths are robust to the current working directory.
_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))


class PI0:

    def __init__(
        self,
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        use_torch_checkpoint: bool = False,
        torch_checkpoint_dir: str | None = None,
        norm_stats_path: str | None = None,
    ):
        """
        Wrapper around OpenPI policy for RoboTwin evaluation.

        Args:
            train_config_name: Name of the OpenPI TrainConfig to use.
            model_name: Logical model name (kept for compatibility with JAX checkpoints).
            checkpoint_id: Checkpoint step id for the JAX checkpoint layout.
            pi0_step: Action horizon used during evaluation.
            use_torch_checkpoint: If True, load weights from the PyTorch checkpoint directory
                                  `policy/pi05/checkpoint_torch` instead of the default JAX
                                  checkpoint layout under `policy/pi05/checkpoints/.../30000`.
        """
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        if use_torch_checkpoint:
            # Use the consolidated Torch checkpoint directory by default, but allow overriding
            # so callers can point to a custom directory containing:
            #   - model.safetensors
            #   - assets/<asset_id>/...
            checkpoint_dir = (
                torch_checkpoint_dir
                if torch_checkpoint_dir is not None
                else os.path.join(_PROJECT_ROOT, "policy", "pi05", "checkpoint_torch")
            )
            assets_dir = os.path.join(checkpoint_dir, "assets")
        else:
            # Default: use the original JAX-style checkpoint layout.
            checkpoint_dir = os.path.join(
                _PROJECT_ROOT,
                "policy",
                "pi05",
                "checkpoints",
                self.train_config_name,
                self.model_name,
                str(self.checkpoint_id),
            )
            assets_dir = os.path.join(checkpoint_dir, "assets")

        # Optionally load norm stats directly from a JSON file to avoid depending on
        # `assets/<asset_id>/norm_stats.json` layout (useful when only weights + norm_stats.json are shipped).
        loaded_norm_stats = None
        candidate_norm_stats_file = None
        if norm_stats_path is not None:
            candidate_norm_stats_file = norm_stats_path
        else:
            maybe_local = os.path.join(checkpoint_dir, "norm_stats.json")
            if os.path.isfile(maybe_local):
                candidate_norm_stats_file = maybe_local

        if candidate_norm_stats_file is not None:
            try:
                loaded_norm_stats = _normalize.deserialize_json(
                    open(candidate_norm_stats_file, "r", encoding="utf-8").read()
                )
                print(f"\033[94m[PI05] Loaded norm stats from {candidate_norm_stats_file}\033[0m")
            except Exception as e:
                raise RuntimeError(f"Failed to load norm stats from {candidate_norm_stats_file}: {e}") from e

        assets_id = None
        if loaded_norm_stats is None:
            # Discover the asset id (used to locate normalization statistics / assets).
            entries = os.listdir(assets_dir)
            if not entries:
                raise RuntimeError(f"No assets found in {assets_dir}")
            assets_id = entries[0]

        # Let OpenPI's policy_config decide whether to load JAX or PyTorch weights
        # based on the presence of `model.safetensors` in the checkpoint directory.
        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            robotwin_repo_id=assets_id,
            norm_stats=loaded_norm_stats,
        )

        print(f"loading model success! (use_torch_checkpoint={use_torch_checkpoint})")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
