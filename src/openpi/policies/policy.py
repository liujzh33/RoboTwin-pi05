from collections import deque
from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._predict_progress = getattr(model, "predict_progress", None)
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            # Keep a separate RNG stream for progress probing so monitoring does not alter action sampling.
            self._progress_rng = jax.random.key(1)
            if hasattr(model, "predict_progress"):
                self._predict_progress = nnx_utils.module_jit(model.predict_progress)
            else:
                self._predict_progress = None
        
        # Initialize observation history queues for multi-frame stacking
        # Check if model has n_obs_steps config (for pi0/pi05 models)
        self._n_obs_steps = getattr(model, 'config', None) and getattr(model.config, 'n_obs_steps', 1) or 1
        if self._n_obs_steps > 1:
            # Initialize empty queues, will be populated dynamically based on input keys
            self._observation_queues = {
                "images": {},
                "image_masks": {},
                "state": deque(maxlen=self._n_obs_steps),
            }
        else:
            self._observation_queues = None

    def reset(self):
        """Reset observation history queues. Should be called on env.reset()."""
        if self._observation_queues is not None:
            for key in list(self._observation_queues["images"].keys()):
                self._observation_queues["images"][key].clear()
                self._observation_queues["image_masks"][key].clear()
            self._observation_queues["state"].clear()

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        
        # Handle multi-frame stacking if enabled
        if self._observation_queues is not None and self._n_obs_steps > 1:
            # Add current observation to queues
            current_images = inputs.get("image", {})
            current_masks = inputs.get("image_mask", {})
            current_state = inputs.get("state")
            
            # Update queues (copy first frame if queue not full)
            # Dynamically create queues for keys that appear in the input
            for key in current_images:
                if key not in self._observation_queues["images"]:
                    self._observation_queues["images"][key] = deque(maxlen=self._n_obs_steps)
                    self._observation_queues["image_masks"][key] = deque(maxlen=self._n_obs_steps)
                
                queue = self._observation_queues["images"][key]
                mask_queue = self._observation_queues["image_masks"][key]
                if len(queue) < self._n_obs_steps:
                    # Initialize by copying the first observation
                    while len(queue) < self._n_obs_steps:
                        queue.append(np.asarray(current_images[key]))
                        mask_queue.append(np.asarray(current_masks.get(key, True)))
                else:
                    # Add latest observation
                    queue.append(np.asarray(current_images[key]))
                    mask_queue.append(np.asarray(current_masks.get(key, True)))
            
            if current_state is not None:
                state_queue = self._observation_queues["state"]
                if len(state_queue) < self._n_obs_steps:
                    while len(state_queue) < self._n_obs_steps:
                        state_queue.append(np.asarray(current_state))
                else:
                    state_queue.append(np.asarray(current_state))
            
            # Stack observations from queues
            stacked_images = {}
            stacked_masks = {}
            for key in self._observation_queues["images"]:
                if len(self._observation_queues["images"][key]) > 0:
                    # Stack along time dimension: [n_obs_steps, h, w, c] -> [1, n_obs_steps, h, w, c]
                    stacked_images[key] = np.stack(list(self._observation_queues["images"][key]), axis=0)[np.newaxis, ...]
                    stacked_masks[key] = np.stack(list(self._observation_queues["image_masks"][key]), axis=0)[np.newaxis, ...]
            
            if len(self._observation_queues["state"]) > 0:
                stacked_state = np.stack(list(self._observation_queues["state"]), axis=0)[np.newaxis, ...]
            else:
                stacked_state = current_state[np.newaxis, ...] if current_state is not None else None
            
            # Replace inputs with stacked observations
            inputs["image"] = stacked_images
            inputs["image_mask"] = stacked_masks
            if stacked_state is not None:
                inputs["state"] = stacked_state
            
            # For multi-frame mode, ensure tokenized_prompt has batch dimension
            # Tokenized prompt is the same for all frames, so we don't stack it in time dimension
            # Just add batch dimension if missing
            for key in ["tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"]:
                if key in inputs and inputs[key] is not None:
                    arr = inputs[key]
                    # If 1D, add batch dimension to make it [1, l]
                    if isinstance(arr, np.ndarray) and arr.ndim == 1:
                        inputs[key] = arr[np.newaxis, ...]
        
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            # If already batched (from stacking), don't add another batch dimension
            if self._observation_queues is not None and self._n_obs_steps > 1:
                inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            else:
                inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            if self._observation_queues is not None and self._n_obs_steps > 1:
                inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
            else:
                inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        subtask_text = None
        if (
            self._is_pytorch_model
            and "tokenized_prompt_subtask_prefix" in inputs
            and hasattr(self._model, "sample_subtask")
            and self._metadata.get("tokenizer") is not None
        ):
            tokenizer = self._metadata["tokenizer"]
            stop_ids = inputs["subtask_stop_token_ids"]
            if hasattr(stop_ids, "numpy"):
                stop_ids = np.asarray(stop_ids)
            else:
                stop_ids = np.asarray(stop_ids)
            subtask_inputs = {
                **inputs,
                "tokenized_prompt": inputs["tokenized_prompt_subtask_prefix"],
                "tokenized_prompt_mask": inputs["tokenized_prompt_subtask_prefix_mask"],
                "token_ar_mask": inputs["tokenized_prompt_subtask_prefix_mask"],
            }
            obs_subtask = _model.Observation.from_dict(subtask_inputs)
            gen_ids = self._model.sample_subtask(
                sample_rng_or_pytorch_device, obs_subtask, stop_ids, max_decode_len=64
            )
            if gen_ids:
                subtask_text = tokenizer.detokenize(np.array(gen_ids)).strip()
            else:
                subtask_text = ""

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        if subtask_text is not None:
            outputs["subtask_text"] = subtask_text
        if self._is_pytorch_model and "tokenized_prompt" in inputs:
            tok = inputs["tokenized_prompt"]
            if hasattr(tok, "detach"):
                outputs["debug_tokens"] = np.asarray(tok[0, :20].detach().cpu())
            else:
                outputs["debug_tokens"] = np.asarray(tok[0, :20])
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def get_progress(self, obs: dict) -> float | None:
        """Predict progress ∈ [0,1] from raw observation dict. Returns None if model has no progress head."""
        if self._predict_progress is None:
            return None

        # Keep behavior consistent with infer(): transforms may mutate inputs in-place.
        # Always copy first to avoid corrupting caller-side observation buffers.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            observation = _model.Observation.from_dict(inputs)
            p = self._predict_progress(self._pytorch_device, observation)
            return float(np.asarray(p[0].detach().cpu()))
        else:
            inputs = jax.tree.map(lambda x: np.asarray(x)[None, ...], inputs)
            observation = _model.Observation.from_dict(inputs)
            self._progress_rng, progress_rng = jax.random.split(self._progress_rng)
            p = self._predict_progress(progress_rng, observation)
            return float(np.asarray(p[0]))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
