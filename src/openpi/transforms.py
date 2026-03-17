from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space.
    
    Supports both single-frame and multi-frame state inputs.
    For multi-frame state, uses the last frame (current frame) to compute deltas.
    """

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        state = np.asarray(state)
        actions = np.asarray(actions)
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        
        # Handle multi-frame state: use the last frame (current frame) for delta computation
        if state.ndim == 2:
            # Multi-frame state: (n_frames, dim) -> use last frame
            current_state = state[-1]  # Shape: (dim,)
        else:
            # Single-frame state: (dim,)
            current_state = state
        
        # Compute delta: actions are relative to current state
        # actions shape: (action_horizon, action_dim)
        # current_state[:dims] shape: (dims,)
        # We need to broadcast: (action_horizon, dims) - (dims,) -> (action_horizon, dims)
        state_delta = np.where(mask, current_state[:dims], 0)  # Shape: (dims,)
        actions[..., :dims] -= state_delta  # Broadcast: (action_horizon, dims) - (dims,)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space.
    
    Supports both single-frame and multi-frame state inputs.
    For multi-frame state, uses the last frame (current frame) to compute absolutes.
    """

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        state = np.asarray(state)
        actions = np.asarray(actions)
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        
        # Handle multi-frame state: use the last frame (current frame) for absolute computation
        if state.ndim == 2:
            # Multi-frame state: (n_frames, dim) -> use last frame
            current_state = state[-1]  # Shape: (dim,)
        else:
            # Single-frame state: (dim,)
            current_state = state
        
        # Compute absolute: actions are absolute, add current state
        # actions shape: (action_horizon, action_dim)
        # current_state[:dims] shape: (dims,)
        # We need to broadcast: (action_horizon, dims) + (dims,) -> (action_horizon, dims)
        state_abs = np.where(mask, current_state[:dims], 0)  # Shape: (dims,)
        actions[..., :dims] += state_abs  # Broadcast: (action_horizon, dims) + (dims,)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks, loss_mask = self.tokenizer.tokenize(prompt, state)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class TokenizeHighLowPrompt(DataTransformFn):
    """Tokenize high-level prompt and low-level (subtask) prompt for subtask generation."""
    tokenizer: _tokenizer.PaligemmaTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        # 推理时（缺少 high_prompt 和 low_prompt），使用普通的 prompt
        high_prompt = data.pop("high_prompt", None)
        low_prompt = data.pop("low_prompt", None)
        
        if high_prompt is None or low_prompt is None:
            # 推理模式：使用普通的 prompt 字段
            if (prompt := data.pop("prompt", None)) is None:
                raise ValueError("Either (high_prompt, low_prompt) or prompt is required")
            
            if not isinstance(prompt, str):
                prompt = prompt.item()
            
            # 使用普通的 tokenize 方法（返回 3 个值：tokens, mask, loss_mask）
            tokens, token_masks, loss_mask = self.tokenizer.tokenize(prompt, state=data.get("state", None))
            return {
                **data,
                "tokenized_prompt": tokens,
                "tokenized_prompt_mask": token_masks,
                "token_loss_mask": loss_mask,  # 推理时也保留 loss_mask（虽然不会用到）
            }
        
        # 训练模式：使用 high_prompt 和 low_prompt
        if not isinstance(high_prompt, str):
            high_prompt = high_prompt.item()
        if not isinstance(low_prompt, str):
            low_prompt = low_prompt.item()

        # Remove other text fields that JAX can't handle
        data.pop("instructions", None)
        data.pop("subtasks", None)
        data.pop("phase_info", None)

        tokens, token_masks, ar_mask, loss_mask = self.tokenizer.tokenize_high_low_prompt(high_prompt, low_prompt)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class LoadSubtaskFromInstructions(DataTransformFn):
    """
    Load high-level prompt and low-level (subtask) prompt from instructions.json.
    
    Expects data to have 'instructions', 'subtasks', 'frame_idx', and 'phase_info' keys.
    Selects the appropriate subtask based on the current frame's phase.
    """
    use_first_instruction: bool = True  # If True, use first instruction; if False, randomly select

    def __call__(self, data: DataDict) -> DataDict:
        import json
        
        # 推理时（没有 instructions 字段），直接跳过此 transform
        if "instructions" not in data:
            return data
        if "subtasks" not in data:
            return data
        if "frame_idx" not in data:
            return data
        if "phase_info" not in data:
            return data

        # Parse JSON strings if needed
        instructions = data["instructions"]
        if isinstance(instructions, str):
            instructions = json.loads(instructions)
        
        subtasks = data["subtasks"]
        if isinstance(subtasks, str):
            subtasks = json.loads(subtasks)
        
        frame_idx = data["frame_idx"]
        if isinstance(frame_idx, np.ndarray):
            frame_idx = int(frame_idx.item())
        elif not isinstance(frame_idx, int):
            frame_idx = int(frame_idx)
        
        phase_info = data["phase_info"]
        if isinstance(phase_info, str):
            phase_info = json.loads(phase_info)

        if len(instructions) == 0:
            raise ValueError("instructions list is empty")
        if len(subtasks) == 0:
            raise ValueError("subtasks list is empty")
        if len(instructions) != len(subtasks):
            raise ValueError(f"instructions ({len(instructions)}) and subtasks ({len(subtasks)}) must have same length")

        # Select instruction (randomly or first one)
        if self.use_first_instruction:
            idx = 0
        else:
            idx = np.random.randint(0, len(instructions))

        high_prompt = instructions[idx]
        
        # Determine current phase based on frame_idx and checkpoints
        checkpoints = phase_info.get("checkpoints", [])
        phase_idx = 0
        for i, checkpoint in enumerate(checkpoints):
            if frame_idx < checkpoint:
                phase_idx = i
                break
        else:
            phase_idx = len(checkpoints)  # Last phase
        
        # Select the subtask corresponding to the current phase
        if isinstance(subtasks[idx], list) and len(subtasks[idx]) > 0:
            # Clamp phase_idx to valid range
            phase_idx = min(phase_idx, len(subtasks[idx]) - 1)
            low_prompt = subtasks[idx][phase_idx]  # Use phase-specific subtask
        else:
            raise ValueError(f"subtasks[{idx}] must be a non-empty list")

        return {
            **{k: v for k, v in data.items() if k not in ["instructions", "subtasks"]},
            "high_prompt": high_prompt,
            "low_prompt": low_prompt,
        }


@dataclasses.dataclass(frozen=True)
class ComputeProgressLabel(DataTransformFn):
    """
    动态根据 frame_idx 和 phase_info.total_steps 生成进度标签 progress_label ∈ [0, 1]。
    
    不修改底层数据集，只在 dataloader 采样时附加一个标量标签，供模型的进度检测头训练使用。
    """

    def __call__(self, data: DataDict) -> DataDict:
        if "frame_idx" not in data or "phase_info" not in data:
            return data

        frame_idx = data["frame_idx"]
        if isinstance(frame_idx, np.ndarray):
            frame_idx = int(frame_idx.item())
        else:
            frame_idx = int(frame_idx)

        phase_info = data["phase_info"]
        if isinstance(phase_info, str):
            import json

            phase_info = json.loads(phase_info)

        total_steps = int(phase_info.get("total_steps", 0) or 0)
        denom = max(1, total_steps - 1)
        progress = np.asarray(frame_idx / float(denom), dtype=np.float32)

        data["progress_label"] = progress
        return data


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class FrameStack(DataTransformFn):
    """Stacks historical frames from delta_timestamps into a time dimension.
    
    This transform expects data from LeRobotDataset with delta_timestamps, where:
    - Images come as (n_frames, c, h, w) and need to be converted to (n_frames, h, w, c)
    - State comes as (n_frames, dim) and stays as (n_frames, dim)
    - Image masks are created based on padding flags from delta_timestamps
    
    After transformation:
    - Images: (n_frames, h, w, c) - ready for model input
    - State: (n_frames, dim) - ready for model input
    - Image masks: (n_frames,) - boolean mask indicating valid frames
    
    Note: This transform expects data AFTER repack, so images are in a nested dict:
    data["images"][key] for each camera key.
    """

    n_obs_steps: int
    image_keys: Sequence[str]

    def __call__(self, data: DataDict) -> DataDict:
        if self.n_obs_steps <= 1:
            return data
        
        # Process images: they are in a nested dict after repack: data["images"][key]
        if "images" in data and isinstance(data["images"], dict):
            images_dict = data["images"]
            image_masks_dict = {}
            
            for key in self.image_keys:
                if key in images_dict:
                    image = np.asarray(images_dict[key])  # Shape: (n_frames, c, h, w) from PyTorch
                    # Convert to (n_frames, h, w, c)
                    image = np.transpose(image, (0, 2, 3, 1))
                    images_dict[key] = image
                    
                    # Create image mask from padding flags
                    # The padding key is at the dataset level before repack: "observation.images.{cam_key}_is_pad"
                    # After repack, it might be at the root level or nested. Check both.
                    pad_key = f"observation.images.{key}_is_pad"
                    if pad_key in data:
                        # Invert: True means valid (not padded), False means padded
                        image_mask = ~np.asarray(data[pad_key])
                        # Remove the padding key as it's no longer needed
                        del data[pad_key]
                    else:
                        # Try alternative key format
                        pad_key_alt = f"{key}_is_pad"
                        if pad_key_alt in data:
                            image_mask = ~np.asarray(data[pad_key_alt])
                            del data[pad_key_alt]
                        else:
                            # If no padding info, assume all frames are valid
                            image_mask = np.ones(self.n_obs_steps, dtype=bool)
                    image_masks_dict[key] = image_mask
            
            # Store image masks in the expected format (will be processed by later transforms)
            if image_masks_dict:
                data["image_masks"] = image_masks_dict
        
        # Process state: ensure it has the right shape (n_frames, dim)
        if "state" in data:
            state = np.asarray(data["state"])
            if state.ndim == 1:
                # Single frame case - expand to (1, dim)
                state = state[np.newaxis, :]
            # State should already be (n_frames, dim) from delta_timestamps
            data["state"] = state
            
            # Handle state padding if available
            state_pad_key = "observation.state_is_pad"
            if state_pad_key in data:
                # Remove padding key as it's not used by the model
                del data[state_pad_key]
            else:
                # Try alternative key
                state_pad_key_alt = "state_is_pad"
                if state_pad_key_alt in data:
                    del data[state_pad_key_alt]
        
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
