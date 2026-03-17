import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]
        
        # Check if we have existing image_masks from FrameStack (multi-frame mode)
        existing_masks = data.get("image_masks", {})
        
        # Determine if multi-frame mode based on image shape
        # Before batching: (n_obs_steps, h, w, c) - 4D
        # After batching: (batch, n_obs_steps, h, w, c) - 5D
        # Single-frame: (h, w, c) - 3D or (batch, h, w, c) - 4D
        is_multi_frame = base_image.ndim == 5 or (base_image.ndim == 4 and base_image.shape[0] > 3)
        
        # Determine mask shape based on image shape
        if is_multi_frame:
            if base_image.ndim == 5:
                # Already batched: (batch, n_obs_steps, h, w, c)
                n_obs_steps = base_image.shape[1]
                mask_shape = (base_image.shape[0], n_obs_steps)  # (batch, n_obs_steps)
            elif base_image.ndim == 4:
                # Single sample: (n_obs_steps, h, w, c)
                n_obs_steps = base_image.shape[0]
                mask_shape = (n_obs_steps,)  # (n_obs_steps,)
            else:
                mask_shape = None
        else:
            # Single-frame mode
            mask_shape = None
            n_obs_steps = 1

        images = {
            "base_0_rgb": base_image,
        }
        
        # Use existing mask if available (from FrameStack), otherwise create new one
        # FrameStack creates masks with keys like "cam_high", map them to "base_0_rgb"
        if "cam_high" in existing_masks:
            mask = existing_masks["cam_high"]
            # Ensure mask has correct shape
            if is_multi_frame and mask.ndim == 1 and base_image.ndim == 5:
                # Single sample mask (n_obs_steps,) needs to be expanded for batch
                # This shouldn't happen in practice, but handle it gracefully
                mask = np.broadcast_to(mask, mask_shape)
            image_masks = {
                "base_0_rgb": mask,
            }
        else:
            if mask_shape is not None:
                # Multi-frame: create mask with correct shape
                image_masks = {
                    "base_0_rgb": np.ones(mask_shape, dtype=bool),
                }
            else:
                # Single-frame: scalar mask
                image_masks = {
                    "base_0_rgb": np.True_,
                }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                # Use existing mask if available
                if source in existing_masks:
                    mask = existing_masks[source]
                    # Ensure mask has correct shape
                    if is_multi_frame and mask.ndim == 1 and base_image.ndim == 5:
                        mask = np.broadcast_to(mask, mask_shape)
                    image_masks[dest] = mask
                else:
                    if mask_shape is not None:
                        image_masks[dest] = np.ones(mask_shape, dtype=bool)
                    else:
                        image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                if mask_shape is not None:
                    image_masks[dest] = np.zeros(mask_shape, dtype=bool)
                else:
                    image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Preserve additional fields needed for subtask training
        for key in ["instructions", "subtasks", "frame_idx", "phase_info", "high_prompt", "low_prompt", "progress_label"]:
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Handle both single-frame and multi-frame cases
        # Single-frame: (c, h, w) or (h, w, c)
        # Multi-frame: (n_frames, c, h, w) or (n_frames, h, w, c)
        if img.ndim == 3:
            # Single frame: (c, h, w) or (h, w, c)
            if img.shape[0] == 3 or img.shape[0] == 1:
                # (c, h, w) format - convert to (h, w, c)
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                return einops.rearrange(img, "c h w -> h w c")
            else:
                # Already (h, w, c) format
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                return img
        elif img.ndim == 4:
            # Multi-frame: (n_frames, c, h, w) or (n_frames, h, w, c)
            if img.shape[1] == 3 or img.shape[1] == 1:
                # (n_frames, c, h, w) format - convert to (n_frames, h, w, c)
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                return einops.rearrange(img, "t c h w -> t h w c")
            else:
                # Already (n_frames, h, w, c) format
                if np.issubdtype(img.dtype, np.floating):
                    img = (255 * img).astype(np.uint8)
                return img
        else:
            # Unexpected shape, try to handle gracefully
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            return img

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """Decode Aloha state, handling both single-frame and multi-frame cases.
    
    Args:
        state: State array, shape (dim,) for single-frame or (n_frames, dim) for multi-frame
        adapt_to_pi: Whether to adapt to pi internal runtime format
    
    Returns:
        Decoded state with same shape as input
    """
    if not adapt_to_pi:
        return state
    
    # Handle multi-frame case: state shape is (n_frames, dim)
    if state.ndim == 2:
        # Multi-frame: apply decoding to each frame
        decoded_state = np.zeros_like(state)
        for i in range(state.shape[0]):
            frame_state = state[i]
            # Flip the joints.
            frame_state = _joint_flip_mask() * frame_state
            # Reverse the gripper transformation that is being applied by the Aloha runtime.
            frame_state[[6, 13]] = _gripper_to_angular(frame_state[[6, 13]])
            decoded_state[i] = frame_state
        return decoded_state
    else:
        # Single-frame case: state shape is (dim,)
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
        return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
