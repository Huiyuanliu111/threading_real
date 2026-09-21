"""Physical units: state=[xyz, rotation columns 0/1], action=TCP delta."""

import dataclasses

import numpy as np

FRONT = "observation.images.exterior_image_1_left"
SIDE = "observation.images.exterior_image_2_right"
STATE_NAMES = [
    "tcp_x", "tcp_y", "tcp_z",
    "tcp_rotation_col0_x", "tcp_rotation_col0_y", "tcp_rotation_col0_z",
    "tcp_rotation_col1_x", "tcp_rotation_col1_y", "tcp_rotation_col1_z",
]
ACTION_NAMES = ["dx", "dy", "dz", "drotvec_x", "drotvec_y", "drotvec_z"]


def parse_image(value):
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}")
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Floating RGB images must be finite and in [0, 1]")
        image = np.rint(image * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError("Images must be uint8 or float in [0, 1]")
    return image


@dataclasses.dataclass(frozen=True)
class ThreadingInputs:
    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (9,) or not np.isfinite(state).all():
            raise ValueError("Expected finite 9D tcp_pose_6d state")
        front, side = parse_image(data["front"]), parse_image(data["side"])
        result = {
            "state": state,
            # The second exterior camera occupies the model's second image slot.
            # This name does not change its physical meaning into a wrist camera.
            "image": {"base_0_rgb": front, "left_wrist_0_rgb": side,
                      "right_wrist_0_rgb": np.zeros_like(front)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_,
                           "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            actions = np.array(data["actions"], dtype=np.float32, copy=True)
            if actions.ndim != 2 or actions.shape[-1] != 6 or not np.isfinite(actions).all():
                raise ValueError("Expected finite [horizon, 6] Cartesian delta actions")
            result["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = prompt.decode() if isinstance(prompt, bytes) else prompt
        return result


@dataclasses.dataclass(frozen=True)
class ThreadingOutputs:
    def __call__(self, data):
        actions = np.array(data["actions"][..., :6], copy=True)
        return {"actions": actions}


def validate_info(info):
    if info["fps"] != 30:
        raise ValueError("Expected a 30 Hz dataset; do not silently change the action timeline")
    features = info["features"]
    for key, names in [("observation.state", STATE_NAMES), ("action", ACTION_NAMES)]:
        if features[key]["shape"] != [len(names)] or features[key].get("names") != names:
            raise ValueError(f"Unexpected physical semantics for {key}: {features[key]}")
    for key in (FRONT, SIDE):
        if key not in features:
            raise ValueError(f"Missing camera: {key}")
