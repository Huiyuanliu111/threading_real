"""Fixed pixel ROIs, applied before resizing. Arrays may be RGB or BGR."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METADATA_FILE = "visual_preprocessing.json"
CAMERAS = {"cam1": "sideview", "cam3": "frontview"}


def validate(config: dict) -> dict:
    if config.get("version") != 1 or config.get("output_size") != 224:
        raise ValueError("Expected version=1 and output_size=224")
    if config.get("resize") != "letterbox":
        raise ValueError("resize must be letterbox")
    if set(config.get("cameras", {})) != set(CAMERAS):
        raise ValueError("Exactly cam1 and cam3 must be configured")
    for name, camera in config["cameras"].items():
        size, box = camera.get("source_size"), camera.get("roi")
        if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError(f"{name}: source_size must be [width, height] in pixels")
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError(f"{name}: roi must be integer [x0, y0, x1, y1]")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
            raise ValueError(f"{name}: ROI is empty or outside source image")
    return config


def load(path: str | Path) -> dict:
    return validate(json.loads(Path(path).read_text()))


def save(config: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(validate(config), indent=2) + "\n")


def transform(image: np.ndarray, camera: str, config: dict) -> np.ndarray:
    import cv2

    spec = config["cameras"][camera]
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{camera}: expected a uint8 HxWx3 raw image")
    if [image.shape[1], image.shape[0]] != spec["source_size"]:
        raise ValueError(f"{camera}: raw resolution {image.shape[1]}x{image.shape[0]} "
                         f"does not match configured {spec['source_size']}; do not crop resized images")
    x0, y0, x1, y1 = spec["roi"]
    crop = image[y0:y1, x0:x1]
    size = config["output_size"]
    scale = size / max(crop.shape[:2])
    width, height = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    output = np.zeros((size, size, 3), dtype=np.uint8)
    left, top = (size - width) // 2, (size - height) // 2
    output[top:top + height, left:left + width] = resized
    return output


def make_reader(base_reader, config):
    """Adapt the raw converter's decoder without changing timestamp alignment."""
    class CroppedReader(base_reader):
        def read(self, target):
            # Never replace the base reader's cached raw frame with a cropped frame.
            return transform(super().read(target), self.path.stem, config)
    return CroppedReader


def make_observation_adapter(original, config):
    def observe(sideview_rgb, wrist_rgb, frontview_rgb, q, gripper_width,
                image_size, pre_resize_image_size=None, tcp_position=None):
        if image_size != config["output_size"] or pre_resize_image_size is not None:
            raise ValueError("Crop deployment requires image_size=224 and no pre-resize")
        return original(transform(sideview_rgb, "cam1", config), wrist_rgb,
                        transform(frontview_rgb, "cam3", config), q, gripper_width,
                        image_size, None, tcp_position=tcp_position)
    return observe
