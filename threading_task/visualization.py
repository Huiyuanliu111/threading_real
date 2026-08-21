"""Shared video helpers for Threading dataset playback and policy evaluation."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected an image, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image[:3], 0, -1)
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.size and image.max() <= 1.5:
            image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def compose_views(
    agentview: np.ndarray,
    eye_in_hand: np.ndarray,
    lines: list[str] | None = None,
    size: int = 256,
) -> np.ndarray:
    return compose_camera_views(
        [agentview, eye_in_hand],
        ["agentview", "robot0_eye_in_hand"],
        lines=lines,
        size=size,
    )


def compose_camera_views(
    images: list[np.ndarray],
    labels: list[str],
    lines: list[str] | None = None,
    size: int = 256,
) -> np.ndarray:
    if not images or len(images) != len(labels):
        raise ValueError("images and labels must be non-empty and have equal length")
    views = [
        cv2.resize(as_rgb_uint8(image), (size, size), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    frame = np.concatenate(views, axis=1)
    for index, label in enumerate(labels):
        cv2.putText(
            frame,
            label,
            (index * size + 8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
    for index, line in enumerate(lines or []):
        y = size - 10 - (len(lines or []) - index - 1) * 20
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return frame


class VideoWriter:
    """Streaming RGB MP4 writer that never buffers a complete episode."""

    def __init__(self, path: str | Path, fps: int = 20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    def append(self, frame: np.ndarray) -> None:
        frame = as_rgb_uint8(frame)
        if self._writer is None:
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {self.path}")
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
