"""Offline soft supervision from complete per-episode TCP arc length."""
from dataclasses import asdict, dataclass

import numpy as np

from .spatial_rule import trajectory_progress


@dataclass(frozen=True)
class ProgressRule:
    task: str
    split_progress: float
    h: int
    H: int
    transition_width: float = 0.1
    progress_mode: str = "arc_length"
    version: str = "arc_length_progress_v1"

    def __post_init__(self):
        if self.task not in {"threading", "maze"}:
            raise ValueError("task must be threading or maze")
        if not (0 < self.split_progress < 1 and 0 < self.transition_width
                <= 2 * min(self.split_progress, 1 - self.split_progress)):
            raise ValueError("transition must lie within episode progress [0,1]")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (self.h, self.H)) or not 0 < self.h < self.H:
            raise ValueError("require positive integer h < H")
        if self.progress_mode != "arc_length":
            raise ValueError("progress labels require TCP arc length")

    def label(self, xyz):
        progress = trajectory_progress(xyz)
        direction = 1 if self.task == "threading" else -1
        fine = np.clip(.5 + direction * (progress - self.split_progress)
                       / self.transition_width, 0, 1)
        probabilities = np.column_stack((fine, 1 - fine)).astype(np.float32)
        expected = probabilities @ np.array([self.h, self.H], dtype=np.float32)
        return probabilities, expected, progress

    def to_dict(self):
        return asdict(self)
