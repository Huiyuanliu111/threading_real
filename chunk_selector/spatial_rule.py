"""Training-trajectory-derived spatial soft labels, independent of motion speed."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def trajectory_progress(xyz: np.ndarray, mode: str = "arc_length") -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError("trajectory must contain at least two finite XYZ positions")
    if mode == "frames":
        return np.linspace(0, 1, len(xyz))
    if mode != "arc_length":
        raise ValueError("progress mode must be arc_length or frames")
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    if lengths[-1] <= 1e-9:
        raise ValueError("stationary trajectory has no arc-length progress")
    return lengths / lengths[-1]


@dataclass(frozen=True)
class SpatialRule:
    task: str
    center_xyz_m: tuple[float, float, float]
    boundary_radius_m: float
    transition_width_m: float = 0.02
    h: int = 3
    H: int = 10
    split_progress: float = 0.8
    progress_mode: str = "arc_length"
    version: str = "spatial_rule_v1"

    def __post_init__(self):
        if self.task not in {"threading", "maze"}:
            raise ValueError("task must be threading or maze")
        center = np.asarray(self.center_xyz_m)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("center must be a finite XYZ vector")
        if (not np.isfinite([self.boundary_radius_m, self.transition_width_m]).all()
                or not 0 < self.transition_width_m < 2 * self.boundary_radius_m):
            raise ValueError("require 0 < transition_width_m < 2 * boundary_radius_m")
        if self.h not in (3, 4, 5) or self.H != 10:
            raise ValueError("spatial rule requires h in (3,4,5) and H=10")
        if not 0 < self.split_progress < 1 or self.progress_mode not in {"arc_length", "frames"}:
            raise ValueError("invalid split progress or progress mode")

    @classmethod
    def fit(cls, trajectories, *, task, split_progress=None, progress_mode="arc_length",
            transition_width_m=0.02, h=3):
        """Fit a shared sphere using only the supplied training episodes.

        Median start/end XYZ is the center. Median distance of interpolated
        progress nodes to that center defines the 50% probability boundary.
        """
        if task not in {"threading", "maze"}:
            raise ValueError("task must be threading or maze")
        if split_progress is None:
            split_progress = .8 if task == "threading" else .3
        if not 0 < split_progress < 1:
            raise ValueError("split_progress must lie strictly between zero and one")
        paths = [np.asarray(path, dtype=float) for path in trajectories]
        if not paths:
            raise ValueError("at least one training trajectory is required")
        nodes = []
        for path in paths:
            progress = trajectory_progress(path, progress_mode)
            nodes.append([np.interp(split_progress, progress, path[:, axis]) for axis in range(3)])
        anchor = -1 if task == "threading" else 0
        center = np.median([path[anchor] for path in paths], axis=0)
        radius = float(np.median(np.linalg.norm(np.asarray(nodes) - center, axis=1)))
        return cls(task=task, center_xyz_m=tuple(center), boundary_radius_m=radius,
                   transition_width_m=transition_width_m, h=h,
                   split_progress=split_progress, progress_mode=progress_mode)

    def label(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("positions must be a finite [T,3] array")
        distance = np.linalg.norm(xyz - self.center_xyz_m, axis=1)
        p_fine = np.clip(.5 + (self.boundary_radius_m - distance) / self.transition_width_m, 0, 1)
        probabilities = np.column_stack((p_fine, 1 - p_fine)).astype(np.float32)
        expected = probabilities @ np.array([self.h, self.H], dtype=np.float32)
        return probabilities, expected, distance

    def to_dict(self):
        return asdict(self)
