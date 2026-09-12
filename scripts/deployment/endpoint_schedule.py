"""Select an execution prefix using a demonstration-derived TCP endpoint."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class EndpointExecutionSchedule:
    endpoint_xyz_m: np.ndarray
    fine_radius_m: float = 0.06
    coarse_steps: int = 10
    fine_steps: int = 3
    fine_mode: bool = False

    def __post_init__(self):
        self.endpoint_xyz_m = np.asarray(self.endpoint_xyz_m, dtype=float)
        if self.endpoint_xyz_m.shape != (3,) or not np.isfinite(self.endpoint_xyz_m).all():
            raise ValueError("endpoint_xyz_m must be a finite XYZ vector")
        if not np.isfinite(self.fine_radius_m) or self.fine_radius_m <= 0:
            raise ValueError("fine_radius_m must be positive and finite")
        if any(isinstance(n, bool) or not isinstance(n, int)
               for n in (self.coarse_steps, self.fine_steps)):
            raise ValueError("execution step counts must be integers")
        if not 1 <= self.fine_steps <= self.coarse_steps:
            raise ValueError("require 1 <= fine_steps <= coarse_steps")

    @classmethod
    def from_file(cls, path: Path):
        data = yaml.safe_load(path.expanduser().read_text())
        if data.get("frame") != "panda_link0" or data.get("tcp_frame") != "panda_hand_tcp":
            raise ValueError("execution schedule must use panda_link0 / panda_hand_tcp")
        return cls(**{key: data[key] for key in
                      ("endpoint_xyz_m", "fine_radius_m", "coarse_steps", "fine_steps")})

    def reset(self):
        self.fine_mode = False

    def select(self, current_xyz, target_poses):
        """Inspect the clipped Cartesian path, including crossings between targets.

        The entire candidate path is checked before selection by the runner's
        existing action guards. Entering fine mode is latched until reset.
        """
        current = np.asarray(current_xyz, dtype=float)
        poses = np.asarray(target_poses, dtype=float)
        if (current.shape != (3,) or poses.shape != (self.coarse_steps, 4, 4)
                or not np.isfinite(current).all() or not np.isfinite(poses).all()):
            raise ValueError("schedule requires a finite current XYZ and full coarse pose chunk")
        path = np.concatenate((current[None], poses[:, :3, 3]))
        start, delta = path[:-1], np.diff(path, axis=0)
        length2 = (delta * delta).sum(axis=1)
        fraction = np.clip(((self.endpoint_xyz_m - start) * delta).sum(axis=1)
                           / np.maximum(length2, 1e-20), 0, 1)
        nearest = start + fraction[:, None] * delta
        distances = np.linalg.norm(nearest - self.endpoint_xyz_m, axis=1)
        current_distance = float(np.linalg.norm(current - self.endpoint_xyz_m))
        crossing = np.flatnonzero(distances <= self.fine_radius_m)
        was_fine = self.fine_mode
        self.fine_mode |= current_distance <= self.fine_radius_m or bool(len(crossing))
        steps = self.fine_steps if self.fine_mode else self.coarse_steps
        return steps, {
            "execution_phase": "fine" if self.fine_mode else "coarse",
            "execution_steps": steps,
            "endpoint_distance_m": current_distance,
            "predicted_path_min_distance_m": float(distances.min()),
            "fine_mode_just_entered": self.fine_mode and not was_fine,
            "first_fine_region_segment": int(crossing[0] + 1) if len(crossing) else None,
        }
