"""Filtered high-resolution point-cloud dataset for the Threading MVT ARP."""
from __future__ import annotations

import copy
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from pushbox.diffusion_policy.dataset.base_dataset import BaseImageDataset
from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from threading_task.kinematics import PandaForwardKinematics


def pose_control_points(pose: np.ndarray, axis_length: float) -> np.ndarray:
    """Encode a rigid pose with origin, +X and +Y control points."""
    origin = pose[:3, 3]
    return np.stack((origin, origin + pose[:3, 0] * axis_length,
                     origin + pose[:3, 1] * axis_length)).astype(np.float32)


class ThreadingMVTDataset(BaseImageDataset):
    def __init__(self, dataset_path: str, urdf_path: str, horizon: int = 10,
                 n_obs_steps: int = 1, val_ratio: float = 0.2, seed: int = 42,
                 axis_length: float = 0.04, _validation: bool = False) -> None:
        super().__init__()
        if horizon <= 0 or n_obs_steps != 1:
            raise ValueError("MVT dataset requires horizon>0 and n_obs_steps=1")
        self.dataset_path = str(Path(dataset_path).expanduser().resolve())
        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        self.horizon, self.n_obs_steps = int(horizon), int(n_obs_steps)
        self.val_ratio, self.seed = float(val_ratio), int(seed)
        self.axis_length, self._validation = float(axis_length), bool(_validation)
        self._file = None
        fk = PandaForwardKinematics(self.urdf_path)
        with h5py.File(self.dataset_path, "r") as source:
            if source.attrs.get("format") != "threading-mvt-pointcloud-v1":
                raise ValueError(f"unsupported MVT dataset {self.dataset_path}")
            self.bounds = np.asarray(source.attrs["bounds_m"], np.float32)
            self.episode_keys = sorted(source.keys())
            self.states, self.actions, self.current_points, self.target_points = [], [], [], []
            self.target_grippers, self.lengths = [], []
            for key in self.episode_keys:
                state = source[key]["observation_state"][:].astype(np.float32)
                action = source[key]["action"][:].astype(np.float32)
                current, target, gripper = [], [], []
                for s, a in zip(state, action, strict=True):
                    pose = fk.pose(s[:7])
                    target_pose = pose.copy()
                    target_pose[:3, 3] += a[:3]
                    target_pose[:3, :3] = Rotation.from_rotvec(a[3:6]).as_matrix() @ pose[:3, :3]
                    current.append(pose_control_points(pose, self.axis_length))
                    target.append(pose_control_points(target_pose, self.axis_length))
                    gripper.append(s[7] + a[6])
                self.states.append(state); self.actions.append(action)
                self.current_points.append(np.stack(current)); self.target_points.append(np.stack(target))
                self.target_grippers.append(np.asarray(gripper, np.float32)); self.lengths.append(len(state))

        count = len(self.episode_keys)
        val_count = max(1, round(count * self.val_ratio))
        val_set = set(np.random.default_rng(self.seed).permutation(count)[:val_count].tolist())
        selected = val_set if self._validation else set(range(count)) - val_set
        self.selected_episodes = sorted(selected)
        self.sample_indices = [(ep, row) for ep in self.selected_episodes
                               for row in range(max(0, self.lengths[ep] - self.horizon + 1))]

    def _h5(self):
        if self._file is None: self._file = h5py.File(self.dataset_path, "r", swmr=True)
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy(); state["_file"] = None; return state

    def __len__(self): return len(self.sample_indices)

    def __getitem__(self, index):
        episode, row = self.sample_indices[index]
        group = self._h5()[self.episode_keys[episode]]
        ids = np.arange(row, row + self.horizon)
        clipped = np.minimum(ids, self.lengths[episode] - 1)
        return {
            "obs": {
                "points": torch.from_numpy(group["points"][row].astype(np.float32)[None]),
                "colors": torch.from_numpy(group["colors"][row].astype(np.float32)[None] / 255.0),
                "valid_points": torch.tensor([int(group["valid_points"][row])]),
                "agent_pos": torch.from_numpy(self.states[episode][row:row+1]),
                "control_points": torch.from_numpy(self.current_points[episode][row:row+1]),
            },
            "action": torch.from_numpy(self.actions[episode][clipped]),
            "target_control_points": torch.from_numpy(self.target_points[episode][clipped]),
            "target_gripper": torch.from_numpy(self.target_grippers[episode][clipped, None]),
            "action_is_pad": torch.from_numpy(ids >= self.lengths[episode]),
            "episode_index": torch.tensor(episode, dtype=torch.int64),
        }

    def get_validation_dataset(self):
        result = copy.copy(self); result._file = None; result._validation = not self._validation
        other = set(range(len(self.episode_keys))) - set(self.selected_episodes)
        result.selected_episodes = sorted(other)
        result.sample_indices = [(ep, row) for ep in result.selected_episodes
                                 for row in range(max(0, result.lengths[ep] - result.horizon + 1))]
        return result

    def get_normalizer(self, **kwargs):
        norm = LinearNormalizer()
        norm.fit({"agent_pos": np.concatenate([self.states[i] for i in self.selected_episodes]),
                  "action": np.concatenate([self.actions[i] for i in self.selected_episodes])},
                 last_n_dims=1, mode="limits")
        return norm

    def get_all_actions(self):
        return torch.from_numpy(np.concatenate([self.actions[i] for i in self.selected_episodes]))
