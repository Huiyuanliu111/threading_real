"""Small synchronized point-cloud dataset for spatial real-robot ARP."""
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


def rasterize_bev(points: np.ndarray, colors: np.ndarray, camera_id: np.ndarray,
                  bounds: np.ndarray, size: int = 64) -> np.ndarray:
    """Rasterize fused points into a translation-equivariant top-down tensor."""
    points, colors = np.asarray(points), np.asarray(colors)
    lower, upper = np.asarray(bounds[:3]), np.asarray(bounds[3:])
    unit = (points[:, :2] - lower[:2]) / (upper[:2] - lower[:2])
    ix = np.floor(unit[:, 0] * size).astype(np.int64)
    iy = np.floor(unit[:, 1] * size).astype(np.int64)
    keep = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
    ix, iy, xyz, rgb, source = ix[keep], iy[keep], points[keep], colors[keep], camera_id[keep]
    flat = iy * size + ix
    cells = size * size
    max_z = np.full(cells, lower[2], dtype=np.float32)
    np.maximum.at(max_z, flat, xyz[:, 2])
    top = xyz[:, 2] >= max_z[flat] - 0.004
    sums = np.zeros((6, cells), dtype=np.float32)
    count = np.zeros(cells, dtype=np.float32)
    np.add.at(count, flat[top], 1)
    for channel in range(3):
        np.add.at(sums[channel], flat[top], rgb[top, channel])
    np.add.at(sums[3], flat, 1)
    np.add.at(sums[4], flat[source == 0], 1)
    np.add.at(sums[5], flat[source == 1], 1)
    output = np.zeros((7, cells), dtype=np.float32)
    output[:3] = sums[:3] / np.maximum(count, 1)[None]
    output[3] = np.clip((max_z - lower[2]) / (upper[2] - lower[2]), 0, 1)
    output[4:] = np.log1p(sums[3:]) / np.log(32.0)
    return output.reshape(7, size, size)


class ThreadingPointCloudSpatialDataset(BaseImageDataset):
    """Read the compact HDF5 produced by ``build_vla_pointcloud_dataset.py``.

    Splitting is episode-wise.  Only the early part of each approach is used by
    default: late frames reveal the answer through the robot TCP itself and let
    a small dataset learn trajectory identity instead of block geometry.
    """

    def __init__(
        self,
        dataset_path: str,
        urdf_path: str,
        horizon: int = 1,
        n_obs_steps: int = 1,
        val_ratio: float = 0.2,
        seed: int = 42,
        split_strategy: str = "spatial_farthest",
        episode_end_fraction: float = 0.65,
        goal_backoff_frames: int = 0,
        max_points: int = 8192,
        bev_size: int = 64,
        bev_bounds: tuple[float, ...] = (0.28, -0.15, -0.15, 0.56, 0.10, 0.20),
        _validation: bool = False,
    ) -> None:
        super().__init__()
        if horizon <= 0 or n_obs_steps != 1:
            raise ValueError("point-cloud spatial dataset requires horizon>0 and n_obs_steps=1")
        self.dataset_path = str(Path(dataset_path).expanduser().resolve())
        self.urdf_path = str(Path(urdf_path).expanduser().resolve())
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.split_strategy = str(split_strategy)
        self.episode_end_fraction = float(episode_end_fraction)
        self.goal_backoff_frames = int(goal_backoff_frames)
        self.max_points = int(max_points)
        self.bev_size = int(bev_size)
        self.bev_bounds = np.asarray(bev_bounds, dtype=np.float32)
        self._validation = bool(_validation)
        self._file: h5py.File | None = None

        fk = PandaForwardKinematics(self.urdf_path)
        with h5py.File(self.dataset_path, "r") as source:
            if source.attrs.get("format") != "threading-base-pointcloud-v1":
                raise ValueError(f"unsupported point-cloud file: {self.dataset_path}")
            self.bounds = np.asarray(source.attrs["bounds_m"], dtype=np.float32)
            self.episode_keys = sorted(source.keys())
            self.lengths = [len(source[key]["points"]) for key in self.episode_keys]
            self.goals = []
            self.tcp_positions = []
            self.actions = []
            self.states = []
            for key in self.episode_keys:
                state = source[key]["observation_state"][:].astype(np.float32)
                future = source[key]["action_state"][:].astype(np.float32)
                tcp = fk.positions(state[:, :7]).astype(np.float32)
                future_tcp = fk.positions(future[:, :7]).astype(np.float32)
                goal_row = max(0, len(future_tcp) - 1 - self.goal_backoff_frames)
                self.goals.append(future_tcp[goal_row])
                self.tcp_positions.append(tcp)
                self.states.append(state)
                delta = np.zeros((len(state), 7), dtype=np.float32)
                delta[:, :3] = future_tcp - tcp
                for row, (q0, q1) in enumerate(zip(state[:, :7], future[:, :7])):
                    r0 = fk.pose(q0)[:3, :3]
                    r1 = fk.pose(q1)[:3, :3]
                    delta[row, 3:6] = Rotation.from_matrix(r1 @ r0.T).as_rotvec()
                delta[:, 6] = future[:, 7] - state[:, 7]
                self.actions.append(delta)

        episode_count = len(self.episode_keys)
        val_count = max(1, int(round(episode_count * self.val_ratio)))
        if self.split_strategy == "random":
            rng = np.random.default_rng(self.seed)
            val_set = set(rng.permutation(episode_count)[:val_count].tolist())
        elif self.split_strategy == "spatial_farthest":
            # X/Y carry the block placement variation; Z is nearly constant
            # and should not dominate which episodes are held out.
            goals = np.asarray(self.goals)[:, :2]
            scale = np.ptp(goals, axis=0).clip(1e-4)
            normalized = (goals - goals.mean(axis=0)) / scale
            chosen = [int(np.linalg.norm(normalized, axis=1).argmin())]
            while len(chosen) < val_count:
                distance = np.stack([
                    np.linalg.norm(normalized - normalized[index], axis=1)
                    for index in chosen
                ]).min(axis=0)
                distance[chosen] = -1
                chosen.append(int(distance.argmax()))
            val_set = set(chosen)
        else:
            raise ValueError(f"unknown split_strategy {self.split_strategy!r}")
        selected = val_set if self._validation else set(range(episode_count)) - val_set
        self.selected_episodes = sorted(selected)
        self.sample_indices: list[tuple[int, int]] = []
        for episode in self.selected_episodes:
            usable = max(1, min(self.lengths[episode], int(np.ceil(
                self.lengths[episode] * self.episode_end_fraction
            ))))
            self.sample_indices.extend((episode, row) for row in range(usable))
        if not self.sample_indices:
            raise ValueError("dataset split contains no samples")

    def _h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.dataset_path, "r", swmr=True)
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        file = getattr(self, "_file", None)
        if file is not None:
            try:
                if file.id.valid:
                    file.close()
            except (AttributeError, TypeError, ValueError):
                # Interpreter shutdown may already have torn down h5py internals.
                pass

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, row = self.sample_indices[index]
        group = self._h5()[self.episode_keys[episode]]
        count = min(self.max_points, group["points"].shape[1])
        points = group["points"][row, :count].astype(np.float32)
        colors = group["colors"][row, :count].astype(np.float32) / 255.0
        camera_id = group["camera_id"][row, :count].astype(np.int64)
        bev = rasterize_bev(points, colors, camera_id, self.bev_bounds, self.bev_size)
        return {
            "obs": {
                "points": torch.from_numpy(points[None]),
                "colors": torch.from_numpy(colors[None]),
                "camera_id": torch.from_numpy(camera_id[None]),
                "bev": torch.from_numpy(bev[None]),
                "agent_pos": torch.from_numpy(self.states[episode][row : row + 1]),
                "tcp_pos": torch.from_numpy(self.tcp_positions[episode][row : row + 1]),
            },
            "action": torch.from_numpy(self.actions[episode][np.minimum(
                np.arange(row, row + self.horizon), self.lengths[episode] - 1
            )]),
            "action_is_pad": torch.from_numpy(
                (np.arange(row, row + self.horizon) >= self.lengths[episode])
            ),
            "spatial_goal_xyz": torch.from_numpy(self.goals[episode].copy()),
            "episode_index": torch.tensor(episode, dtype=torch.int64),
        }

    def get_validation_dataset(self):
        result = copy.copy(self)
        result._file = None
        result._validation = True
        val_episodes = set(range(len(self.episode_keys))) - set(self.selected_episodes)
        result.selected_episodes = sorted(val_episodes)
        result.sample_indices = []
        for episode in result.selected_episodes:
            usable = max(1, min(result.lengths[episode], int(np.ceil(
                result.lengths[episode] * result.episode_end_fraction
            ))))
            result.sample_indices.extend((episode, row) for row in range(usable))
        return result

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer.fit({
            "agent_pos": np.concatenate([self.states[i] for i in self.selected_episodes]),
            "action": np.concatenate([self.actions[i] for i in self.selected_episodes]),
        }, last_n_dims=1, mode="limits")
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([self.actions[i] for i in self.selected_episodes]))
