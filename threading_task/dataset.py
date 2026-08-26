"""Stream Threading datasets for ARP training.

This module contains two dataset frontends:

* ``ThreadingImageDataset`` for simulator/MimicGen HDF5 files.
* ``ThreadingRealLeRobotDataset`` for real-robot LeRobot-style datasets
  produced from Record_layer recordings.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as torch_functional

from pushbox.diffusion_policy.common.normalize_util import get_image_range_normalizer
from pushbox.diffusion_policy.common.pytorch_util import dict_apply
from pushbox.diffusion_policy.common.sampler import downsample_mask, get_val_mask
from pushbox.diffusion_policy.dataset.base_dataset import BaseImageDataset
from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer

AGENT_STATE_DIM = 9
EEF_AGENT_STATE_DIM = 8
ACTION_DIM = 7
DEFAULT_CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
DEFAULT_REAL_CAMERA_KEYS = (
    "observation.images.exterior_image_2_right",
    "observation.images.wrist_image_left",
)
DEFAULT_REAL_CAMERA_OUTPUT_KEYS = ("sideview", "wrist")


def sorted_demo_keys(data_group: h5py.Group) -> list[str]:
    """Return demo keys in numeric order while tolerating nonstandard names."""

    def order(key: str) -> tuple[int, str]:
        try:
            return int(key.rsplit("_", 1)[-1]), key
        except ValueError:
            return 10**12, key

    return sorted(data_group.keys(), key=order)


def read_env_metadata(dataset_path: str | Path) -> dict[str, Any]:
    with h5py.File(Path(dataset_path).expanduser(), "r") as f:
        raw = f["data"].attrs.get("env_args", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _as_chw_float(images: np.ndarray, key: str, image_size: int) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"{key}: expected 4D image array, got {images.shape}")
    if images.shape[-1] in (1, 3, 4):
        images = images[..., :3]
    elif images.shape[1] in (1, 3):
        images = np.moveaxis(images, 1, -1)
    else:
        raise ValueError(f"{key}: cannot infer channel axis from {images.shape}")
    if images.shape[1:3] != (image_size, image_size):
        interpolation = cv2.INTER_AREA if max(images.shape[1:3]) > image_size else cv2.INTER_LINEAR
        images = np.stack(
            [cv2.resize(frame, (image_size, image_size), interpolation=interpolation) for frame in images]
        )
    images = np.moveaxis(images, -1, 1).astype(np.float32, copy=False)
    if images.size and images.max() > 1.5:
        images /= 255.0
    return np.clip(images, 0.0, 1.0)


def _read_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """Read possibly repeated, ordered rows without h5py fancy-index restrictions."""
    start = int(indices.min())
    stop = int(indices.max()) + 1
    block = np.asarray(dataset[start:stop])
    return block[indices - start]


def _resize_video_frames(frames: np.ndarray, image_size: int) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError(f"expected video frames with shape (T,H,W,C), got {frames.shape}")
    if frames.shape[-1] in (1, 3, 4):
        frames = frames[..., :3]
    else:
        raise ValueError(f"cannot infer channel axis from video shape {frames.shape}")
    if frames.shape[1:3] != (image_size, image_size):
        interpolation = cv2.INTER_AREA if max(frames.shape[1:3]) > image_size else cv2.INTER_LINEAR
        frames = np.stack(
            [cv2.resize(frame, (image_size, image_size), interpolation=interpolation) for frame in frames]
        )
    return frames.astype(np.float32, copy=False) / 255.0


def _fixed_list_column_to_numpy(column: Any) -> np.ndarray:
    return np.asarray(column.to_pylist(), dtype=np.float32)


def _as_chw_image_tensor(image: Any, image_size: int, key: str) -> torch.Tensor:
    """Normalize an official LeRobot decoded frame to CHW float32 [0, 1]."""
    value = image.detach().cpu() if isinstance(image, torch.Tensor) else torch.as_tensor(image)
    if value.ndim != 3:
        raise ValueError(f"{key}: expected a 3D decoded image, got {tuple(value.shape)}")
    if value.shape[0] in (1, 3, 4):
        value = value[:3]
    elif value.shape[-1] in (1, 3, 4):
        value = value[..., :3].permute(2, 0, 1)
    else:
        raise ValueError(f"{key}: cannot infer channel axis from {tuple(value.shape)}")
    value = value.to(torch.float32)
    if value.numel() and float(value.max()) > 1.5:
        value = value / 255.0
    if tuple(value.shape[-2:]) != (image_size, image_size):
        value = torch_functional.interpolate(
            value.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return value.clamp_(0.0, 1.0)


def _build_sample_indices(
    lengths: np.ndarray,
    episode_mask: np.ndarray,
    horizon: int,
    pad_before: int,
    pad_after: int,
) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    pad_before = min(max(int(pad_before), 0), horizon - 1)
    pad_after = min(max(int(pad_after), 0), horizon - 1)
    for episode_index, length in enumerate(lengths):
        if not episode_mask[episode_index]:
            continue
        min_start = -pad_before
        max_start = int(length) - horizon + pad_after
        rows.extend((episode_index, start) for start in range(min_start, max_start + 1))
    if not rows:
        selected = lengths[episode_mask]
        raise ValueError(
            "No Threading sequences can be sampled: "
            f"horizon={horizon}, pad_before={pad_before}, pad_after={pad_after}, "
            f"selected episode lengths={selected.tolist()}"
        )
    return np.asarray(rows, dtype=np.int64)


class ThreadingImageDataset(BaseImageDataset):
    """HDF5-backed ARP dataset with bounded memory use.

    The official core dataset already contains compressed 84x84 RGB frames. An
    optional camera-only HDF5 can provide additional views without duplicating
    actions and states. Only the requested observation context is loaded; actions
    and robot states retain the full training horizon.
    """

    def __init__(
        self,
        dataset_path: str,
        horizon: int = 50,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.2,
        max_train_episodes: int | None = None,
        max_frames_per_ep: int | None = None,
        joint_key: str = "robot0_joint_pos",
        gripper_key: str = "robot0_gripper_qpos",
        eef_pos_key: str = "robot0_eef_pos",
        eef_quat_key: str = "robot0_eef_quat",
        state_mode: str = "joint",
        camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS,
        camera_output_keys: tuple[str, ...] | None = None,
        image_size: int = 96,
        image_sizes: tuple[int, ...] | None = None,
        auxiliary_dataset_path: str | None = None,
        n_obs_steps: int = 2,
        max_validation_sequences: int | None = None,
    ):
        super().__init__()
        path = Path(dataset_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Threading dataset not found: {path}")
        self.dataset_path = str(path)
        self.joint_key = joint_key
        self.gripper_key = gripper_key
        self.eef_pos_key = eef_pos_key
        self.eef_quat_key = eef_quat_key
        if state_mode not in {"joint", "eef"}:
            raise ValueError(f"state_mode must be 'joint' or 'eef', got {state_mode!r}")
        self.state_mode = state_mode
        self.agent_state_dim = AGENT_STATE_DIM if state_mode == "joint" else EEF_AGENT_STATE_DIM
        self.camera_keys = tuple(camera_keys)
        if camera_output_keys is None:
            if len(self.camera_keys) != 2:
                raise ValueError("camera_output_keys is required when using other than two cameras")
            camera_output_keys = ("top45", "sideview")
        self.camera_output_keys = tuple(camera_output_keys)
        if len(self.camera_output_keys) != len(self.camera_keys):
            raise ValueError("camera_keys and camera_output_keys must have equal length")
        if len(set(self.camera_output_keys)) != len(self.camera_output_keys):
            raise ValueError(f"camera_output_keys must be unique: {self.camera_output_keys}")
        self.image_size = int(image_size)
        if image_sizes is None:
            self.image_sizes = (self.image_size,) * len(self.camera_keys)
        else:
            self.image_sizes = tuple(int(size) for size in image_sizes)
            if len(self.image_sizes) != len(self.camera_keys):
                raise ValueError("image_sizes must contain one value per camera")
        auxiliary_path = (
            Path(auxiliary_dataset_path).expanduser().resolve()
            if auxiliary_dataset_path
            else None
        )
        if auxiliary_path is not None and not auxiliary_path.is_file():
            raise FileNotFoundError(f"Threading auxiliary dataset not found: {auxiliary_path}")
        self.auxiliary_dataset_path = str(auxiliary_path) if auxiliary_path else None
        self.n_obs_steps = int(n_obs_steps)
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.seed = int(seed)
        self.max_validation_sequences = (
            None
            if max_validation_sequences is None
            else int(max_validation_sequences)
        )
        if (
            self.max_validation_sequences is not None
            and self.max_validation_sequences <= 0
        ):
            raise ValueError("max_validation_sequences must be positive")
        self._h5: h5py.File | None = None
        self._aux_h5: h5py.File | None = None
        self._h5_pid: int | None = None

        state_keys = (
            [joint_key, gripper_key]
            if state_mode == "joint"
            else [eef_pos_key, eef_quat_key, gripper_key]
        )
        required = list(state_keys)
        with h5py.File(path, "r") as f, (
            h5py.File(auxiliary_path, "r") if auxiliary_path else _NullH5()
        ) as auxiliary:
            if "data" not in f:
                raise KeyError(f"{path}: missing /data group")
            self.demo_keys = sorted_demo_keys(f["data"])
            if not self.demo_keys:
                raise RuntimeError(f"No demonstrations found in {path}")
            auxiliary_data = auxiliary["data"] if auxiliary_path else None
            if auxiliary_data is not None and sorted_demo_keys(auxiliary_data) != self.demo_keys:
                raise ValueError("Primary and auxiliary Threading demo keys do not match")
            first_obs = f["data"][self.demo_keys[0]]["obs"]
            first_aux_obs = (
                auxiliary_data[self.demo_keys[0]]["obs"] if auxiliary_data is not None else None
            )
            self.camera_sources: dict[str, str] = {}
            for output_key, dataset_key in zip(self.camera_output_keys, self.camera_keys):
                if dataset_key in first_obs:
                    self.camera_sources[output_key] = "primary"
                elif first_aux_obs is not None and dataset_key in first_aux_obs:
                    self.camera_sources[output_key] = "auxiliary"
                else:
                    raise KeyError(
                        f"Camera {dataset_key!r} was not found in the primary or auxiliary dataset"
                    )
            lengths = []
            for key in self.demo_keys:
                demo = f["data"][key]
                if "actions" not in demo or "obs" not in demo:
                    raise KeyError(f"{demo.name}: missing actions or obs")
                missing = [name for name in required if name not in demo["obs"]]
                if missing:
                    raise KeyError(f"{demo.name}: missing observation keys {missing}")
                length = len(demo["actions"])
                if any(len(demo["obs"][name]) != length for name in required):
                    raise ValueError(f"{demo.name}: inconsistent trajectory lengths")
                for output_key, dataset_key in zip(self.camera_output_keys, self.camera_keys):
                    if (
                        self.camera_sources[output_key] == "auxiliary"
                        and not bool(auxiliary_data[key].attrs.get("complete", False))
                    ):
                        raise ValueError(f"{key}: auxiliary camera rendering is incomplete")
                    source_obs = (
                        demo["obs"]
                        if self.camera_sources[output_key] == "primary"
                        else auxiliary_data[key]["obs"]
                    )
                    if dataset_key not in source_obs or len(source_obs[dataset_key]) != length:
                        raise ValueError(
                            f"{key}: camera {dataset_key!r} is missing or has the wrong length"
                        )
                lengths.append(min(length, max_frames_per_ep) if max_frames_per_ep else length)
        self.episode_lengths = np.asarray(lengths, dtype=np.int64)

        n_ep = len(self.demo_keys)
        self.val_mask = get_val_mask(n_episodes=n_ep, val_ratio=val_ratio, seed=seed)
        self.train_mask = downsample_mask(~self.val_mask, max_n=max_train_episodes, seed=seed)
        self.sample_indices = _build_sample_indices(
            self.episode_lengths,
            self.train_mask,
            self.horizon,
            self.pad_before,
            self.pad_after,
        )
        print(
            f"[ThreadingImageDataset] streaming {n_ep} episodes: "
            f"{int(self.train_mask.sum())} train / {int(self.val_mask.sum())} val, "
            f"{len(self.sample_indices)} train sequences"
        )

    def _get_h5(self) -> h5py.File:
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                self._h5.close()
            if self._aux_h5 is not None:
                self._aux_h5.close()
            self._h5 = h5py.File(self.dataset_path, "r")
            self._aux_h5 = (
                h5py.File(self.auxiliary_dataset_path, "r")
                if self.auxiliary_dataset_path
                else None
            )
            self._h5_pid = pid
        return self._h5

    def _get_camera_obs(self, demo_key: str, output_key: str) -> h5py.Group:
        primary = self._get_h5()
        if self.camera_sources[output_key] == "primary":
            return primary["data"][demo_key]["obs"]
        if self._aux_h5 is None:
            raise RuntimeError(f"Auxiliary camera source is unavailable for {output_key}")
        return self._aux_h5["data"][demo_key]["obs"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_aux_h5"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self):
        for handle in (getattr(self, "_h5", None), getattr(self, "_aux_h5", None)):
            if handle is None:
                continue
            try:
                handle.close()
            except Exception:
                pass

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set._h5 = None
        val_set._aux_h5 = None
        val_set._h5_pid = None
        val_indices = _build_sample_indices(
            self.episode_lengths,
            self.val_mask,
            self.horizon,
            self.pad_before,
            self.pad_after,
        )
        if (
            self.max_validation_sequences is not None
            and len(val_indices) > self.max_validation_sequences
        ):
            rng = np.random.default_rng(self.seed + 1)
            selected = np.sort(
                rng.choice(
                    len(val_indices),
                    size=self.max_validation_sequences,
                    replace=False,
                )
            )
            val_indices = val_indices[selected]
        val_set.sample_indices = val_indices
        val_set.train_mask = self.val_mask.copy()
        return val_set

    def _normalizer_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        actions: list[np.ndarray] = []
        states: list[np.ndarray] = []
        with h5py.File(self.dataset_path, "r") as f:
            for episode_index in np.flatnonzero(self.train_mask):
                demo = f["data"][self.demo_keys[int(episode_index)]]
                length = int(self.episode_lengths[episode_index])
                actions.append(np.asarray(demo["actions"][:length], dtype=np.float32))
                rows = np.arange(length, dtype=np.int64)
                states.append(self._read_state(demo["obs"], rows))
        return np.concatenate(actions), np.concatenate(states)

    def _read_state(self, obs: h5py.Group, rows: np.ndarray) -> np.ndarray:
        gripper = _read_rows(obs[self.gripper_key], rows).astype(np.float32)
        if self.state_mode == "joint":
            joints = _read_rows(obs[self.joint_key], rows).astype(np.float32)
            state = np.concatenate([joints, gripper], axis=-1)
        else:
            eef_pos = _read_rows(obs[self.eef_pos_key], rows).astype(np.float32)
            eef_quat = _read_rows(obs[self.eef_quat_key], rows).astype(np.float32)
            state = np.concatenate([eef_pos, eef_quat, gripper[..., :1]], axis=-1)
        if state.shape[-1] != self.agent_state_dim:
            raise ValueError(
                f"Expected {self.agent_state_dim}D {self.state_mode} state, got {state.shape}"
            )
        return state

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        actions, states = self._normalizer_arrays()
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": actions, "agent_pos": states},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        actions, _ = self._normalizer_arrays()
        return torch.from_numpy(actions)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_index, sequence_start = self.sample_indices[idx]
        length = int(self.episode_lengths[episode_index])
        demo = self._get_h5()["data"][self.demo_keys[int(episode_index)]]

        raw_sequence_rows = int(sequence_start) + np.arange(self.horizon)
        action_is_pad = (raw_sequence_rows < 0) | (raw_sequence_rows >= length)
        sequence_rows = np.clip(raw_sequence_rows, 0, length - 1).astype(np.int64)
        obs_rows = sequence_rows[: min(self.n_obs_steps, self.horizon)]
        actions = _read_rows(demo["actions"], sequence_rows).astype(np.float32)
        state = self._read_state(demo["obs"], sequence_rows)
        camera_data = {}
        for output_key, dataset_key, image_size in zip(
            self.camera_output_keys,
            self.camera_keys,
            self.image_sizes,
        ):
            camera_obs = self._get_camera_obs(self.demo_keys[int(episode_index)], output_key)
            frames = _read_rows(camera_obs[dataset_key], obs_rows)
            camera_data[output_key] = _as_chw_float(frames, dataset_key, image_size)
        data = {
            "obs": {
                **camera_data,
                "agent_pos": state,
            },
            "action": actions,
            "action_is_pad": action_is_pad,
        }
        return dict_apply(data, torch.from_numpy)


class _NullH5:
    """Context manager used to keep optional HDF5 opening readable."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class ThreadingRealLeRobotDataset(BaseImageDataset):
    """LeRobot-backed dataset for real Franka threading recordings.

    Official LeRobot v3 data is reconstructed from its consolidated parquet
    shards and decoded through ``LeRobotDataset``. The previous episode-file
    layout remains readable for backward compatibility. The real robot action
    is next-frame joint position plus gripper width by default.
    """

    def __init__(
        self,
        dataset_path: str,
        horizon: int = 20,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.2,
        max_train_episodes: int | None = None,
        max_frames_per_ep: int | None = None,
        camera_keys: tuple[str, ...] = DEFAULT_REAL_CAMERA_KEYS,
        camera_output_keys: tuple[str, ...] = DEFAULT_REAL_CAMERA_OUTPUT_KEYS,
        image_size: int = 96,
        image_sizes: tuple[int, ...] | None = None,
        n_obs_steps: int = 2,
        max_validation_sequences: int | None = None,
        state_key: str = "observation.state",
        action_key: str = "action",
        max_cached_video_episodes: int = 2,
    ):
        super().__init__()
        path = Path(dataset_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Real LeRobot dataset not found: {path}")
        if not (path / "meta" / "info.json").exists():
            raise FileNotFoundError(f"{path}: missing meta/info.json")
        self.dataset_path = str(path)
        self.camera_keys = tuple(camera_keys)
        self.camera_output_keys = tuple(camera_output_keys)
        if len(self.camera_keys) != len(self.camera_output_keys):
            raise ValueError("camera_keys and camera_output_keys must have equal length")
        if len(set(self.camera_output_keys)) != len(self.camera_output_keys):
            raise ValueError(f"camera_output_keys must be unique: {self.camera_output_keys}")
        self.image_size = int(image_size)
        self.image_sizes = (
            tuple(int(size) for size in image_sizes)
            if image_sizes is not None
            else (self.image_size,) * len(self.camera_keys)
        )
        if len(self.image_sizes) != len(self.camera_keys):
            raise ValueError("image_sizes must contain one value per camera")
        self.n_obs_steps = int(n_obs_steps)
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.seed = int(seed)
        self.state_key = state_key
        self.action_key = action_key
        self.max_cached_video_episodes = max(int(max_cached_video_episodes), 0)
        self.max_validation_sequences = (
            None
            if max_validation_sequences is None
            else int(max_validation_sequences)
        )
        if (
            self.max_validation_sequences is not None
            and self.max_validation_sequences <= 0
        ):
            raise ValueError("max_validation_sequences must be positive")

        self.info = json.loads((path / "meta" / "info.json").read_text())
        self.is_lerobot_v3 = str(self.info.get("codebase_version", "")).startswith("v3")
        self._lerobot_dataset: Any | None = None
        features = self.info.get("features", {})
        missing_features = [
            key for key in (self.state_key, self.action_key, *self.camera_keys)
            if key not in features
        ]
        if missing_features:
            raise KeyError(f"{path}: missing LeRobot features {missing_features}")

        self.episode_paths = sorted((path / "data").rglob("episode_*.parquet"))
        if not self.episode_paths:
            self.episode_paths = sorted((path / "data").rglob("*.parquet"))
        if not self.episode_paths:
            raise FileNotFoundError(f"{path}: no parquet files under data/")

        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.dataset_row_indices: list[np.ndarray] = []
        lengths = []
        if self.is_lerobot_v3:
            self._read_v3_parquet_episodes(max_frames_per_ep)
        else:
            self._read_parquet_episodes(max_frames_per_ep)
        for state, action in zip(self.states, self.actions):
            if len(state) != len(action):
                raise ValueError("state/action length mismatch in real LeRobot dataset")
            lengths.append(len(action))
        self.episode_lengths = np.asarray(lengths, dtype=np.int64)
        self.agent_state_dim = int(self.states[0].shape[-1])
        self.action_dim = int(self.actions[0].shape[-1])

        self._video_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        n_ep = len(self.states)
        self.val_mask = get_val_mask(n_episodes=n_ep, val_ratio=val_ratio, seed=seed)
        self.train_mask = downsample_mask(~self.val_mask, max_n=max_train_episodes, seed=seed)
        self.sample_indices = _build_sample_indices(
            self.episode_lengths,
            self.train_mask,
            self.horizon,
            self.pad_before,
            self.pad_after,
        )
        print(
            f"[ThreadingRealLeRobotDataset] streaming {n_ep} real episodes: "
            f"{int(self.train_mask.sum())} train / {int(self.val_mask.sum())} val, "
            f"{len(self.sample_indices)} train sequences"
        )

    def _read_parquet_episodes(self, max_frames_per_ep: int | None) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "ThreadingRealLeRobotDataset requires pyarrow to read LeRobot parquet files"
            ) from exc
        for parquet_path in self.episode_paths:
            table = pq.read_table(str(parquet_path), columns=[self.state_key, self.action_key])
            state = _fixed_list_column_to_numpy(table.column(self.state_key))
            action = _fixed_list_column_to_numpy(table.column(self.action_key))
            if max_frames_per_ep is not None:
                state = state[:max_frames_per_ep]
                action = action[:max_frames_per_ep]
            if state.size == 0 or action.size == 0:
                raise ValueError(f"{parquet_path}: empty state/action episode")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"{parquet_path}: state/action contains NaN or Inf")
            self.states.append(state.astype(np.float32, copy=False))
            self.actions.append(action.astype(np.float32, copy=False))

    def _read_v3_parquet_episodes(self, max_frames_per_ep: int | None) -> None:
        """Read v3 tabular shards and reconstruct episodes from their index columns."""
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "ThreadingRealLeRobotDataset requires pyarrow to read LeRobot parquet files"
            ) from exc

        state_parts = []
        action_parts = []
        episode_parts = []
        frame_parts = []
        index_parts = []
        required_columns = [
            self.state_key,
            self.action_key,
            "episode_index",
            "frame_index",
            "index",
        ]
        for parquet_path in self.episode_paths:
            table = pq.read_table(str(parquet_path), columns=required_columns)
            state_parts.append(_fixed_list_column_to_numpy(table.column(self.state_key)))
            action_parts.append(_fixed_list_column_to_numpy(table.column(self.action_key)))
            episode_parts.append(
                np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
            )
            frame_parts.append(np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64))
            index_parts.append(np.asarray(table.column("index").to_pylist(), dtype=np.int64))

        states = np.concatenate(state_parts)
        actions = np.concatenate(action_parts)
        episode_indices = np.concatenate(episode_parts)
        frame_indices = np.concatenate(frame_parts)
        dataset_indices = np.concatenate(index_parts)
        order = np.argsort(dataset_indices, kind="stable")
        states = states[order]
        actions = actions[order]
        episode_indices = episode_indices[order]
        frame_indices = frame_indices[order]
        dataset_indices = dataset_indices[order]
        if not np.array_equal(dataset_indices, np.arange(len(dataset_indices))):
            raise ValueError("LeRobot v3 global index column must be contiguous and start at zero")

        for episode_index in np.unique(episode_indices):
            rows = np.flatnonzero(episode_indices == episode_index)
            rows = rows[np.argsort(frame_indices[rows], kind="stable")]
            if not np.array_equal(frame_indices[rows], np.arange(len(rows))):
                raise ValueError(
                    f"episode {episode_index}: frame_index is not contiguous from zero"
                )
            if max_frames_per_ep is not None:
                rows = rows[:max_frames_per_ep]
            state = states[rows].astype(np.float32, copy=False)
            action = actions[rows].astype(np.float32, copy=False)
            if state.size == 0 or action.size == 0:
                raise ValueError(f"episode {episode_index}: empty state/action episode")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"episode {episode_index}: state/action contains NaN or Inf")
            self.states.append(state)
            self.actions.append(action)
            self.dataset_row_indices.append(dataset_indices[rows])

    def _get_lerobot_dataset(self):
        if self._lerobot_dataset is None:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except ImportError as exc:
                raise ImportError(
                    "Official LeRobot v3 image decoding requires lerobot>=0.4.0"
                ) from exc
            self._lerobot_dataset = LeRobotDataset(
                repo_id="local/threading_real",
                root=self.dataset_path,
                download_videos=False,
            )
        return self._lerobot_dataset

    def _video_path(self, episode_index: int, camera_key: str) -> Path:
        episode_name = self.episode_paths[episode_index].stem
        video_path_template = self.info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        chunk = episode_index // int(self.info.get("chunks_size", 1000))
        formatted = video_path_template.format(
            episode_chunk=chunk,
            video_key=camera_key,
            episode_index=episode_index,
        )
        path = Path(self.dataset_path) / formatted
        if path.exists():
            return path
        matches = sorted((Path(self.dataset_path) / "videos").rglob(f"{camera_key}/{episode_name}.mp4"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"missing video for episode {episode_index}, camera {camera_key!r}")

    def _load_video_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index in self._video_cache:
            self._video_cache.move_to_end(episode_index)
            return self._video_cache[episode_index]

        episode_videos: dict[str, np.ndarray] = {}
        expected_length = int(self.episode_lengths[episode_index])
        for output_key, camera_key, image_size in zip(
            self.camera_output_keys,
            self.camera_keys,
            self.image_sizes,
        ):
            path = self._video_path(episode_index, camera_key)
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError(f"could not open video: {path}")
            frames = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            finally:
                capture.release()
            if len(frames) < expected_length:
                raise ValueError(
                    f"{path}: only {len(frames)} frames for episode length {expected_length}"
                )
            episode_videos[output_key] = _resize_video_frames(
                np.asarray(frames[:expected_length]),
                image_size,
            )

        if self.max_cached_video_episodes > 0:
            self._video_cache[episode_index] = episode_videos
            self._video_cache.move_to_end(episode_index)
            while len(self._video_cache) > self.max_cached_video_episodes:
                self._video_cache.popitem(last=False)
        return episode_videos

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_video_cache"] = OrderedDict()
        state["_lerobot_dataset"] = None
        return state

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set._video_cache = OrderedDict()
        val_set._lerobot_dataset = None
        val_indices = _build_sample_indices(
            self.episode_lengths,
            self.val_mask,
            self.horizon,
            self.pad_before,
            self.pad_after,
        )
        if (
            self.max_validation_sequences is not None
            and len(val_indices) > self.max_validation_sequences
        ):
            rng = np.random.default_rng(self.seed + 1)
            selected = np.sort(
                rng.choice(
                    len(val_indices),
                    size=self.max_validation_sequences,
                    replace=False,
                )
            )
            val_indices = val_indices[selected]
        val_set.sample_indices = val_indices
        val_set.train_mask = self.val_mask.copy()
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        actions = np.concatenate(
            [self.actions[index] for index in np.flatnonzero(self.train_mask)],
            axis=0,
        )
        states = np.concatenate(
            [self.states[index] for index in np.flatnonzero(self.train_mask)],
            axis=0,
        )
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": actions, "agent_pos": states},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.concatenate(self.actions, axis=0))

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_index, sequence_start = self.sample_indices[idx]
        episode_index = int(episode_index)
        length = int(self.episode_lengths[episode_index])
        raw_sequence_rows = int(sequence_start) + np.arange(self.horizon)
        action_is_pad = (raw_sequence_rows < 0) | (raw_sequence_rows >= length)
        sequence_rows = np.clip(raw_sequence_rows, 0, length - 1).astype(np.int64)
        obs_rows = sequence_rows[: min(self.n_obs_steps, self.horizon)]

        if self.is_lerobot_v3:
            dataset = self._get_lerobot_dataset()
            decoded_frames = [
                dataset[int(self.dataset_row_indices[episode_index][row])]
                for row in obs_rows
            ]
            images = {
                output_key: torch.stack(
                    [
                        _as_chw_image_tensor(frame[camera_key], image_size, camera_key)
                        for frame in decoded_frames
                    ]
                )
                for output_key, camera_key, image_size in zip(
                    self.camera_output_keys,
                    self.camera_keys,
                    self.image_sizes,
                    strict=True,
                )
            }
            return {
                "obs": {
                    **images,
                    "agent_pos": torch.from_numpy(
                        self.states[episode_index][sequence_rows].astype(
                            np.float32, copy=False
                        )
                    ),
                },
                "action": torch.from_numpy(
                    self.actions[episode_index][sequence_rows].astype(
                        np.float32, copy=False
                    )
                ),
                "action_is_pad": torch.from_numpy(action_is_pad),
            }

        videos = self._load_video_episode(episode_index)
        data = {
            "obs": {
                **{
                    key: np.moveaxis(videos[key][obs_rows], -1, 1).astype(np.float32, copy=False)
                    for key in self.camera_output_keys
                },
                "agent_pos": self.states[episode_index][sequence_rows].astype(np.float32, copy=False),
            },
            "action": self.actions[episode_index][sequence_rows].astype(np.float32, copy=False),
            "action_is_pad": action_is_pad,
        }
        return dict_apply(data, torch.from_numpy)
