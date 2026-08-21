"""Load PushBox LeRobot v3 dataset into ARP training format."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from pushbox.diffusion_policy.common.pytorch_util import dict_apply
from pushbox.diffusion_policy.common.replay_buffer import ReplayBuffer
from pushbox.diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from pushbox.diffusion_policy.dataset.base_dataset import BaseImageDataset
from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from pushbox.diffusion_policy.common.normalize_util import get_image_range_normalizer

from pushbox.paths import DEFAULT_DEMO_DIR, EXPECTED_NUM_EPISODES

AGENT_STATE_DIM = 9   # 7 joint pos + 2 gripper qpos
ACTION_DIM = 7         # 7D OSC_POSE
BOX_POS_DIM = 2
IMG_SIZE = 96


def _load_lerobot_dataset(demo_dir: str | Path, max_frames_per_ep: int = 500) -> tuple[list[dict], int]:
    """Load all episodes from a LeRobot v3 dataset directory, subsampling to max_frames_per_ep."""
    from pushbox.lerobot_io import load_lerobot_episodes  # noqa: PLC0415

    episodes = load_lerobot_episodes(demo_dir, max_frames_per_ep=max_frames_per_ep)

    # Ensure correct shape for all episodes
    for ep in episodes:
        top45 = ep["top45"]
        sideview = ep["sideview"]

        # If shape is (T, H, W, C) instead of (T, C, H, W), transpose
        if top45.ndim == 4 and top45.shape[-1] in (3, 1) and top45.shape[1] not in (3, 1):
            ep["top45"] = np.moveaxis(top45, -1, 1)
            ep["sideview"] = np.moveaxis(sideview, -1, 1)
        elif top45.ndim == 3:
            # single frame (H, W, C) → (1, C, H, W)
            ep["top45"] = np.moveaxis(top45, -1, 0)[None]
            ep["sideview"] = np.moveaxis(sideview, -1, 0)[None]

        # Handle uint8 → float32 [0,1] conversion if needed
        if ep["top45"].max() > 1.5:
            ep["top45"] = ep["top45"].astype(np.float32) / 255.0
            ep["sideview"] = ep["sideview"].astype(np.float32) / 255.0

    n_loaded = len(episodes)
    total_frames = sum(len(ep["action"]) for ep in episodes)
    print(
        f"[PushBoxImageDataset] loaded {n_loaded} episodes, "
        f"{total_frames} frames from LeRobot dataset at {demo_dir}"
    )
    return episodes, n_loaded


def _find_dataset_root(start: Path) -> Path | None:
    """Search upward from start for meta/info.json."""
    for p in [start] + list(start.parents):
        if (p / "meta" / "info.json").exists():
            return p
    return None


def resolve_demo_dir(demo_dir: str | Path | None = None) -> Path:
    if demo_dir is None:
        return DEFAULT_DEMO_DIR
    p = Path(demo_dir).expanduser()
    if p.is_dir():
        return p
    raise FileNotFoundError(f"Demo directory not found: {p}")


def build_replay_buffer(demo_dir: str | Path, max_frames_per_ep: int = 500) -> ReplayBuffer:
    demo_dir = resolve_demo_dir(demo_dir)
    episodes, n_loaded = _load_lerobot_dataset(demo_dir, max_frames_per_ep=max_frames_per_ep)

    if n_loaded == 0:
        raise RuntimeError(f"No valid episodes found in LeRobot dataset at {demo_dir}")

    rb = ReplayBuffer.create_empty_numpy()
    for ep in episodes:
        rb.add_episode(ep)
    return rb


class PushBoxImageDataset(BaseImageDataset):
    def __init__(
        self,
        demo_dir: str | None = None,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.2,
        max_train_episodes: int | None = None,
        max_frames_per_ep: int = 500,
        image_size: int = IMG_SIZE,
    ):
        super().__init__()
        self.image_size = image_size
        self.demo_dir = resolve_demo_dir(demo_dir)
        self.replay_buffer = build_replay_buffer(self.demo_dir, max_frames_per_ep=max_frames_per_ep)
        n_ep = self.replay_buffer.n_episodes
        if n_ep != EXPECTED_NUM_EPISODES:
            print(
                f"[PushBoxImageDataset] warning: found {n_ep} episodes, "
                f"expected {EXPECTED_NUM_EPISODES} (80 train / 20 val)"
            )

        self.val_mask = get_val_mask(
            n_episodes=n_ep,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~self.val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        print(
            f"[PushBoxImageDataset] split: {int(train_mask.sum())} train / "
            f"{int(self.val_mask.sum())} val (no test) from {n_ep} episodes"
        )

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.train_mask = self.val_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
            "box_pos": self.replay_buffer["box_pos"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["image"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)
        top45 = sample["top45"]  # already (T, C, H, W) from LeRobot
        sideview = sample["sideview"]

        # LeRobot images are already [0,1] float32 (C,H,W)
        # Ensure correct value range
        if top45.max() > 1.5:
            top45 = top45 / 255.0
        if sideview.max() > 1.5:
            sideview = sideview / 255.0

        return {
            "obs": {
                "top45": top45.astype(np.float32),
                "sideview": sideview.astype(np.float32),
                "agent_pos": agent_pos,
                "box_pos": sample["box_pos"].astype(np.float32),
            },
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)
