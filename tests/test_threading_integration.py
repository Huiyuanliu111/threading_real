from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from hydra.utils import get_class
from omegaconf import OmegaConf

from envs.threading_env import needle_center_reaches_ring
from threading_task.dataset import ThreadingImageDataset
from threading_task.env import (
    _environment_kwargs,
    agent_state_from_obs,
    image_from_obs,
)
from threading_task.metrics import EpisodeMetrics, aggregate_episodes
from pushbox.workspace import aligned_action_target, masked_action_mse


def _write_demo(path):
    rng = np.random.default_rng(4)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {"env_name": "Threading_D0", "type": 1, "env_kwargs": {}}
        )
        for index in range(3):
            n = 24 + index
            demo = data.create_group(f"demo_{index}")
            demo.create_dataset("actions", data=rng.uniform(-1, 1, (n, 7)).astype("f4"))
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_joint_pos", data=rng.normal(size=(n, 7)).astype("f4"))
            obs.create_dataset("robot0_gripper_qpos", data=rng.normal(size=(n, 2)).astype("f4"))
            obs.create_dataset("robot0_eef_pos", data=rng.normal(size=(n, 3)).astype("f4"))
            obs.create_dataset("robot0_eef_quat", data=rng.normal(size=(n, 4)).astype("f4"))
            obs.create_dataset("agentview_image", data=rng.integers(0, 256, (n, 84, 84, 3), dtype="u1"))
            obs.create_dataset("robot0_eye_in_hand_image", data=rng.integers(0, 256, (n, 84, 84, 3), dtype="u1"))


def _write_auxiliary_camera(path):
    rng = np.random.default_rng(5)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for index in range(3):
            n = 24 + index
            demo = data.create_group(f"demo_{index}")
            demo.attrs["complete"] = True
            obs = demo.create_group("obs")
            obs.create_dataset(
                "threading_closeup_image",
                data=rng.integers(0, 256, (n, 128, 128, 3), dtype="u1"),
            )


def test_threading_dataset_schema(tmp_path):
    path = tmp_path / "threading.hdf5"
    _write_demo(path)
    dataset = ThreadingImageDataset(str(path), horizon=10, val_ratio=1 / 3)
    sample = dataset[0]
    assert sample["obs"]["agent_pos"].shape == (10, 9)
    assert sample["obs"]["top45"].shape == (2, 3, 96, 96)
    assert sample["action"].shape == (10, 7)
    assert sample["action_is_pad"].shape == (10,)
    assert sample["action_is_pad"][0]
    assert sample["obs"]["top45"].max() <= 1
    normalizer = dataset.get_normalizer()
    assert "agent_pos" in normalizer.params_dict


def test_threading_eef_dataset_schema(tmp_path):
    path = tmp_path / "threading.hdf5"
    _write_demo(path)
    dataset = ThreadingImageDataset(
        str(path),
        horizon=10,
        val_ratio=1 / 3,
        state_mode="eef",
        image_size=84,
    )
    sample = dataset[0]
    assert sample["obs"]["agent_pos"].shape == (10, 8)
    assert sample["obs"]["top45"].shape == (2, 3, 84, 84)
    assert dataset.get_normalizer()["agent_pos"].params_dict["scale"].shape == (8,)


def test_threading_three_camera_dataset_with_auxiliary_hdf5(tmp_path):
    path = tmp_path / "threading.hdf5"
    auxiliary_path = tmp_path / "threading_closeup.hdf5"
    _write_demo(path)
    _write_auxiliary_camera(auxiliary_path)
    dataset = ThreadingImageDataset(
        str(path),
        auxiliary_dataset_path=str(auxiliary_path),
        horizon=10,
        val_ratio=1 / 3,
        state_mode="eef",
        camera_keys=(
            "agentview_image",
            "robot0_eye_in_hand_image",
            "threading_closeup_image",
        ),
        camera_output_keys=("top45", "wrist", "sideview"),
        image_sizes=(84, 84, 128),
    )
    sample = dataset[0]
    assert sample["obs"]["top45"].shape == (2, 3, 84, 84)
    assert sample["obs"]["wrist"].shape == (2, 3, 84, 84)
    assert sample["obs"]["sideview"].shape == (2, 3, 128, 128)


def test_threading_validation_subset_is_fixed(tmp_path):
    path = tmp_path / "threading.hdf5"
    _write_demo(path)
    dataset = ThreadingImageDataset(
        str(path),
        horizon=10,
        val_ratio=1 / 3,
        max_validation_sequences=5,
        seed=17,
    )
    first = dataset.get_validation_dataset()
    second = dataset.get_validation_dataset()
    assert len(first) == 5
    np.testing.assert_array_equal(first.sample_indices, second.sample_indices)


def test_action_metric_uses_aligned_non_padded_targets():
    class PolicyShape:
        n_obs_steps = 2
        horizon = 2

    batch = {
        "action": torch.tensor(
            [[[100.0], [1.0], [2.0], [999.0]]]
        ),
        "action_is_pad": torch.tensor([[False, False, True, True]]),
    }
    target, valid = aligned_action_target(PolicyShape(), batch)
    prediction = torch.tensor([[[2.0], [200.0]]])
    assert target.tolist() == [[[1.0], [2.0]]]
    assert valid.tolist() == [[True, False]]
    assert masked_action_mse(prediction, target, valid).item() == 1.0


def test_live_eef_state_matches_dataset_layout():
    obs = {
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.array([0.1, 0.2, 0.3, 0.4]),
        "robot0_gripper_qpos": np.array([0.5, -0.5]),
    }
    state = agent_state_from_obs(obs, state_mode="eef")
    np.testing.assert_allclose(state, [1, 2, 3, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_metric_aggregation():
    base = dict(
        steps=100,
        episode_return=1.0,
        max_reward=1.0,
        timeout=False,
        num_policy_calls=5,
        wall_time_sec=1.0,
        inference_time_mean_ms=10.0,
        inference_time_p95_ms=12.0,
    )
    episodes = [
        EpisodeMetrics(episode=0, seed=1, success=True, **base),
        EpisodeMetrics(episode=1, seed=2, success=False, **{**base, "timeout": True}),
    ]
    result = aggregate_episodes(episodes)
    assert result["success_rate"] == 0.5
    assert result["timeout_rate"] == 0.5


def test_robosuite_14_uses_compatible_physics():
    kwargs = _environment_kwargs(
        {"env_version": "1.4.1", "env_kwargs": {}},
        ("agentview", "robot0_eye_in_hand"),
    )
    assert kwargs["lite_physics"] is False


def test_threading_success_requires_needle_center_to_reach_ring():
    ring_center = np.array([1.0, 2.0, 3.0])
    assert needle_center_reaches_ring(
        ring_center + np.array([0.0, 0.011, 0.0]),
        ring_center,
        ring_radius=0.012,
    )
    # A free tip at the ring while the needle center remains one half-length
    # away is not a deep enough insertion.
    assert not needle_center_reaches_ring(
        ring_center + np.array([0.0, 0.06, 0.0]),
        ring_center,
        ring_radius=0.012,
    )


def test_threading_camera_size_can_match_external_policy():
    kwargs = _environment_kwargs(
        {"env_version": "1.4.1", "env_kwargs": {}},
        ("agentview", "robot0_eye_in_hand"),
        camera_size=84,
    )
    assert kwargs["camera_heights"] == [84, 84]
    assert kwargs["camera_widths"] == [84, 84]


def test_threading_supports_mixed_camera_sizes():
    kwargs = _environment_kwargs(
        {"env_version": "1.4.1", "env_kwargs": {}},
        ("agentview", "robot0_eye_in_hand", "threading_closeup"),
        camera_size=(84, 84, 128),
    )
    assert kwargs["camera_heights"] == [84, 84, 128]
    assert kwargs["camera_widths"] == [84, 84, 128]


def test_live_images_match_robomimic_vertical_orientation():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    result = image_from_obs({"camera": image}, "camera")
    np.testing.assert_allclose(result, np.moveaxis(image[::-1] / 255.0, -1, 0))


def test_threading_config_selects_rollout_workspace():
    cfg = OmegaConf.load("pushbox/configs/threading_arp.yaml")
    workspace_cls = get_class(cfg._target_)
    assert workspace_cls.__name__ == "ThreadingARPWorkspace"


def test_threading_v2_config_is_strong_two_view_base_policy():
    cfg = OmegaConf.load("pushbox/configs/threading_arp_v2.yaml")
    v3_cfg = OmegaConf.load("pushbox/configs/threading_arp_v3.yaml")
    policy_cls = get_class(cfg.policy._target_)
    assert policy_cls.__name__ == "ThreadingARPolicy"
    assert cfg.task.shape_meta.obs.agent_pos.shape == [8]
    assert cfg.task.dataset.state_mode == "eef"
    assert list(cfg.task.dataset.camera_output_keys) == ["top45", "wrist"]
    assert len(cfg.policy.camera_obs_keys) == 2
    assert "auxiliary_dataset_path" not in cfg.task.dataset
    assert cfg.policy.arp_cfg.action_predictor == "regression"
    assert cfg.policy.use_view_fusion is False
    assert cfg.policy.horizon == v3_cfg.policy.horizon == 8
    assert cfg.policy.n_action_steps == v3_cfg.policy.n_action_steps == 8
    assert cfg.policy.arp_cfg.plan_steps == v3_cfg.policy.arp_cfg.plan_steps == 0
    assert cfg.training.gradient_accumulate_every == 8
    assert cfg.training.num_epochs == v3_cfg.training.num_epochs == 20
    assert cfg.val_dataloader.shuffle is False


def test_threading_v3_config_uses_strong_three_view_base_policy():
    cfg = OmegaConf.load("pushbox/configs/threading_arp_v3.yaml")
    policy_cls = get_class(cfg.policy._target_)
    assert policy_cls.__name__ == "ThreadingARPolicy"
    assert len(cfg.policy.camera_obs_keys) == 3
    assert cfg.policy.backbone == "resnet18"
    assert cfg.policy.arp_cfg.n_embd == 256
    assert cfg.policy.arp_cfg.num_layers == 8
    assert cfg.policy.n_action_steps == 8
    assert cfg.policy.arp_cfg.action_predictor == "regression"
    assert cfg.policy.use_view_fusion is False
    assert cfg.training.gradient_accumulate_every == 8
    assert cfg.val_dataloader.shuffle is False
