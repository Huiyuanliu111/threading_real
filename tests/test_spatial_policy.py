from __future__ import annotations

import json

import numpy as np
import torch

from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from scripts.calibration.fit import (
    farthest_point_rows,
    fit_projection_matrix,
    reprojection_errors,
)
from scripts.calibration.live import (
    _parser,
    fit_camera_extrinsics,
    load_calibration_progress,
    save_calibration_progress,
)
from threading_task.spatial_geometry import (
    load_projection_calibration,
    project_points_numpy,
    triangulate_dlt,
)
from threading_task.spatial_policy import ThreadingSpatialARPolicy


def synthetic_projections(size: int = 64) -> np.ndarray:
    intrinsic = np.array(
        [[80.0, 0.0, size / 2], [0.0, 80.0, size / 2], [0.0, 0.0, 1.0]]
    )
    first = intrinsic @ np.c_[np.eye(3), np.zeros(3)]
    second = intrinsic @ np.c_[np.eye(3), np.array([-0.12, 0.0, 0.0])]
    return np.stack((first, second)).astype(np.float32)


def write_calibration(path, size: int = 64) -> None:
    matrices = synthetic_projections(size)
    payload = {
        "format": "threading-spatial-projection-v1",
        "cameras": {
            key: {
                "image_width": size,
                "image_height": size,
                "projection_matrix": matrix.tolist(),
            }
            for key, matrix in zip(("sideview", "frontview"), matrices)
        },
    }
    path.write_text(json.dumps(payload))


def test_projection_and_dlt_triangulation_are_inverse() -> None:
    matrices = synthetic_projections()
    points = np.array([[0.02, -0.04, 0.8], [-0.08, 0.03, 1.1]], dtype=np.float32)
    pixels, _ = project_points_numpy(points, matrices)
    recovered = triangulate_dlt(torch.from_numpy(pixels), torch.from_numpy(matrices))
    np.testing.assert_allclose(recovered.numpy(), points, atol=1e-5)


def test_bad_recorded_calibration_is_rejected(tmp_path) -> None:
    calibration = tmp_path / "bad.json"
    write_calibration(calibration)
    payload = json.loads(calibration.read_text())
    payload["cameras"]["sideview"]["reprojection_rmse_px"] = 12.0
    calibration.write_text(json.dumps(payload))
    import pytest

    with pytest.raises(ValueError, match="RMSE"):
        load_projection_calibration(calibration, ("sideview", "frontview"), 64, 64)


def test_calibration_dlt_recovers_reprojection() -> None:
    rng = np.random.default_rng(4)
    xyz = rng.uniform([-0.2, -0.15, 0.7], [0.2, 0.15, 1.2], size=(30, 3))
    matrix = synthetic_projections()[0]
    pixels, _ = project_points_numpy(xyz, matrix[None])
    fitted = fit_projection_matrix(xyz, pixels[:, 0])
    assert np.sqrt(np.mean(reprojection_errors(fitted, xyz, pixels[:, 0]) ** 2)) < 1e-4


def test_calibration_candidates_cover_spatial_extremes() -> None:
    points = np.c_[np.linspace(0, 1, 101), np.zeros(101), np.zeros(101)]
    rows = farthest_point_rows(points, 2)
    assert set(rows.tolist()) == {0, 100}


def test_live_pnp_recovers_camera_from_base() -> None:
    rng = np.random.default_rng(7)
    points = rng.uniform([-0.15, -0.12, 0.65], [0.18, 0.14, 1.10], size=(40, 3))
    intrinsic = np.array([[610.0, 0.0, 320.0], [0.0, 608.0, 240.0], [0.0, 0.0, 1.0]])
    rotation_vector = np.array([0.08, -0.13, 0.04])
    import cv2

    rotation = cv2.Rodrigues(rotation_vector)[0]
    translation = np.array([0.03, -0.02, 0.25])
    camera_from_base = np.c_[rotation, translation]
    pixels, _ = project_points_numpy(points, (intrinsic @ camera_from_base)[None])
    result = fit_camera_extrinsics(
        points,
        pixels[:, 0],
        intrinsic,
        np.zeros(5),
        reprojection_threshold=1.0,
    )
    np.testing.assert_allclose(result["camera_from_base"][:3, :3], rotation, atol=1e-5)
    np.testing.assert_allclose(result["camera_from_base"][:3, 3], translation, atol=1e-5)


def test_live_calibration_progress_round_trip_is_atomic(tmp_path) -> None:
    progress = tmp_path / "spatial.progress.json"
    records = [
        {
            "tcp_base": [0.4, -0.1, 0.02],
            "pixels": {"sideview": [120.0, 220.0], "frontview": [310.0, 180.0]},
        }
    ]
    save_calibration_progress(progress, records, "side-serial", "front-serial")

    assert load_calibration_progress(
        progress, "side-serial", "front-serial"
    ) == records
    assert not progress.with_suffix(progress.suffix + ".tmp").exists()

    import pytest

    with pytest.raises(ValueError, match="sideview serial"):
        load_calibration_progress(progress, "different-camera", "front-serial")


def test_live_calibration_progress_modes_are_mutually_exclusive() -> None:
    import pytest

    with pytest.raises(SystemExit):
        _parser().parse_args(["--resume-progress", "--fresh"])


def test_spatial_policy_forward_backward(tmp_path) -> None:
    calibration = tmp_path / "calibration.json"
    write_calibration(calibration)
    shape_meta = {
        "action": {"shape": [7]},
        "obs": {
            "agent_pos": {"shape": [8], "type": "low_dim"},
            "tcp_pos": {"shape": [3], "type": "low_dim"},
            "sideview": {"shape": [3, 64, 64], "type": "rgb"},
            "wrist": {"shape": [3, 64, 64], "type": "rgb"},
            "frontview": {"shape": [3, 64, 64], "type": "rgb"},
        },
    }
    policy = ThreadingSpatialARPolicy(
        shape_meta,
        str(calibration),
        pretrained=False,
        hidden_dim=64,
        transformer_layers=1,
        transformer_heads=4,
        heatmap_size=64,
        triangulation_loss_weight=0.1,
    )
    normalizer = LinearNormalizer()
    normalizer.fit(
        {"agent_pos": torch.randn(32, 8), "action": torch.randn(32, 7)},
        last_n_dims=1,
        mode="limits",
    )
    policy.set_normalizer(normalizer)
    batch_size, steps = 2, 2
    goal = np.array([[0.02, 0.01, 0.9], [-0.03, -0.02, 1.0]], dtype=np.float32)
    pixels, _ = project_points_numpy(goal, synthetic_projections())
    batch = {
        "obs": {
            "agent_pos": torch.randn(batch_size, steps, 8),
            "tcp_pos": torch.tensor(goal)[:, None].repeat(1, steps, 1) - 0.02,
            "sideview": torch.rand(batch_size, steps, 3, 64, 64),
            "wrist": torch.rand(batch_size, steps, 3, 64, 64),
            "frontview": torch.rand(batch_size, steps, 3, 64, 64),
        },
        "spatial_goal_pixels": torch.from_numpy(pixels),
        "spatial_goal_valid": torch.ones(batch_size, 2, dtype=torch.bool),
        "spatial_goal_xyz": torch.from_numpy(goal),
    }
    losses = policy.compute_loss(batch)
    assert set(losses) == {"spatial_heatmap_ce", "spatial_xyz"}
    assert all(torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()
    assert policy.obs_encoder[0].weight.grad is not None
    policy.eval()
    logits = policy._heatmap_logits(batch["obs"])
    changed_state = {key: value.clone() for key, value in batch["obs"].items()}
    changed_state["agent_pos"].normal_(mean=20.0, std=5.0)
    changed_state["tcp_pos"].normal_(mean=-10.0, std=2.0)
    torch.testing.assert_close(policy._heatmap_logits(changed_state), logits)
    result = policy.predict_action(batch["obs"])
    assert result["action"].shape == (batch_size, 1, 7)
    assert result["spatial_heatmap_logits"].shape == (batch_size, 2, 64, 64)
    assert torch.isfinite(result["spatial_goal_xyz"]).all()

    exact_logits = torch.full((batch_size, 2, 64, 64), -20.0)
    heatmap_pixels = np.rint(pixels).astype(int)
    for sample in range(batch_size):
        for camera in range(2):
            x, y = heatmap_pixels[sample, camera]
            exact_logits[sample, camera, y, x] = 20.0
    policy._heatmap_logits = lambda _observations: exact_logits
    geometric_result = policy.predict_action(batch["obs"])
    step_norm = geometric_result["action"][:, 0, :3].norm(dim=-1)
    torch.testing.assert_close(step_norm, torch.full_like(step_norm, 0.008), atol=1e-6, rtol=0)
