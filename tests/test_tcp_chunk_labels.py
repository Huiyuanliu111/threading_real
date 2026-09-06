from __future__ import annotations

import numpy as np

from threading_task.tcp_chunk_labels import label_tcp_motion, smooth_chunk_labels, tcp_motion_metrics


def test_tcp_motion_metrics_use_cartesian_action_units() -> None:
    positions = np.zeros((3, 3), dtype=np.float64)
    actions = np.zeros((3, 7), dtype=np.float64)
    actions[:, 0] = 0.01
    actions[:, 3] = 0.02

    metrics = tcp_motion_metrics(positions, actions, fps=10, smoothing_window=1)

    np.testing.assert_allclose(metrics["tcp_linear_speed_mps"], 0.1)
    np.testing.assert_allclose(metrics["tcp_angular_speed_radps"], 0.2)


def test_tcp_curvature_ignores_near_stationary_direction_noise() -> None:
    positions = np.zeros((3, 3), dtype=np.float64)
    actions = np.zeros((3, 7), dtype=np.float64)
    actions[0, :3] = [1.0e-6, 0.0, 0.0]
    actions[1, :3] = [0.0, 1.0e-6, 0.0]

    metrics = tcp_motion_metrics(positions, actions, fps=30, smoothing_window=1)

    np.testing.assert_allclose(metrics["tcp_curvature_radps"], 0.0)


def test_slow_motion_maps_to_a_shorter_chunk_than_fast_motion() -> None:
    metrics = {
        "tcp_linear_speed_mps": np.array([0.001, 0.01, 0.10, 1.0]),
        "tcp_angular_speed_radps": np.zeros(4),
        "tcp_curvature_radps": np.zeros(4),
        "tcp_acceleration_mps2": np.zeros(4),
        "gripper_speed_mps": np.zeros(4),
    }

    chunks, score = label_tcp_motion(metrics, candidate_chunks=(1, 2, 4, 8))

    assert chunks[0] < chunks[-1]
    assert score[0] > score[-1]


def test_chunk_median_smoothing_removes_a_single_frame_spike() -> None:
    result = smooth_chunk_labels(
        np.array([1, 1, 20, 1, 1]), candidate_chunks=(1, 2, 4, 8, 20), window=3
    )
    assert np.array_equal(result, np.array([1, 1, 1, 1, 1]))
