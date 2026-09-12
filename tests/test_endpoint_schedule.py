import json

import numpy as np
import pytest

from scripts.deployment.endpoint_schedule import EndpointExecutionSchedule
from scripts.deployment.cartesian import integrate_cartesian_delta_chunk


def poses_at(points):
    poses = np.tile(np.eye(4), (len(points), 1, 1))
    poses[:, :3, 3] = points
    return poses


def test_far_path_executes_full_chunk():
    schedule = EndpointExecutionSchedule(np.zeros(3))
    poses = poses_at(np.tile([0.2, 0, 0], (10, 1)))
    steps, diagnostics = schedule.select([0.3, 0, 0], poses)
    assert steps == 10 and diagnostics["execution_phase"] == "coarse"
    json.dumps(diagnostics)  # trace output must remain serializable


def test_current_position_enters_fine_mode_and_episode_reset_clears_it():
    schedule = EndpointExecutionSchedule(np.zeros(3))
    far_poses = poses_at(np.tile([0.2, 0, 0], (10, 1)))
    assert schedule.select([0.04, 0, 0], far_poses)[0] == 3
    assert schedule.select([0.3, 0, 0], far_poses)[0] == 3
    schedule.reset()
    assert schedule.select([0.3, 0, 0], far_poses)[0] == 10


def test_lookahead_detects_crossing_even_when_waypoints_are_outside():
    schedule = EndpointExecutionSchedule(np.zeros(3))
    poses = poses_at(np.tile([-0.1, 0, 0], (10, 1)))
    steps, diagnostics = schedule.select([0.1, 0, 0], poses)
    assert steps == 3
    assert diagnostics["first_fine_region_segment"] == 1
    assert diagnostics["predicted_path_min_distance_m"] == pytest.approx(0)


def test_future_entry_shortens_prefix_before_contact():
    schedule = EndpointExecutionSchedule(np.zeros(3))
    points = np.zeros((10, 3))
    points[:, 0] = np.linspace(0.2, 0.02, 10)
    steps, diagnostics = schedule.select([0.22, 0, 0], poses_at(points))
    assert steps == 3
    assert diagnostics["first_fine_region_segment"] > 3


def test_schedule_uses_clipped_execution_path():
    schedule = EndpointExecutionSchedule(np.zeros(3), fine_radius_m=.06)
    current = np.eye(4)
    current[0, 3] = .25
    actions = np.zeros((10, 7))
    actions[0, 0] = -.24  # raw endpoint is close; actual clipped trajectory stays far
    poses, _, _ = integrate_cartesian_delta_chunk(
        actions, current, .023, max_first_translation=.04, max_step_translation=.025,
        max_first_rotation=.4, max_step_rotation=.25, workspace_min=None, workspace_max=None)
    assert schedule.select(current[:3, 3], poses)[0] == 10


def test_stationary_path_at_endpoint_is_finite():
    schedule = EndpointExecutionSchedule(np.zeros(3))
    steps, diagnostics = schedule.select([0, 0, 0], poses_at(np.zeros((10, 3))))
    assert steps == 3
    assert diagnostics["predicted_path_min_distance_m"] == 0


def test_schedule_rejects_incomplete_or_nonfinite_prediction():
    schedule = EndpointExecutionSchedule(np.zeros(3))
    with pytest.raises(ValueError):
        schedule.select([0, 0, 0], poses_at(np.zeros((5, 3))))
    points = np.zeros((10, 3)); points[0, 0] = np.nan
    with pytest.raises(ValueError):
        schedule.select([0, 0, 0], poses_at(points))
