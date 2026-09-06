from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "convert_lerobot_v3_to_cartesian.py"
SPEC = importlib.util.spec_from_file_location("convert_lerobot_v3_to_cartesian", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
conversion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conversion)
URDF = ROOT / "remote_controller/src/remote_controller/assets/panda/panda_arm.urdf"


def test_zero_configuration_tcp_pose_matches_panda_urdf() -> None:
    fk = conversion.UrdfForwardKinematics(URDF)
    pose = fk.pose(np.zeros(7))

    # panda_hand_tcp is 10.34 cm below the downward-facing hand at q=0.
    np.testing.assert_allclose(pose[:3, 3], [0.088, 0.0, 0.8226], atol=1e-9)
    np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-12)


def test_cartesian_delta_is_world_frame_current_to_target() -> None:
    fk = conversion.UrdfForwardKinematics(URDF)
    state = np.array([[0.1, -0.2, 0.3, -1.2, 0.2, 1.5, 0.7, 0.08]])
    target = state.copy()
    target[0, 0] += 0.01
    target[0, 7] = 0.04
    action = conversion.cartesian_delta_actions(state, target, fk)[0]
    current = fk.pose(state[0, :7])
    goal = fk.pose(target[0, :7])

    np.testing.assert_allclose(action[:3], goal[:3, 3] - current[:3, 3], atol=1e-7)
    expected_rotation = Rotation.from_matrix(goal[:3, :3] @ current[:3, :3].T).as_rotvec()
    np.testing.assert_allclose(action[3:6], expected_rotation, atol=1e-7)
    np.testing.assert_allclose(action[6], -0.04, atol=1e-7)
