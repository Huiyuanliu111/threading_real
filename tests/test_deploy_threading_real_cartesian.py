from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "deploy_threading_real_cartesian.py"
SPEC = importlib.util.spec_from_file_location("deploy_threading_real_cartesian", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_cartesian_pose_error() -> None:
    current = np.eye(4)
    target = np.eye(4)
    target[:3, 3] = [0.003, 0.004, 0.0]
    target[:3, :3] = Rotation.from_rotvec([0.0, 0.0, 0.1]).as_matrix()

    translation, rotation = runner.cartesian_pose_error(current, target)

    assert translation == pytest.approx(0.005)
    assert rotation == pytest.approx(0.1)


def test_synchronous_wait_completes_segment_despite_impedance_error() -> None:
    class Manager:
        lock = threading.Lock()
        completed = True

    class Streamer:
        manager = Manager()

    class Client:
        def get_latest_state(self, allow_stale: bool = False):
            return {"q": np.zeros(7), "arm_state": "MOVING"}, {"level": "fresh"}

        def get_tcp_pose_from_q(self, robot_model, q, frame_name=None):
            return np.eye(4)

    target = np.eye(4)
    target[0, 3] = 0.004
    result = runner.wait_for_trackc_segment(
        Streamer(),
        Client(),
        object(),
        target,
        position_tolerance=0.001,
        rotation_tolerance=0.02,
        timeout=0.1,
        settle_samples=1,
        poll_hz=50.0,
        stop_requested=lambda: False,
    )

    assert result["stopped"] is False
    assert result["within_tolerance"] is False
    assert result["translation_error"] == pytest.approx(0.004)
