from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import pty
import select
import sys
import threading

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "deployment" / "cartesian.py"
SPEC = importlib.util.spec_from_file_location("deployment_cartesian", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_start_enter_does_not_leak_into_episode_stop(monkeypatch):
    master, slave = pty.openpty()
    terminal = os.fdopen(slave, "r")
    monkeypatch.setattr(runner.sys, "stdin", terminal)
    try:
        os.write(master, b"\n")  # leftover input from before the start prompt
        assert select.select([slave], [], [], 1)[0]

        def show_start_prompt(*args, **kwargs):
            assert not select.select([slave], [], [], 0)[0]
            os.write(master, b"\n\n")  # queued/repeated start Enter

        monkeypatch.setattr(runner, "print", show_start_prompt, raising=False)
        assert runner.wait_for_episode_enter("Press Enter to start", lambda: False)
        assert not runner.episode_end_requested()

        # A genuinely new Enter after startup must still stop immediately.
        os.write(master, b"\n")
        assert select.select([slave], [], [], 1)[0]
        assert runner.episode_end_requested()
        assert not runner.episode_end_requested()
    finally:
        terminal.close()
        os.close(master)


def test_synchronous_execution_is_default() -> None:
    parser = runner.build_parser()

    defaults = parser.parse_args(["checkpoint"])
    assert defaults.synchronous is True
    assert defaults.episodes == 1
    assert defaults.grasp_before_inference is True
    assert parser.parse_args(["checkpoint", "--no-synchronous"]).synchronous is False
    assert parser.parse_args(["checkpoint", "--no-grasp-before-inference"]).grasp_before_inference is False
    assert parser.parse_args(["checkpoint", "--episodes", "10"]).episodes == 10


def test_auto_detects_pi05_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi05"}))
    sentinel = object()
    monkeypatch.setattr(runner, "PI05DeploymentPolicy", lambda *args, **kwargs: sentinel)

    policy = runner.load_deployment_policy(
        tmp_path,
        device="cpu",
        weights="model",
        policy_kind="auto",
        task="insert the grasped block through the needle",
    )

    assert policy is sentinel


def test_adaptive_pi05_parser_modes() -> None:
    parser = runner.build_parser()
    required = parser.parse_args(
        [
            "checkpoint",
            "--chunk-selector",
            "selector",
            "--prediction-mode",
            "required_only",
            "--trace-output",
            "trace.jsonl",
        ]
    )

    assert required.chunk_selector == Path("selector")
    assert required.prediction_mode == "required_only"
    assert required.trace_output == Path("trace.jsonl")


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
