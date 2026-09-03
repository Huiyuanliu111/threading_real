from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "deploy_threading_real.py"
SPEC = importlib.util.spec_from_file_location("deploy_threading_real", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_image_to_policy_tensor_resizes_and_scales() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[..., 0] = 255
    result = runner.image_to_policy_tensor(image, 8)
    assert result.shape == (3, 8, 8)
    assert result.dtype == torch.float32
    assert torch.all(result[0] == 1.0)
    assert torch.all(result[1:] == 0.0)


def test_sanitize_action_chunk_limits_first_and_following_steps() -> None:
    q_now = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    actions = np.tile(np.r_[q_now, 0.09], (2, 1))
    actions[0, 0] = 1.0
    actions[1, 0] = 2.0

    safe_q, widths, stats = runner.sanitize_action_chunk(
        actions,
        q_now,
        joint_limit_margin=0.05,
        max_first_delta=0.05,
        max_step_delta=0.02,
    )

    assert safe_q[0, 0] == pytest.approx(0.05)
    assert safe_q[1, 0] == pytest.approx(0.07)
    assert np.all(widths == 0.08)
    assert stats["raw_first_delta"] == pytest.approx(1.0)
    assert stats["raw_step_delta"] == pytest.approx(1.0)


def test_sanitize_action_chunk_rejects_non_finite_values() -> None:
    q_now = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    actions = np.tile(np.r_[q_now, 0.04], (2, 1))
    actions[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        runner.sanitize_action_chunk(
            actions,
            q_now,
            joint_limit_margin=0.05,
            max_first_delta=0.05,
            max_step_delta=0.02,
        )


def test_raw_step_delta_excludes_current_to_first_target() -> None:
    q_now = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    actions = np.tile(np.r_[q_now, 0.04], (2, 1))
    actions[:, 0] += [0.04, 0.041]
    _, _, stats = runner.sanitize_action_chunk(
        actions,
        q_now,
        joint_limit_margin=0.05,
        max_first_delta=0.05,
        max_step_delta=0.02,
    )
    assert stats["raw_first_delta"] == pytest.approx(0.04)
    assert stats["raw_step_delta"] == pytest.approx(0.001)


def test_integrate_delta_actions() -> None:
    state = np.arange(8, dtype=np.float64)
    deltas = np.array([[0.1] * 8, [-0.2] * 8])
    absolute = runner.integrate_delta_actions(deltas, state)
    np.testing.assert_allclose(absolute[0], state + 0.1)
    np.testing.assert_allclose(absolute[1], state - 0.1)


def test_gripper_hysteresis() -> None:
    kwargs = {"close_threshold": 0.035, "open_threshold": 0.055}
    assert runner.choose_gripper_transition("open", 0.02, **kwargs) == "close"
    assert runner.choose_gripper_transition("open", 0.04, **kwargs) is None
    assert runner.choose_gripper_transition("closed", 0.07, **kwargs) == "open"
    assert runner.choose_gripper_transition("closed", 0.04, **kwargs) is None


def test_stack_observations_has_policy_shapes() -> None:
    frame = runner.ObservationFrame(
        sideview=torch.zeros(3, 96, 96),
        wrist=torch.ones(3, 96, 96),
        agent_pos=torch.arange(8, dtype=torch.float32),
        timestamp=0.0,
    )
    obs = runner.stack_observations([frame, frame], "cpu")
    assert obs["sideview"].shape == (1, 2, 3, 96, 96)
    assert obs["wrist"].shape == (1, 2, 3, 96, 96)
    assert obs["agent_pos"].shape == (1, 2, 8)
