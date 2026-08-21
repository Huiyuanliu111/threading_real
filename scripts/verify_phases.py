"""Smoke test for three-phase PushBox logic."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

from envs.pushbox_env import (
    PHASE_Y_APPROACH,
    PHASE_Y_CROSS,
    PHASE_NAMES,
    PushBoxEnv,
)


def test_infer_phase():
    env = PushBoxEnv.__new__(PushBoxEnv)
    assert env._infer_phase_from_y(-0.22) == 0
    assert env._infer_phase_from_y(-0.06) == 0
    assert env._infer_phase_from_y(-0.05) == 1
    assert env._infer_phase_from_y(0.0) == 1
    assert env._infer_phase_from_y(0.06) == 1
    assert env._infer_phase_from_y(0.07) == 2
    print("infer_phase: OK")


def test_env_reset_phase_state():
    env = PushBoxEnv(
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        horizon=300,
        hard_reset=False,
    )
    env.reset()
    assert env.current_phase == 0
    assert env.phase_success == [False, False, False]
    box_y = env._box_y()
    assert box_y < PHASE_Y_APPROACH, f"expected spawn in approach zone, got y={box_y}"
    info = env._build_phase_info()
    assert info["current_phase_name"] == PHASE_NAMES[0]
    env.close()
    print(f"reset_phase_state: OK (spawn box_y={box_y:.3f})")


def test_phase_advance_by_teleport():
    env = PushBoxEnv(
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        horizon=300,
        hard_reset=False,
    )
    env.reset()
    joint = env.box.joints[0]
    qpos = np.array(env.sim.data.get_joint_qpos(joint), dtype=float)

    for target_y, expected_phase in [(-0.04, 1), (0.10, 2)]:
        qpos[1] = target_y
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
        env._update_phase(env._box_y())
        assert env.current_phase == expected_phase, (
            f"y={target_y}: expected phase {expected_phase}, got {env.current_phase}"
        )

    assert env.phase_success[0] and env.phase_success[1]
    env.close()
    print("phase_advance_by_teleport: OK")


if __name__ == "__main__":
    test_infer_phase()
    test_env_reset_phase_state()
    test_phase_advance_by_teleport()
    print("All phase checks passed.")
