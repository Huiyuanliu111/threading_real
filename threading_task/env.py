"""Native robosuite 1.5 environment adapter for the Threading task."""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import ACTION_DIM, AGENT_STATE_DIM, EEF_AGENT_STATE_DIM, read_env_metadata
from .visualization import as_rgb_uint8


def register_mimicgen() -> None:
    """Register the Threading implementation matching the active robosuite."""
    import robosuite  # noqa: PLC0415

    version_numbers = tuple(
        int(part)
        for part in str(getattr(robosuite, "__version__", "")).split(".")[:2]
        if part.isdigit()
    )
    if version_numbers and version_numbers < (1, 5):
        import mimicgen.envs.robosuite.threading  # noqa: F401, PLC0415
    else:
        import envs.threading_env  # noqa: F401, PLC0415


def configure_headless_rendering() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.pop("DISPLAY", None)


def _install_mujoco_mass_matrix_compatibility() -> None:
    """Bridge the MuJoCo 3.11 mass-matrix API change for robosuite 1.5."""
    import mujoco  # noqa: PLC0415

    if not hasattr(mujoco.MjData, "qM") and hasattr(mujoco.MjData, "M"):
        # robosuite calls the legacy ``mj_fullM(model, dst, data.qM)`` API.
        # MuJoCo 3.11 changed it to ``mj_fullM(model, data, dst)`` and removed
        # qM. Return the owning data object through the legacy attribute so the
        # wrapper can reorder the arguments without searching for it globally.
        mujoco.MjData.qM = property(lambda data: data)
        original_mj_full_m = mujoco.mj_fullM
        if not getattr(original_mj_full_m, "_pushbox_legacy_compatible", False):
            def legacy_compatible_mj_full_m(model, second, third):
                if isinstance(third, mujoco.MjData):
                    return original_mj_full_m(model, third, second)
                return original_mj_full_m(model, second, third)

            legacy_compatible_mj_full_m._pushbox_legacy_compatible = True
            mujoco.mj_fullM = legacy_compatible_mj_full_m


class ThreadingEnvAdapter:
    """Small subset of the robomimic EnvBase API without mujoco-py."""

    def __init__(self, env):
        self.env = env
        self._match_dataset_rendering()

    def _match_dataset_rendering(self) -> None:
        """Match the geom groups used to render robomimic image datasets.

        Threading core images contain the normal visual meshes from geom group
        1. This remains true for the native 1.5 port: disabling the group leaves
        the policy cameras showing almost only the dark background and creates
        a catastrophic train/eval visual-domain mismatch.
        """
        context = getattr(self.env.sim, "_render_context_offscreen", None)
        visual_options = getattr(context, "vopt", None)
        geom_groups = getattr(visual_options, "geomgroup", None)
        if geom_groups is not None and len(geom_groups) > 1:
            geom_groups[1] = 1

    def reset(self):
        obs = self.env.reset()
        self._match_dataset_rendering()
        return obs

    def step(self, action):
        return self.env.step(action)

    def reset_to(self, state: dict[str, Any]):
        if "model" in state:
            self.env.reset()
            self.env.reset_from_xml_string(state["model"])
            self.env.sim.reset()
            self._match_dataset_rendering()
        if "states" in state:
            self.env.sim.set_state_from_flattened(np.asarray(state["states"]))
            self.env.sim.forward()
        return self.env._get_observations(force_update=True)

    def is_success(self):
        return {"task": bool(self.env._check_success())}

    def get_reward(self):
        return float(self.env.reward())

    @property
    def action_dimension(self):
        return int(self.env.action_spec[0].shape[0])

    def seed(self, seed: int):
        np.random.seed(seed)
        if hasattr(self.env, "seed"):
            self.env.seed(seed)

    def close(self):
        self.env.close()


def _environment_kwargs(
    env_meta: dict[str, Any],
    cameras: tuple[str, ...],
    camera_size: int | tuple[int, ...] = 96,
) -> dict[str, Any]:
    cameras = tuple(cameras)
    if isinstance(camera_size, int):
        camera_sizes = [int(camera_size)] * len(cameras)
    else:
        camera_sizes = [int(size) for size in camera_size]
        if len(camera_sizes) != len(cameras):
            raise ValueError(
                f"Expected one size per camera, got {len(camera_sizes)} sizes "
                f"for {len(cameras)} cameras"
            )
    kwargs = deepcopy(env_meta.get("env_kwargs", {}))
    for key in (
        "env_name",
        "render",
        "render_offscreen",
        "use_image_obs",
        "postprocess_visual_obs",
        "camera_name",
        "camera_height",
        "camera_width",
    ):
        kwargs.pop(key, None)
    kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=list(cameras),
        camera_heights=camera_sizes,
        camera_widths=camera_sizes,
        hard_reset=False,
    )
    # robosuite 1.5 defaults to its faster "lite physics" stepping path. Its
    # controller requires this to be disabled for trajectories collected with
    # robosuite <= 1.4.1, including the MimicGen core datasets.
    import robosuite  # noqa: PLC0415

    dataset_version = str(env_meta.get("env_version", ""))
    dataset_version_numbers = tuple(
        int(part) for part in dataset_version.split(".")[:2] if part.isdigit()
    )
    runtime_version_numbers = tuple(
        int(part)
        for part in str(getattr(robosuite, "__version__", "")).split(".")[:2]
        if part.isdigit()
    )
    runtime_uses_composite_controller = runtime_version_numbers >= (1, 5)
    if (
        runtime_uses_composite_controller
        and dataset_version_numbers
        and dataset_version_numbers < (1, 5)
    ):
        kwargs.setdefault("lite_physics", False)
    old_controller = kwargs.get("controller_configs")
    if (
        runtime_uses_composite_controller
        and isinstance(old_controller, dict)
        and old_controller.get("type")
        in {"JOINT_VELOCITY", "JOINT_TORQUE", "JOINT_POSITION", "OSC_POSITION", "OSC_POSE", "IK_POSE"}
        and "body_parts" not in old_controller
    ):
        # robosuite <=1.4 stored a single arm controller at the top level.
        # robosuite >=1.5 expects a composite controller with per-part configs.
        from robosuite.controllers import load_composite_controller_config  # noqa: PLC0415

        robots = kwargs.get("robots", ["Panda"])
        robot_name = robots[0] if isinstance(robots, (list, tuple)) else robots
        controller = load_composite_controller_config(robot=str(robot_name))
        arm_name = "right"
        arm = deepcopy(controller["body_parts"][arm_name])
        direct_keys = (
            "type",
            "input_max",
            "input_min",
            "output_max",
            "output_min",
            "kp",
            "impedance_mode",
            "kp_limits",
            "position_limits",
            "orientation_limits",
            "uncouple_pos_ori",
            "interpolation",
            "ramp_ratio",
        )
        for key in direct_keys:
            if key in old_controller:
                arm[key] = deepcopy(old_controller[key])
        if "damping" in old_controller:
            arm["damping_ratio"] = old_controller["damping"]
        if "damping_limits" in old_controller:
            arm["damping_ratio_limits"] = deepcopy(old_controller["damping_limits"])
        if "control_delta" in old_controller:
            arm["input_type"] = "delta" if old_controller["control_delta"] else "absolute"
        arm["gripper"] = {"type": "GRIP"}
        controller["body_parts"][arm_name] = arm
        kwargs["controller_configs"] = controller
    return kwargs


def create_threading_env(
    dataset_path: str | Path,
    env_name: str | None = None,
    camera_names: tuple[str, ...] = ("agentview", "robot0_eye_in_hand"),
    camera_size: int | tuple[int, ...] = 96,
):
    configure_headless_rendering()
    _install_mujoco_mass_matrix_compatibility()
    register_mimicgen()
    import robosuite  # noqa: PLC0415

    env_meta = read_env_metadata(dataset_path)
    if not env_meta:
        raise ValueError(f"{dataset_path}: missing data/env_args metadata")
    resolved_name = env_name or env_meta.get("env_name")
    if not resolved_name or "Threading" not in resolved_name:
        raise ValueError(f"Expected a Threading environment, got {resolved_name!r}")
    env = ThreadingEnvAdapter(
        robosuite.make(
            resolved_name,
            **_environment_kwargs(env_meta, camera_names, camera_size=camera_size),
        )
    )
    if env.action_dimension != ACTION_DIM:
        raise ValueError(
            f"Threading integration expects {ACTION_DIM}D OSC actions, "
            f"environment reports {env.action_dimension}"
        )
    return env, env_meta


def agent_state_from_obs(
    obs: dict[str, Any],
    joint_key: str = "robot0_joint_pos",
    gripper_key: str = "robot0_gripper_qpos",
    eef_pos_key: str = "robot0_eef_pos",
    eef_quat_key: str = "robot0_eef_quat",
    state_mode: str = "joint",
) -> np.ndarray:
    if state_mode not in {"joint", "eef"}:
        raise ValueError(f"state_mode must be 'joint' or 'eef', got {state_mode!r}")
    state_keys = (
        (joint_key, gripper_key)
        if state_mode == "joint"
        else (eef_pos_key, eef_quat_key, gripper_key)
    )
    missing = [key for key in state_keys if key not in obs]
    if missing:
        raise KeyError(f"Environment observation is missing {missing}; got {sorted(obs)}")
    gripper = np.asarray(obs[gripper_key]).reshape(-1)
    if state_mode == "joint":
        state = np.concatenate([np.asarray(obs[joint_key]).reshape(-1), gripper])
        expected_dim = AGENT_STATE_DIM
    else:
        state = np.concatenate(
            [
                np.asarray(obs[eef_pos_key]).reshape(-1),
                np.asarray(obs[eef_quat_key]).reshape(-1),
                gripper[:1],
            ]
        )
        expected_dim = EEF_AGENT_STATE_DIM
    state = state.astype(np.float32)
    if state.shape != (expected_dim,):
        raise ValueError(f"Expected {expected_dim}D {state_mode} state, got {state.shape}")
    return state


def image_from_obs(obs: dict[str, Any], key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"Environment observation is missing {key!r}; got {sorted(obs)}")
    # Raw robosuite camera observations use MuJoCo's bottom-up framebuffer
    # convention. robomimic flips them before writing image observations to
    # HDF5, so online inputs need the same transform as training data.
    image = np.flipud(as_rgb_uint8(np.asarray(obs[key]))).astype(np.float32) / 255.0
    return np.moveaxis(image, -1, 0)


def success_from_env(env) -> bool:
    result = env.is_success()
    return bool(result.get("task", False)) if isinstance(result, dict) else bool(result)


def seed_env(env, seed: int) -> None:
    np.random.seed(seed)
    try:
        env.seed(seed)
    except (AttributeError, TypeError, NotImplementedError):
        pass
