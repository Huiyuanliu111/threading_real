"""PushBox environment: Franka Panda pushes a block to a goal while avoiding obstacles."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import new_body, new_geom
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

# Block half-extent (matches push_box BoxObject size below).
BOX_HALF = 0.025
BOX_FULL = 2 * BOX_HALF

# Z-offset above the table surface so the box never contacts the table.
BOX_Z_OFFSET = 0.02  # box centre stays 2 cm above the table

# Obstacle Z half-extent (full height = 10 cm).
OBS_HALF_Z = 0.05

# Block spawn region (table xy, metres) — far from goal at (0.0, 0.25).
BOX_INIT_X_RANGE = (0.01, 0.05)
BOX_INIT_Y_RANGE = (-0.25, -0.20)

# Fixed corridor obstacles — reference baseline (half-extents in metres).
V2_FIXED_OBSTACLES: list[dict] = [
    {
        "name": "fixed_obs_0",
        "pos": [0.215, 0.00],
        "half_size": [0.185, 0.04, OBS_HALF_Z],
        "euler": [0.0, 0.0, 0.0],
    },
    {
        "name": "fixed_obs_1",
        "pos": [-0.225, 0.00],
        "half_size": [0.175, 0.04, OBS_HALF_Z],
        "euler": [0.0, 0.0, 0.0],
    },
]
NUM_DEFAULT_OBSTACLES = 2

# Three-phase task boundaries (fixed y thresholds, metres).
PHASE_Y_APPROACH = -0.10
PHASE_Y_CROSS = 0.10
NUM_PHASES = 3
PHASE_NAMES = ("approach", "cross", "reach")


def random_obstacle_configs(rng: np.random.Generator, n: int = NUM_DEFAULT_OBSTACLES) -> list[dict]:
    """Generate randomised obstacle configs for one episode."""
    y_offset = float(rng.uniform(-0.025, 0.025))
    obs_len = float(rng.uniform(0.17, 0.18))
    configs = [
        {
            "name": "fixed_obs_0",
            "pos": [0.215, y_offset],
            "half_size": [obs_len, 0.04, OBS_HALF_Z],
            "euler": [0.0, 0.0, 0.0],
        },
        {
            "name": "fixed_obs_1",
            "pos": [-0.225, y_offset],
            "half_size": [obs_len, 0.04, OBS_HALF_Z],
            "euler": [0.0, 0.0, 0.0],
        },
    ]
    return configs[:n]


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    """Rotation about +Z only (table normal)."""
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def _placeholder_obstacle_configs(n_obstacles: int, goal_pos) -> list[dict]:
    """Static layout for MJCF build only (poses replaced each reset)."""
    return V2_FIXED_OBSTACLES[:n_obstacles]


class PushBoxEnv(ManipulationEnv):
    """2-D box-pushing task on a flat table.

    The robot arm (Franka Panda, OSC_POSE) must push a square block from a
    random initial position to a fixed goal zone while avoiding static obstacles.
    """

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        goal_pos=(0.0, 0.25),
        goal_radius=0.05,
        boundary_half_size=0.35,
        num_obstacles=NUM_DEFAULT_OBSTACLES,
        obstacle_configs=None,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="top45",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=500,
        ignore_done=False,
        hard_reset=True,
        camera_names="top45,sideview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        box_init_y_range=None,
        box_init_x_range=None,
        box_x_sampler=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.table_top_z = float(self.table_offset[2])
        table_half_xy = min(self.table_full_size[0], self.table_full_size[1]) / 2.0
        self.table_xy_limit = table_half_xy - BOX_HALF - 0.03

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        self.goal_pos = np.array(goal_pos, dtype=float)
        self.goal_radius = goal_radius
        self.boundary_half_size = boundary_half_size

        self.num_obstacles = num_obstacles
        self.obstacle_configs = obstacle_configs
        self.placement_initializer = placement_initializer
        self._box_init_y_range: tuple[float, float] | None = box_init_y_range
        self._box_init_x_range: tuple[float, float] | None = box_init_x_range
        self._box_x_sampler = box_x_sampler
        self._obstacle_body_names: list[str] = []
        self.use_object_obs = use_object_obs

        self._obstacle_geom_names: list[str] = []
        self.box_body_id: int | None = None

        self.np_random = np.random.RandomState(seed)

        self.current_phase = 0
        self.phase_step_start = [0, 0, 0]
        self.phase_done = [False, False, False]
        self.phase_success = [False, False, False]
        self.phase_steps = [0, 0, 0]
        self._phase_horizons = self._compute_phase_horizons(horizon)

        self._consecutive_arm_collision: int = 0
        self._consecutive_oob: int = 0
        self._grace_steps: int = 5

        self._grip_active: bool = False

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        table_top_z = self.table_offset[2]

        # ---- goal zone ----
        gx, gy = self.goal_pos
        goal_geom = new_geom(
            name="goal_zone",
            type="cylinder",
            size=[self.goal_radius, 0.001],
            pos=[gx, gy, table_top_z + 0.002],
            rgba=[0.1, 0.8, 0.1, 0.45],
            contype="0",
            conaffinity="0",
            group=1,
        )
        mujoco_arena.worldbody.append(goal_geom)

        # ---- top-down camera ----
        cam_elem = ET.SubElement(
            mujoco_arena.worldbody,
            "camera",
            name="top45",
            pos="0.45 0 1.25",
            xyaxes="0 1 0 -0.7071 0 0.7071",
            fovy="70",
        )

        # ---- obstacles ----
        self._obstacle_geom_names = []
        self._obstacle_body_names = []
        n_obs = self.num_obstacles if self.obstacle_configs is None else len(self.obstacle_configs)
        placeholder = _placeholder_obstacle_configs(n_obs, self.goal_pos)
        obs_list = self.obstacle_configs if self.obstacle_configs is not None else placeholder
        for obs_cfg in obs_list:
            hx, hy, hz = obs_cfg.get("half_size", [0.18, 0.18, OBS_HALF_Z])
            obs_name = obs_cfg["name"]
            obs_body = new_body(name=obs_name, pos=[0, 0, 0], mocap="true")
            col_name = f"{obs_name}_col"
            col_geom = new_geom(
                name=col_name,
                type="box",
                size=[hx, hy, hz],
                rgba=[0.60, 0.30, 0.10, 1.0],
            )
            vis_geom = new_geom(
                name=f"{obs_name}_vis",
                type="box",
                size=[hx, hy, hz],
                rgba=[0.85, 0.55, 0.25, 1.0],
                contype="0",
                conaffinity="0",
                group=1,
            )
            obs_body.append(col_geom)
            obs_body.append(vis_geom)
            mujoco_arena.worldbody.append(obs_body)
            self._obstacle_geom_names.append(col_name)
            self._obstacle_body_names.append(obs_name)

        # ---- movable box ----
        self.box = BoxObject(
            name="push_box",
            size=[BOX_HALF, BOX_HALF, BOX_HALF],
            rgba=[0.8, 0.1, 0.1, 1.0],
            friction=(1.0, 5e-3, 1e-4),
        )

        # ---- visual guide ----
        guide_geom = new_geom(
            name="block_guide_vis",
            type="cylinder",
            size=[0.004, 0.20],
            rgba=[0.2, 0.6, 1.0, 0.55],
            contype="0",
            conaffinity="0",
            group=1,
        )
        guide_body = new_body(name="block_guide", pos=[0, 0, 0], mocap="true")
        guide_body.append(guide_geom)
        mujoco_arena.worldbody.append(guide_body)
        self._guide_body_name = "block_guide"

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.box)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="BoxSampler",
                mujoco_objects=self.box,
                x_range=BOX_INIT_X_RANGE,
                y_range=BOX_INIT_Y_RANGE,
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.02 + BOX_Z_OFFSET,
                rng=self.np_random,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.box,
        )

    # ------------------------------------------------------------------
    # References & observables
    # ------------------------------------------------------------------

    def _setup_references(self):
        super()._setup_references()
        self.box_body_id = self.sim.model.body_name2id(self.box.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def box_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.box_body_id])

            @sensor(modality=modality)
            def box_pos_xy(obs_cache):
                return np.array(self.sim.data.body_xpos[self.box_body_id])[:2]

            @sensor(modality=modality)
            def goal_pos_obs(obs_cache):
                return self.goal_pos.copy()

            @sensor(modality=modality)
            def box_to_goal(obs_cache):
                box_xy = np.array(self.sim.data.body_xpos[self.box_body_id])[:2]
                return self.goal_pos - box_xy

            sensors = [box_pos, box_pos_xy, goal_pos_obs, box_to_goal]
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors += [
                self._get_obj_eef_sensor(full_pf, "box_pos", f"{arm_pf}gripper_to_box", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]
            for s in sensors:
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _zero_joint_velocity(self, joint_name: str) -> None:
        jid = self.sim.model.joint_name2id(joint_name)
        if self.sim.model.jnt_type[jid] == 0:
            dofadr = self.sim.model.jnt_dofadr[jid]
            self.sim.data.qvel[dofadr : dofadr + 6] = 0.0

    def _update_guide(self) -> None:
        if not hasattr(self, "_guide_body_name"):
            return
        bx, by, bz = self.sim.data.body_xpos[self.box_body_id]
        self.sim.data.set_mocap_pos(
            self._guide_body_name, np.array([bx, by, self.table_top_z + 0.05])
        )

    def _stabilize_box_on_table(self) -> None:
        for _ in range(40):
            self.sim.step()
        joint = self.box.joints[0]
        qpos = np.array(self.sim.data.get_joint_qpos(joint), dtype=float)
        qpos[0] = np.clip(qpos[0], -self.table_xy_limit, self.table_xy_limit)
        qpos[1] = np.clip(qpos[1], -self.table_xy_limit, self.table_xy_limit)
        min_z = self.table_top_z + BOX_HALF + BOX_Z_OFFSET
        if qpos[2] < min_z:
            qpos[2] = min_z
        self.sim.data.set_joint_qpos(joint, qpos)
        self._zero_joint_velocity(joint)
        self.sim.forward()

    def _place_obstacles(self, configs: list[dict]) -> None:
        hz_default = float(OBS_HALF_Z)
        for i, body_name in enumerate(self._obstacle_body_names):
            if i < len(configs):
                cfg = configs[i]
                hx, hy, hz = cfg.get("half_size", [0.18, 0.18, hz_default])
                ox, oy = cfg["pos"]
                oz = self.table_top_z + hz
                yaw = float(cfg.get("euler", [0, 0, 0])[2])
                self.sim.data.set_mocap_pos(body_name, np.array([ox, oy, oz]))
                self.sim.data.set_mocap_quat(body_name, yaw_to_quat_wxyz(yaw))
            else:
                self.sim.data.set_mocap_pos(body_name, np.array([0.0, 0.0, self.table_top_z - 0.5]))
                self.sim.data.set_mocap_quat(body_name, np.array([1.0, 0.0, 0.0, 0.0]))
        self.sim.forward()

    def _verify_obstacles_in_sim(self, configs: list[dict]) -> None:
        for body_name, cfg in zip(self._obstacle_body_names, configs):
            bid = self.sim.model.body_name2id(body_name)
            if self.sim.data.body_xpos[bid][2] < self.table_top_z:
                raise RuntimeError(f"{body_name} below table surface")

    def _reset_internal(self):
        self._grip_active = False
        super()._reset_internal()

        if not self.deterministic_reset:
            placements = self.placement_initializer.sample()
            for obj_pos, _obj_quat, obj in placements.values():
                if self._box_init_y_range is not None and obj is self.box:
                    y_min, y_max = self._box_init_y_range
                    obj_pos = list(obj_pos)
                    obj_pos[1] = float(self.np_random.uniform(y_min, y_max))
                if self._box_init_x_range is not None and obj is self.box:
                    x_min, x_max = self._box_init_x_range
                    obj_pos = list(obj_pos)
                    if self._box_x_sampler is not None:
                        obj_pos[0] = float(self._box_x_sampler(self.np_random, x_min, x_max))
                    else:
                        obj_pos[0] = float(self.np_random.uniform(x_min, x_max))
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), [1.0, 0.0, 0.0, 0.0]]),
                )
            self._stabilize_box_on_table()

        if self.obstacle_configs is not None:
            configs = self.obstacle_configs
        else:
            configs = random_obstacle_configs(self.np_random, self.num_obstacles)
        self._place_obstacles(configs)
        self.sim.forward()
        self._verify_obstacles_in_sim(configs)
        self._update_guide()
        self._init_phase_state()

    # ------------------------------------------------------------------
    # Reward & termination
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_phase_horizons(horizon: int) -> list[int]:
        base = max(horizon // NUM_PHASES, 1)
        return [base, base, max(horizon - 2 * base, 1)]

    def step(self, action):
        """Keep box rigidly attached to EEF via position override.

        Two-stage: (1) pre-step positioning so the simulation starts with
        the box already attached, (2) post-step correction for any drift.
        """
        if not self._grip_active:
            return super().step(action)

        # EEF site
        eef_sid = self.robots[0].eef_site_id
        if isinstance(eef_sid, dict):
            eef_sid = eef_sid.get("right", next(iter(eef_sid.values())))
        box_joint = self.box.joints[0]
        jid = self.sim.model.joint_name2id(box_joint)
        dofadr = self.sim.model.jnt_dofadr[jid]

        # ---- Pre-step: box at EEF pos + match EEF velocity ----
        eef_pre = self.sim.data.site_xpos[eef_sid].copy()
        eef_body = self.sim.model.site_bodyid[eef_sid]
        eef_vel = self.sim.data.cvel[eef_body][3:6].copy()
        # offset: box centre = EEF - (0, 0, 0.012)
        offset = np.array([0.0, 0.0, -0.012], dtype=np.float32)
        box_target = eef_pre + offset
        self.sim.data.set_joint_qpos(
            box_joint,
            np.concatenate([box_target, [1.0, 0.0, 0.0, 0.0]]),
        )
        self.sim.data.qvel[dofadr : dofadr + 3] = eef_vel
        self.sim.data.qvel[dofadr + 3 : dofadr + 6] = 0.0
        self.sim.forward()

        # ---- Physics step ----
        obs, reward, done, info = super().step(action)

        # ---- Post-step: correct residual drift ----
        eef_post = self.sim.data.site_xpos[eef_sid]
        box_corrected = eef_post + offset
        self.sim.data.set_joint_qpos(
            box_joint,
            np.concatenate([box_corrected, [1.0, 0.0, 0.0, 0.0]]),
        )
        self.sim.data.qvel[dofadr : dofadr + 6] = 0.0
        self.sim.forward()

        # Patch box observations
        if "box_pos_xy" in obs:
            obs["box_pos_xy"] = box_corrected[:2].copy()
        if "box_pos" in obs:
            obs["box_pos"] = box_corrected.copy()
        if "box_to_goal" in obs:
            obs["box_to_goal"] = self.goal_pos - box_corrected[:2]

        return obs, reward, done, info

    def _init_phase_state(self) -> None:
        self.current_phase = 0
        self.phase_step_start = [self.timestep, 0, 0]
        self.phase_done = [False, False, False]
        self.phase_success = [False, False, False]
        self.phase_steps = [0, 0, 0]
        self._phase_horizons = self._compute_phase_horizons(self.horizon)
        self._phase_transition_reward = 0.0
        self._consecutive_arm_collision = 0
        self._consecutive_oob = 0

    def _box_y(self) -> float:
        return float(self.sim.data.body_xpos[self.box_body_id][1])

    def _infer_phase_from_y(self, box_y: float) -> int:
        if box_y > PHASE_Y_CROSS:
            return 2
        if box_y > PHASE_Y_APPROACH:
            return 1
        return 0

    def _finalize_phase_duration(self, phase: int) -> None:
        if phase < 0 or phase >= NUM_PHASES:
            return
        start = self.phase_step_start[phase]
        if start > 0 or phase == 0:
            self.phase_steps[phase] = max(self.timestep - start, 0)

    def _mark_phase_success(self, phase: int) -> None:
        if self.phase_done[phase]:
            return
        self.phase_done[phase] = True
        self.phase_success[phase] = True
        self._finalize_phase_duration(phase)
        self._phase_transition_reward = 1.0

    def _mark_phase_failure(self, phase: int) -> None:
        if self.phase_done[phase]:
            return
        self.phase_done[phase] = True
        self.phase_success[phase] = False
        self._finalize_phase_duration(phase)

    def _update_phase(self, box_y: float) -> None:
        inferred = self._infer_phase_from_y(box_y)
        if inferred <= self.current_phase:
            return
        for phase in range(self.current_phase, inferred):
            self._mark_phase_success(phase)
        self.current_phase = inferred
        if not self.phase_done[inferred]:
            self.phase_step_start[inferred] = self.timestep

    def _phase_elapsed_steps(self, phase: int | None = None) -> int:
        phase = self.current_phase if phase is None else phase
        start = self.phase_step_start[phase]
        return max(self.timestep - start, 0)

    def _check_phase_timeout(self, phase: int | None = None) -> bool:
        phase = self.current_phase if phase is None else phase
        if self.phase_done[phase]:
            return False
        return self._phase_elapsed_steps(phase) >= self._phase_horizons[phase]

    def _check_phase_success(self, phase: int) -> bool:
        box_y = self._box_y()
        if phase == 0:
            return box_y > PHASE_Y_APPROACH
        if phase == 1:
            return box_y > PHASE_Y_CROSS
        if phase == 2:
            return self._check_success()
        return False

    def _phase_shaping_reward(self) -> float:
        if not self.reward_shaping:
            return 0.0
        box_y = self._box_y()
        if self.current_phase == 0:
            return (box_y - PHASE_Y_APPROACH) * 2.0
        if self.current_phase == 1:
            return (box_y - PHASE_Y_CROSS) * 2.0
        box_xy = np.array(self.sim.data.body_xpos[self.box_body_id])[:2]
        return -float(np.linalg.norm(self.goal_pos - box_xy)) * 0.5

    def _build_phase_info(self) -> dict:
        active_steps = list(self.phase_steps)
        if not self.phase_done[self.current_phase]:
            active_steps[self.current_phase] = self._phase_elapsed_steps()
        return {
            "current_phase": int(self.current_phase),
            "current_phase_name": PHASE_NAMES[self.current_phase],
            "phase_steps": np.array(active_steps, dtype=np.int32),
            "phase_success": np.array(self.phase_success, dtype=bool),
            "phase_done": np.array(self.phase_done, dtype=bool),
            "all_phases_success": bool(all(self.phase_success)),
        }

    def reward(self, action=None):
        arm_col = self._check_arm_collision()
        oob = self._check_out_of_bounds()
        box_fell = self._check_box_fell()
        phase_timeout = self._check_phase_timeout()
        success = self._check_success()
        if arm_col or oob or box_fell or phase_timeout:
            r = -1.0
        elif success and self.current_phase == 2:
            r = 1.0
        elif self._phase_transition_reward > 0:
            r = self._phase_transition_reward
            self._phase_transition_reward = 0.0
        else:
            r = self._phase_shaping_reward()
        return r * (self.reward_scale if self.reward_scale is not None else 1.0)

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        box_y = self._box_y()
        self._update_phase(box_y)

        success = self._check_success()
        arm_col = self._check_arm_collision()
        oob = self._check_out_of_bounds()
        box_fell = self._check_box_fell()

        if arm_col:
            self._consecutive_arm_collision += 1
        else:
            self._consecutive_arm_collision = 0
        if oob:
            self._consecutive_oob += 1
        else:
            self._consecutive_oob = 0
        persistent_arm_col = self._consecutive_arm_collision >= self._grace_steps
        persistent_oob = self._consecutive_oob >= self._grace_steps

        self._update_guide()

        # Check phase failure AFTER persistent flags are computed, so
        # the done condition stays consistent with the info dict.
        # Use persistent checks for arm_col/oob, raw for box_fell/phase_timeout.
        phase_failed = False
        phase_fail_reason = ""
        if persistent_arm_col:
            phase_failed, phase_fail_reason = True, "arm_collision"
        elif persistent_oob:
            phase_failed, phase_fail_reason = True, "out_of_bounds"
        elif box_fell:
            phase_failed, phase_fail_reason = True, "box_fell"
        elif self._check_phase_timeout():
            phase_failed, phase_fail_reason = True, "phase_timeout"

        if phase_failed:
            self._mark_phase_failure(self.current_phase)

        if success and self.current_phase == 2:
            self._mark_phase_success(2)
            self.done = True
            done = True
        elif persistent_arm_col or persistent_oob or box_fell or phase_failed:
            self.done = True
            done = True

        info["success"] = success and all(self.phase_success)
        info["arm_collision"] = persistent_arm_col
        info["out_of_bounds"] = persistent_oob
        info["box_fell"] = box_fell
        info["phase_timeout"] = phase_failed and phase_fail_reason == "phase_timeout"
        info["phase_fail_reason"] = phase_fail_reason if phase_failed else ""
        info.update(self._build_phase_info())

        return reward, done, info

    # ------------------------------------------------------------------
    # Terminal condition helpers
    # ------------------------------------------------------------------

    def _check_success(self) -> bool:
        box_xy = np.array(self.sim.data.body_xpos[self.box_body_id])[:2]
        return bool(np.linalg.norm(self.goal_pos - box_xy) < self.goal_radius)

    def _check_arm_collision(self) -> bool:
        if not self._obstacle_geom_names:
            return False
        robot = self.robots[0]
        if self.check_contact(robot.robot_model, self._obstacle_geom_names):
            return True
        for arm in robot.arms:
            if robot.has_gripper[arm]:
                if self.check_contact(robot.gripper[arm], self._obstacle_geom_names):
                    return True
        return False

    def _check_out_of_bounds(self) -> bool:
        eef_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]]
        return bool(
            abs(eef_pos[0]) > self.boundary_half_size or abs(eef_pos[1]) > self.boundary_half_size
        )

    def _check_box_fell(self) -> bool:
        pos = np.array(self.sim.data.body_xpos[self.box_body_id])
        if pos[2] < self.table_top_z + BOX_HALF * 0.2:
            return True
        return bool(
            abs(pos[0]) > self.table_xy_limit + 0.01 or abs(pos[1]) > self.table_xy_limit + 0.01
        )

    def init_with_grip(
        self, verbose: bool = False, record_fn: callable | None = None
    ) -> tuple[dict, int, list | None]:
        """Reset environment, grip the box, and restart the step counter.

        Grip uses a MuJoCo weld equality constraint: after the EEF moves
        to the box, the constraint is activated so the box rigidly follows
        the EEF during every sim.step().

        Grip steps are NOT counted toward the episode horizon.
        """
        self._grip_active = False
        obs = self.reset()
        frames: list | None = [] if record_fn is not None else None
        if frames is not None and record_fn is not None:
            frames.append(record_fn(self))

        box_xy = obs["box_pos_xy"][:2].astype(np.float32)
        if verbose:
            print(
                f"    [init_with_grip] box_xy=({box_xy[0]:.4f}, {box_xy[1]:.4f}), "
                f"table_z={self.table_top_z:.4f}"
            )
        obs, grip_steps = grip_box(self, obs, verbose=verbose, record_fn=record_fn, frames=frames)
        self.done = False
        self.timestep = 0
        self._init_phase_state()
        if frames is not None and record_fn is not None:
            frames.append(record_fn(self))
        if verbose:
            box_final = obs.get("box_pos_xy", np.zeros(2))[:2]
            eef_final = obs.get("robot0_eef_pos", np.zeros(3))[:3]
            print(
                f"    [init_with_grip] done: box_final=({box_final[0]:.4f}, {box_final[1]:.4f}), "
                f"eef=({eef_final[0]:.4f}, {eef_final[1]:.4f}, {eef_final[2]:.4f}), "
                f"grip_active={self._grip_active}"
            )
        return obs, grip_steps, frames


# ── Auto-grip helper ───────────────────────────────────────────────────────


def grip_box(
    env,
    obs: dict,
    max_steps: int = 200,
    verbose: bool = False,
    record_fn: callable | None = None,
    frames: list | None = None,
) -> tuple[dict, int]:
    """Move EEF above the box, descend, activate MuJoCo weld constraint.

    Two phases: (1) move to safe height above box, (2) descend to grip
    height, (3) activate weld + visual close.
    """
    import numpy as np

    fingertip_offset = 0.012
    box_xy = obs["box_pos_xy"][:2].astype(np.float32)
    box_z = env.table_top_z + BOX_HALF + BOX_Z_OFFSET
    target_eef_z = box_z + fingertip_offset
    safe_z = env.table_top_z + 0.18

    # Raise approach height if box is in the narrow obstacle corridor.
    if env.obstacle_configs:
        obs_bounds = []
        for cfg in env.obstacle_configs:
            ox, oy = float(cfg["pos"][0]), float(cfg["pos"][1])
            hx = float(cfg["half_size"][0])
            hy = float(cfg["half_size"][1])
            hz = float(cfg["half_size"][2]) if len(cfg["half_size"]) > 2 else OBS_HALF_Z
            obs_bounds.append((ox - hx, ox + hx, oy - hy, oy + hy, env.table_top_z + hz))
        if len(obs_bounds) >= 2:
            s = sorted(obs_bounds, key=lambda b: b[0])
            if s[0][1] <= float(box_xy[0]) <= s[1][0]:
                safe_z = max(b[4] for b in obs_bounds) + 0.20

    # Open gripper
    open_act = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(10):
        _safe_step(env, open_act)

    step = 0
    total_steps = 10

    def _capture():
        if frames is not None and record_fn is not None:
            frames.append(record_fn(env))

    _capture()

    # Phase 1: move above box at safe height
    target_above = np.array([box_xy[0], box_xy[1], safe_z], dtype=np.float32)
    for _ in range(max_steps):
        eef = obs["robot0_eef_pos"].astype(np.float32)
        if np.linalg.norm(target_above - eef) < 0.01:
            if verbose:
                print(f"    [grip_box] above box step={step}")
            _capture()
            break
        obs = _move_toward(env, eef, target_above)
        step += 1

    # Phase 2: descend to grip height
    target_down = np.array([box_xy[0], box_xy[1], target_eef_z], dtype=np.float32)
    for _ in range(max_steps):
        eef = obs["robot0_eef_pos"].astype(np.float32)
        if np.linalg.norm(target_down - eef) < 0.003:
            if verbose:
                print(f"    [grip_box] at grip   step={step}")
            _capture()
            break
        obs = _move_toward(env, eef, target_down)
        step += 1

    # Phase 3: activate grip
    env._grip_active = True

    # Close gripper visually
    close_act = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    for _ in range(10):
        _safe_step(env, close_act)
    step += 10

    if verbose:
        eef_f = obs["robot0_eef_pos"].astype(np.float32)
        box_f = obs.get("box_pos_xy", np.zeros(2))[:2]
        print(
            f"    [grip_box] weld done step={step}  "
            f"box=({box_f[0]:.4f},{box_f[1]:.4f})  eef=({eef_f[0]:.4f},{eef_f[1]:.4f})"
        )

    return obs, total_steps + step


def _move_toward(env, current: np.ndarray, target: np.ndarray, speed: float = 0.08) -> dict:
    """Take one step toward target at fixed speed."""
    diff = target - current
    dist = float(np.linalg.norm(diff))
    if dist < 1e-6:
        delta = np.zeros(3, dtype=np.float32)
    else:
        delta = (diff / dist * min(speed, dist)).astype(np.float32)
    act = np.array([delta[0], delta[1], delta[2], 0, 0, 0, -1], dtype=np.float32)
    return _safe_step(env, act)


def _safe_step(env, action: np.ndarray) -> dict:
    """env.step() that ignores premature termination during setup."""
    try:
        obs, reward, done, info = env.step(action)
    except ValueError:
        env.done = False
        obs, reward, done, info = env.step(action)
    if env.done:
        env.done = False
    return obs
