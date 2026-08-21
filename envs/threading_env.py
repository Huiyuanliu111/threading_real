"""MimicGen Threading task ported to the robosuite 1.5 ManipulationEnv API.

Task geometry and reset distributions are adapted from NVlabs MimicGen v1.0.0,
copyright NVIDIA CORPORATION & AFFILIATES and used under its source license.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CompositeObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, add_to_dict
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
import robosuite.utils.transform_utils as T


def needle_center_reaches_ring(needle_center, ring_center, ring_radius):
    """Historical MimicGen success rule requiring a deep needle insertion."""
    return bool(
        np.linalg.norm(np.asarray(needle_center) - np.asarray(ring_center))
        < float(ring_radius)
    )


class NeedleObject(CompositeObject):
    def __init__(self, name: str):
        self._name = name
        self.needle_mat_name = "threading_darkwood_mat"
        self._important_sites = {}
        super().__init__(**self._get_geom_attrs())
        self.append_material(
            CustomMaterial(
                texture="WoodDark",
                tex_name="threading_darkwood",
                mat_name=self.needle_mat_name,
                tex_attrib={"type": "cube"},
                mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
            )
        )

    def _get_geom_attrs(self):
        needle_size = (0.005, 0.06, 0.005)
        handle_size = (0.02, 0.02, 0.02)
        args = {}
        add_to_dict(
            dic=args,
            geom_types="box",
            geom_locations=(handle_size[0] - needle_size[0], 0.0, handle_size[2] - needle_size[2]),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=needle_size,
            geom_names="needle",
            geom_rgbas=None,
            geom_materials=self.needle_mat_name,
            geom_frictions=(0.3, 5e-3, 1e-4),
        )
        add_to_dict(
            dic=args,
            geom_types="box",
            geom_locations=(0.0, 2.0 * needle_size[1], 0.0),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=handle_size,
            geom_names="handle",
            geom_rgbas=None,
            geom_materials=self.needle_mat_name,
            geom_frictions=None,
        )
        args.update(
            total_size=(0.02, 0.08, 0.02),
            name=self.name,
            locations_relative_to_center=False,
            obj_types="all",
            density=100.0,
        )
        return args


class RingTripodObject(CompositeObject):
    def __init__(self, name: str):
        self._name = name
        self.tripod_mat_name = "threading_lightwood_mat"
        self._important_sites = {}
        super().__init__(**self._get_geom_attrs())
        self.append_material(
            CustomMaterial(
                texture="WoodLight",
                tex_name="threading_lightwood",
                mat_name=self.tripod_mat_name,
                tex_attrib={"type": "cube"},
                mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
            )
        )

    def _get_geom_attrs(self):
        total_size = (0.05, 0.05, 0.1)
        args = {}
        unit_size = (0.005, 0.002, 0.002)
        pattern = np.ones((6, 1, 6))
        pattern[1:5, 0, 1:5] = 0
        self.ring_size = np.array(
            [unit_size[0] * pattern.shape[1], unit_size[1] * pattern.shape[2], unit_size[2] * pattern.shape[0]]
        )
        ring_offset = (
            total_size[0] - self.ring_size[0],
            total_size[1] - self.ring_size[1],
            2.0 * (total_size[2] - self.ring_size[2]),
        )
        self.num_ring_geoms = 0
        for z, x, y in np.argwhere(pattern > 0):
            add_to_dict(
                dic=args,
                geom_types="box",
                geom_locations=(
                    x * 2.0 * unit_size[0] + ring_offset[0],
                    y * 2.0 * unit_size[1] + ring_offset[1],
                    z * 2.0 * unit_size[2] + ring_offset[2],
                ),
                geom_quats=(1, 0, 0, 0),
                geom_sizes=unit_size,
                geom_names=f"ring_{self.num_ring_geoms}",
                geom_rgbas=None,
                geom_materials=self.tripod_mat_name,
                geom_frictions=(0.3, 5e-3, 1e-4),
            )
            self.num_ring_geoms += 1

        radius, half_height = 0.01, 0.03
        leg_locations = [
            (0.0, 0.0, 0.0),
            (0.0, 2.0 * total_size[1] - 2.0 * radius, 0.0),
            (2.0 * total_size[0] - 2.0 * radius, total_size[1] - radius, 0.0),
        ]
        center = np.array([total_size[0], total_size[1], 0.0])
        for index, location in enumerate(leg_locations):
            capsule = np.array(location) + np.array([radius, radius, 0.0])
            capsule[2] = 0
            direction = center - capsule
            direction /= np.linalg.norm(direction)
            axis = np.cross(direction, np.array([0.0, 0.0, 1.0]))
            quat = T.convert_quat(T.mat2quat(T.rotation_matrix(-np.pi / 6.0, axis)), to="wxyz")
            add_to_dict(
                dic=args,
                geom_types="capsule",
                geom_locations=location,
                geom_quats=quat,
                geom_sizes=(radius, half_height),
                geom_names=f"tripod_{index}",
                geom_rgbas=None,
                geom_materials=self.tripod_mat_name,
                geom_frictions=None,
            )

        base_thickness, post_size = 0.005, 0.005
        post_sizes = [
            (total_size[0], total_size[1], base_thickness),
            (post_size, post_size, total_size[2] - self.ring_size[2] - base_thickness - radius - half_height),
        ]
        post_locations = [
            (0.0, 0.0, 2.0 * (radius + half_height)),
            (total_size[0] - post_size, total_size[1] - post_size, 2.0 * (radius + half_height + base_thickness)),
        ]
        for index in range(2):
            add_to_dict(
                dic=args,
                geom_types="box",
                geom_locations=post_locations[index],
                geom_quats=(1, 0, 0, 0),
                geom_sizes=post_sizes[index],
                geom_names=f"post_{index}",
                geom_rgbas=None,
                geom_materials=self.tripod_mat_name,
                geom_frictions=None,
            )
        args.update(
            total_size=total_size,
            name=self.name,
            locations_relative_to_center=False,
            obj_types="all",
            density=100.0,
            solref=(0.02, 1.0),
            solimp=(0.9, 0.95, 0.001),
        )
        return args


class Threading(ManipulationEnv):
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
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        **unused_kwargs,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0.0, 0.0, 0.8))
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
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

    def reward(self, action=None):
        reward = float(self._check_success())
        return reward * self.reward_scale if self.reward_scale is not None else reward

    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])
        # D0's ring is fixed, so a close, face-on camera can resolve the
        # millimeter-scale clearance during the final insertion phase.
        arena.set_camera(
            camera_name="threading_closeup",
            pos=[0.0, -0.36, 1.04],
            quat=[0.7983486481, 0.6021955131, 0.0, 0.0],
            camera_attribs={"fovy": "35"},
        )
        self.needle = NeedleObject("needle_obj")
        self.tripod = RingTripodObject("tripod_obj")
        self._get_placement_initializer()
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.needle, self.tripod],
        )

    def _get_initial_placement_bounds(self):
        return {
            "needle": {"x": (-0.2, -0.05), "y": (0.15, 0.25), "z_rot": (-2*np.pi/3, -np.pi/3)},
            "tripod": {"x": (0.0, 0.0), "y": (-0.15, -0.15), "z_rot": (np.pi/2, np.pi/2)},
        }

    def _get_placement_initializer(self):
        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for label, obj, z_offset in (("Needle", self.needle, 0.0), ("Tripod", self.tripod, 0.001)):
            key = label.lower()
            self.placement_initializer.append_sampler(
                UniformRandomSampler(
                    name=f"{label}Sampler",
                    mujoco_objects=obj,
                    x_range=bounds[key]["x"],
                    y_range=bounds[key]["y"],
                    rotation=bounds[key]["z_rot"],
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    z_offset=z_offset,
                )
            )

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = {
            "needle": self.sim.model.body_name2id(self.needle.root_body),
            "tripod": self.sim.model.body_name2id(self.tripod.root_body),
        }

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            for pos, quat, obj in self.placement_initializer.sample().values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos, quat]))

    def _setup_observables(self) -> OrderedDict:
        observables = super()._setup_observables()
        if self.use_object_obs:
            for name in ("needle", "tripod"):
                @sensor(modality="object")
                def object_pos(obs_cache, object_name=name):
                    return np.array(self.sim.data.body_xpos[self.obj_body_id[object_name]])

                @sensor(modality="object")
                def object_quat(obs_cache, object_name=name):
                    return T.convert_quat(
                        self.sim.data.body_xquat[self.obj_body_id[object_name]], to="xyzw"
                    )

                for suffix, observable in (("pos", object_pos), ("quat", object_quat)):
                    obs_name = f"{name}_{suffix}"
                    observable.__name__ = obs_name
                    observables[obs_name] = Observable(
                        name=obs_name, sensor=observable, sampling_rate=self.control_freq
                    )
        return observables

    def _check_success(self):
        needle_center = np.array(
            self.sim.data.geom_xpos[self.sim.model.geom_name2id("needle_obj_needle")]
        )
        ring_center = np.mean(
            [
                self.sim.data.geom_xpos[
                    self.sim.model.geom_name2id(f"tripod_obj_ring_{index}")
                ]
                for index in range(self.tripod.num_ring_geoms)
            ],
            axis=0,
        )
        # Match the original MimicGen criterion: the needle-bar center, rather
        # than only its free tip, must reach the ring opening. Since the bar has
        # a 6 cm half-length, this requires roughly half the needle to pass
        # through the ring before the task is counted as successful.
        return needle_center_reaches_ring(
            needle_center,
            ring_center,
            self.tripod.ring_size[1],
        )


class Threading_D0(Threading):
    pass


class Threading_D1(Threading_D0):
    def _get_initial_placement_bounds(self):
        return {
            "needle": {"x": (-0.2, 0.05), "y": (0.15, 0.25), "z_rot": (-7*np.pi/6, np.pi/6)},
            "tripod": {"x": (-0.1, 0.15), "y": (-0.2, -0.1), "z_rot": (np.pi/6, 5*np.pi/6)},
        }


class Threading_D2(Threading_D1):
    def _get_initial_placement_bounds(self):
        return {
            "needle": {"x": (-0.2, 0.05), "y": (-0.25, -0.15), "z_rot": (-7*np.pi/6, np.pi/6)},
            "tripod": {"x": (-0.1, 0.15), "y": (0.1, 0.2), "z_rot": (-5*np.pi/6, -np.pi/6)},
        }
