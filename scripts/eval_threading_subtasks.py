#!/usr/bin/env python3
"""Compare action chunk sizes on decoupled MimicGen Threading subtasks.

Each subtask starts from a real simulator state in an expert demonstration:

1. approach: begin at the start of the demonstration and reach the needle handle;
2. pick: begin near the handle and grasp + lift the needle;
3. insert: begin after the needle has been lifted and insert its tip into the ring.

Using demonstration states is important for the latter two stages because it
preserves the robot pose, object pose, velocity, and grasp contacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pushbox.diffusion_policy.common.sampler import get_val_mask
from chunk_selector.chunk_dataset import CoarseChunkFeatureCollector, parse_subtask_chunks
from chunk_selector.execution import PREDICTION_MODES
from threading_task.dataset import sorted_demo_keys
from threading_task.env import (
    agent_state_from_obs,
    create_threading_env,
    image_from_obs,
    seed_env,
    success_from_env,
)
from threading_task.metrics import EpisodeMetrics, aggregate_episodes
from threading_task.visualization import VideoWriter, compose_camera_views
from scripts.eval_policy import load_policy


SUBTASKS = ("approach", "pick", "insert")
SUBTASK_LABELS = {
    "approach": "1. approach object",
    "pick": "2. pick object",
    "insert": "3. move and insert",
}
SUBTASK_DIRS = {
    "approach": "subtask1",
    "pick": "subtask2",
    "insert": "subtask3",
}
SUBTASK_IDS = {"approach": "1", "pick": "2", "insert": "3"}
SUBTASK_ALIASES = {"1": "approach", "2": "pick", "3": "insert"}
DEFAULT_MAX_STEPS = {"approach": 100, "pick": 120, "insert": 300}


@dataclass
class StageStart:
    demo: str
    demo_index: int
    subtask: str
    start_index: int
    previous_state: np.ndarray
    state: np.ndarray
    initial_needle_z: float
    handle_distance: float
    lift_height: float
    grasp_index: int | None = None
    orientation_error_deg: float | None = None
    horizontal_alignment_error: float | None = None
    pregrasp_height: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "demo": self.demo,
            "demo_index": self.demo_index,
            "subtask": self.subtask,
            "start_index": self.start_index,
            "initial_needle_z": self.initial_needle_z,
            "handle_distance": self.handle_distance,
            "lift_height": self.lift_height,
            "grasp_index": self.grasp_index,
            "orientation_error_deg": self.orientation_error_deg,
            "horizontal_alignment_error": self.horizontal_alignment_error,
            "pregrasp_height": self.pregrasp_height,
        }


@dataclass
class SubtaskEpisode:
    chunk_size: int
    subtask: str
    demo: str
    demo_index: int
    start_index: int
    episode: int
    seed: int
    success: bool
    steps: int
    timeout: bool
    done: bool
    num_policy_calls: int
    wall_time_sec: float
    control_frequency_hz: float
    execution_time_sec: float
    computer_time_sec: float
    estimated_total_time_sec: float
    video_overhead_sec: float
    inference_time_mean_ms: float
    inference_time_p95_ms: float
    final_handle_distance: float
    final_lift_height: float
    final_grasped: bool
    final_insert_distance: float
    video: str | None

    def standard_metrics(self) -> EpisodeMetrics:
        return EpisodeMetrics(
            episode=self.episode,
            seed=self.seed,
            success=self.success,
            steps=self.steps,
            episode_return=0.0,
            max_reward=0.0,
            timeout=self.timeout,
            num_policy_calls=self.num_policy_calls,
            wall_time_sec=self.wall_time_sec,
            inference_time_mean_ms=self.inference_time_mean_ms,
            inference_time_p95_ms=self.inference_time_p95_ms,
            video=self.video,
        )


def _quat_rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by xyzw quaternion(s), without an optional scipy dependency."""
    quaternion = np.asarray(quaternion)
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    vectors = np.broadcast_to(np.asarray(vector), xyz.shape)
    return vectors + 2.0 * np.cross(xyz, np.cross(xyz, vectors) + w * vectors)


def _needle_metrics_from_arrays(
    object_obs: np.ndarray,
    eef_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return needle root height, handle position, and EEF-to-handle distance."""
    if object_obs.ndim != 2 or object_obs.shape[1] < 7:
        raise ValueError(f"Expected object observations with at least 7 values, got {object_obs.shape}")
    needle_position = object_obs[:, :3]
    needle_quaternion = object_obs[:, 3:7]
    # NeedleObject places the graspable handle +6 cm along its local y axis.
    handle_position = needle_position + _quat_rotate_xyzw(
        needle_quaternion, np.array([0.0, 0.06, 0.0])
    )
    handle_distance = np.linalg.norm(eef_positions - handle_position, axis=1)
    return needle_position[:, 2], handle_position, handle_distance


def _resolve_demo_indices(
    total: int,
    episodes: int,
    split: str,
    val_ratio: float,
    split_seed: int,
    selection_seed: int,
    explicit: list[int] | None,
) -> list[int]:
    if explicit:
        invalid = [index for index in explicit if index < 0 or index >= total]
        if invalid:
            raise ValueError(f"Demo indices outside [0, {total - 1}]: {invalid}")
        pool = np.asarray(list(dict.fromkeys(explicit)), dtype=np.int64)
    else:
        val_mask = get_val_mask(total, val_ratio, split_seed)
        if split == "validation":
            pool = np.flatnonzero(val_mask)
        elif split == "train":
            pool = np.flatnonzero(~val_mask)
        else:
            pool = np.arange(total)
        pool = np.random.default_rng(selection_seed).permutation(pool)
    if episodes > len(pool):
        raise ValueError(
            f"Requested {episodes} episodes, but split={split!r} only provides {len(pool)} demos"
        )
    return [int(index) for index in pool[:episodes]]


def _quaternion_angle_rad(first: np.ndarray, second: np.ndarray) -> float:
    """Return the shortest rotation angle between two equivalent quaternions."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = np.clip(abs(float(np.dot(first, second))), 0.0, 1.0)
    return float(2.0 * np.arccos(cosine))


def _quaternion_rotation_vector_xyzw(
    target: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    """Return the world-frame rotation vector taking current to target."""
    target = np.asarray(target, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    target /= np.linalg.norm(target)
    current /= np.linalg.norm(current)
    current_inverse = np.concatenate((-current[:3], current[3:]))
    target_xyz, target_w = target[:3], target[3]
    inverse_xyz, inverse_w = current_inverse[:3], current_inverse[3]
    error = np.concatenate(
        (
            target_w * inverse_xyz
            + inverse_w * target_xyz
            + np.cross(target_xyz, inverse_xyz),
            [target_w * inverse_w - np.dot(target_xyz, inverse_xyz)],
        )
    )
    if error[3] < 0.0:
        error = -error
    vector_norm = np.linalg.norm(error[:3])
    if vector_norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(error[3], -1.0, 1.0))
    return error[:3] * (angle / vector_norm)


def _needle_is_grasped(env) -> bool:
    raw_env = env.env
    try:
        return bool(raw_env._check_grasp(raw_env.robots[0].gripper, raw_env.needle))
    except (AttributeError, TypeError):
        return bool(
            raw_env._check_grasp(
                raw_env.robots[0].gripper,
                raw_env.needle.contact_geoms,
            )
        )


def _synthesize_aligned_pregrasp(
    env,
    states: np.ndarray,
    eef_positions: np.ndarray,
    eef_quaternions: np.ndarray,
    approach_index: int,
    *,
    pregrasp_height: float = 0.085,
    stable_grasp_frames: int = 3,
    max_anchor_orientation_error_deg: float = 20.0,
    position_tolerance: float = 0.001,
    orientation_tolerance_deg: float = 2.0,
    max_control_steps: int = 80,
) -> tuple[int, int, np.ndarray, np.ndarray, dict[str, float]]:
    """Build an open-gripper state directly above the expert grasp pose.

    The target shares the expert grasp EEF x/y position and orientation, but is
    lifted vertically by ``pregrasp_height``. The simulator controller then
    moves a nearby contact-free demonstration state to that exact target.
    """
    grasp_flags = np.zeros(len(states), dtype=bool)
    run_length = 0
    grasp_index = None
    for index in range(approach_index, len(states)):
        env.reset_to({"states": states[index]})
        grasp_flags[index] = _needle_is_grasped(env)
        run_length = run_length + 1 if grasp_flags[index] else 0
        if run_length >= stable_grasp_frames:
            grasp_index = index - stable_grasp_frames + 1
            break
    if grasp_index is None:
        raise RuntimeError(
            "expert trajectory never reaches a stable needle grasp "
            f"of {stable_grasp_frames} consecutive frames"
        )

    grasp_position = np.asarray(eef_positions[grasp_index], dtype=np.float64)
    grasp_quaternion = eef_quaternions[grasp_index]
    target_position = grasp_position + np.array([0.0, 0.0, pregrasp_height])
    max_anchor_orientation_error = np.deg2rad(max_anchor_orientation_error_deg)
    candidates = [
        index
        for index in range(grasp_index)
        if (
            not grasp_flags[index]
            and _quaternion_angle_rad(
                eef_quaternions[index],
                grasp_quaternion,
            )
            <= max_anchor_orientation_error
        )
    ]
    if not candidates:
        raise RuntimeError(
            "expert trajectory has no contact-free state with a compatible "
            "pre-grasp wrist orientation"
        )
    anchor_index = min(
        candidates,
        key=lambda index: np.linalg.norm(eef_positions[index] - target_position),
    )

    obs = env.reset_to({"states": states[anchor_index]})
    previous_state = states[max(0, anchor_index - 1)].copy()
    current_state = states[anchor_index].copy()
    initial_needle_position = np.asarray(obs["needle_pos"], dtype=np.float64)
    orientation_tolerance = np.deg2rad(orientation_tolerance_deg)

    for _ in range(max_control_steps):
        current_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        current_quaternion = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        position_error = target_position - current_position
        rotation_vector = _quaternion_rotation_vector_xyzw(
            grasp_quaternion,
            current_quaternion,
        )
        if (
            np.linalg.norm(position_error) <= position_tolerance
            and np.linalg.norm(rotation_vector) <= orientation_tolerance
        ):
            break

        action = np.zeros(7, dtype=np.float32)
        # Threading uses OSC_POSE with +/-5 cm translation and +/-0.5 rad
        # rotation output ranges for normalized actions.
        action[:3] = np.clip(position_error / 0.05, -1.0, 1.0)
        action[3:6] = np.clip(rotation_vector / 0.5, -1.0, 1.0)
        action[6] = -1.0
        previous_state = current_state
        obs, _, done, _ = env.step(action)
        current_state = env.env.sim.get_state().flatten().copy()
        if done:
            raise RuntimeError("environment terminated while constructing pre-grasp state")
    else:
        raise RuntimeError(
            "failed to converge to aligned pre-grasp target within "
            f"{max_control_steps} controller steps"
        )

    current_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    current_quaternion = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    needle_displacement = np.linalg.norm(
        np.asarray(obs["needle_pos"], dtype=np.float64) - initial_needle_position
    )
    if needle_displacement > 0.002 or _needle_is_grasped(env):
        raise RuntimeError(
            "needle moved or was grasped while constructing the contact-free "
            f"pre-grasp state (displacement={needle_displacement:.4f} m)"
        )

    metrics = {
        "orientation_error_deg": float(
            np.rad2deg(_quaternion_angle_rad(current_quaternion, grasp_quaternion))
        ),
        "horizontal_alignment_error": float(
            np.linalg.norm(current_position[:2] - grasp_position[:2])
        ),
        "pregrasp_height": float(current_position[2] - grasp_position[2]),
    }
    return (
        anchor_index,
        grasp_index,
        previous_state,
        current_state,
        metrics,
    )


def build_stage_starts(
    dataset_path: str | Path,
    tasks: tuple[str, ...],
    episodes: int,
    split: str,
    val_ratio: float,
    split_seed: int,
    selection_seed: int,
    demo_indices: list[int] | None,
    approach_distance: float,
    lift_height: float,
    pick_pregrasp_height: float = 0.085,
) -> dict[str, list[StageStart]]:
    """Extract paired expert start states for every requested subtask."""
    result = {task: [] for task in tasks}
    replay_env = None
    try:
        if "pick" in tasks:
            replay_env, _ = create_threading_env(dataset_path)
            replay_env.reset()
        with h5py.File(Path(dataset_path).expanduser(), "r") as h5:
            if "data" not in h5:
                raise KeyError(f"{dataset_path}: missing /data")
            demos = sorted_demo_keys(h5["data"])
            selected = _resolve_demo_indices(
                len(demos),
                episodes,
                split,
                val_ratio,
                split_seed,
                selection_seed,
                demo_indices,
            )
            for demo_index in selected:
                demo_name = demos[demo_index]
                demo = h5["data"][demo_name]
                states = np.asarray(demo["states"])
                obs = demo["obs"]
                object_obs = np.asarray(obs["object"])
                eef_positions = np.asarray(obs["robot0_eef_pos"])
                eef_quaternions = np.asarray(obs["robot0_eef_quat"])
                needle_z, _, handle_distances = _needle_metrics_from_arrays(
                    object_obs, eef_positions
                )
                if len(states) != len(needle_z):
                    raise ValueError(
                        f"{demo_name}: states ({len(states)}) and observations ({len(needle_z)}) differ"
                    )
                initial_z = float(needle_z[0])
                approach_candidates = np.flatnonzero(
                    handle_distances <= approach_distance
                )
                insert_candidates = np.flatnonzero(
                    needle_z - initial_z >= lift_height
                )
                for task in tasks:
                    grasp_index = None
                    orientation_error_deg = None
                    horizontal_alignment_error = None
                    actual_pregrasp_height = None
                    previous_state = None
                    state = None
                    if task == "approach":
                        start_index = 0
                    elif task == "pick":
                        if not len(approach_candidates):
                            raise RuntimeError(
                                f"{demo_name}: no approach boundary satisfying "
                                f"handle distance <= {approach_distance:.3f} m"
                            )
                        if replay_env is None:
                            raise RuntimeError(
                                "pick boundary replay environment is unavailable"
                            )
                        try:
                            (
                                start_index,
                                grasp_index,
                                previous_state,
                                state,
                                pregrasp_metrics,
                            ) = _synthesize_aligned_pregrasp(
                                replay_env,
                                states,
                                eef_positions,
                                eef_quaternions,
                                int(approach_candidates[0]),
                                pregrasp_height=pick_pregrasp_height,
                            )
                            orientation_error_deg = pregrasp_metrics[
                                "orientation_error_deg"
                            ]
                            horizontal_alignment_error = pregrasp_metrics[
                                "horizontal_alignment_error"
                            ]
                            actual_pregrasp_height = pregrasp_metrics[
                                "pregrasp_height"
                            ]
                        except RuntimeError as error:
                            raise RuntimeError(f"{demo_name}: {error}") from error
                    else:
                        if not len(insert_candidates):
                            raise RuntimeError(
                                f"{demo_name}: no insert boundary satisfying "
                                f"lift height >= {lift_height:.3f} m"
                            )
                        start_index = int(insert_candidates[0])
                    previous_index = max(0, start_index - 1)
                    if previous_state is None:
                        previous_state = states[previous_index].copy()
                    if state is None:
                        state = states[start_index].copy()
                    replay_obs = (
                        replay_env.reset_to({"states": state})
                        if task == "pick" and replay_env is not None
                        else None
                    )
                    if replay_obs is not None:
                        replay_object = np.concatenate(
                            (replay_obs["needle_pos"], replay_obs["needle_quat"])
                        )[None]
                        replay_eef = np.asarray(replay_obs["robot0_eef_pos"])[None]
                        replay_needle_z, _, replay_handle_distance = (
                            _needle_metrics_from_arrays(replay_object, replay_eef)
                        )
                        start_needle_z = float(replay_needle_z[0])
                        start_handle_distance = float(replay_handle_distance[0])
                    else:
                        start_needle_z = float(needle_z[start_index])
                        start_handle_distance = float(handle_distances[start_index])
                    result[task].append(
                        StageStart(
                            demo=demo_name,
                            demo_index=demo_index,
                            subtask=task,
                            start_index=start_index,
                            previous_state=previous_state,
                            state=state,
                            initial_needle_z=initial_z,
                            handle_distance=start_handle_distance,
                            lift_height=start_needle_z - initial_z,
                            grasp_index=grasp_index,
                            orientation_error_deg=orientation_error_deg,
                            horizontal_alignment_error=horizontal_alignment_error,
                            pregrasp_height=actual_pregrasp_height,
                        )
                    )
    finally:
        if replay_env is not None:
            replay_env.close()
    return result


def _live_task_metrics(env, obs: dict[str, Any], initial_needle_z: float) -> dict[str, Any]:
    raw_env = env.env
    needle_position = np.asarray(obs["needle_pos"])
    needle_quaternion = np.asarray(obs["needle_quat"])
    handle_position = needle_position + _quat_rotate_xyzw(
        needle_quaternion, np.array([0.0, 0.06, 0.0])
    )
    eef_position = np.asarray(obs["robot0_eef_pos"])
    grasped = _needle_is_grasped(env)
    needle_geom_id = raw_env.sim.model.geom_name2id("needle_obj_needle")
    needle_center = np.asarray(raw_env.sim.data.geom_xpos[needle_geom_id])
    ring_center = np.mean(
        [
            raw_env.sim.data.geom_xpos[
                raw_env.sim.model.geom_name2id(f"tripod_obj_ring_{index}")
            ]
            for index in range(raw_env.tripod.num_ring_geoms)
        ],
        axis=0,
    )
    return {
        "handle_distance": float(np.linalg.norm(eef_position - handle_position)),
        "lift_height": float(needle_position[2] - initial_needle_z),
        "grasped": grasped,
        "insert_distance": float(np.linalg.norm(needle_center - ring_center)),
        "insert_threshold": float(raw_env.tripod.ring_size[1]),
        "inserted": success_from_env(env),
    }


def _task_success(
    task: str,
    metrics: dict[str, Any],
    approach_distance: float,
    lift_height: float,
) -> bool:
    if task == "approach":
        return bool(metrics["handle_distance"] <= approach_distance)
    if task == "pick":
        return bool(metrics["grasped"] and metrics["lift_height"] >= lift_height)
    return bool(metrics["inserted"])


def _policy_observation(
    observations: list[dict[str, Any]],
    device: torch.device,
    state_mode: str,
    camera_keys: tuple[str, ...],
    camera_output_keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    states = [agent_state_from_obs(obs, state_mode=state_mode) for obs in observations]
    result = {"agent_pos": torch.from_numpy(np.stack(states))[None].to(device)}
    for output_key, camera_key in zip(camera_output_keys, camera_keys):
        images = [image_from_obs(obs, camera_key) for obs in observations]
        result[output_key] = torch.from_numpy(np.stack(images))[None].to(device)
    return result


def _should_keep_video(mode: str, success: bool) -> bool:
    return (
        mode == "all"
        or (mode == "successes" and success)
        or (mode == "failures" and not success)
    )


def run_episode(
    env,
    policy,
    start: StageStart,
    chunk_size: int,
    episode_index: int,
    seed: int,
    max_steps: int,
    approach_distance: float,
    lift_height: float,
    state_mode: str,
    camera_keys: tuple[str, ...],
    camera_output_keys: tuple[str, ...],
    output_dir: Path,
    save_video: bool,
    video_fps: int,
    video_size: int,
    control_frequency_hz: float,
    translation_scale: float,
    feature_collector: CoarseChunkFeatureCollector | None = None,
) -> SubtaskEpisode:
    device = next(policy.parameters()).device
    n_obs_steps = int(policy.n_obs_steps)
    seed_env(env, seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    env.reset()
    previous_obs = env.reset_to({"states": start.previous_state})
    obs = env.reset_to({"states": start.state})
    history = [previous_obs] * max(0, n_obs_steps - 1) + [obs]
    history = history[-n_obs_steps:]
    policy.reset()

    video_path = (
        output_dir
        / "videos"
        / SUBTASK_DIRS[start.subtask]
        / f"chunk_{chunk_size:02d}"
        / f"episode_{episode_index:03d}_{start.demo}.mp4"
    )
    writer = VideoWriter(video_path, fps=video_fps) if save_video else None
    inference_ms: list[float] = []
    video_overhead_sec = 0.0
    step = 0
    policy_calls = 0
    done = False
    metrics = _live_task_metrics(env, obs, start.initial_needle_z)
    success = _task_success(start.subtask, metrics, approach_distance, lift_height)
    wall_start = time.perf_counter()

    try:
        while step < max_steps and not success and not done:
            policy_obs = _policy_observation(
                history, device, state_mode, camera_keys, camera_output_keys
            )
            inference_start = time.perf_counter()
            with torch.no_grad():
                action_result = policy.predict_action(policy_obs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_ms.append((time.perf_counter() - inference_start) * 1000.0)
            policy_calls += 1
            if feature_collector is not None:
                features = getattr(policy, "last_chunk_features", None)
                if features is None:
                    raise RuntimeError("Threading policy did not expose shared visual features")
                feature_collector.record(
                    features,
                    episode_id=start.demo,
                    decision_step=step,
                    subtask=start.subtask,
                )
            actions = action_result["action"][0].detach().cpu().numpy()
            if len(actions) != chunk_size:
                raise RuntimeError(
                    f"Requested chunk={chunk_size}, but policy returned {len(actions)} actions"
                )

            for action in actions:
                action = action.copy()
                action[:3] *= translation_scale
                obs, _, done, _ = env.step(action)
                step += 1
                # The policy was trained with consecutive observation frames.
                # Preserve that cadence even when several actions are executed
                # open-loop before the next inference call.
                history = (history + [obs])[-n_obs_steps:]
                metrics = _live_task_metrics(env, obs, start.initial_needle_z)
                success = _task_success(
                    start.subtask, metrics, approach_distance, lift_height
                )
                if writer is not None:
                    video_start = time.perf_counter()
                    video_images = [
                        np.moveaxis(image_from_obs(obs, key), 0, -1) for key in camera_keys
                    ]
                    writer.append(
                        compose_camera_views(
                            video_images,
                            list(camera_output_keys),
                            [
                                f"task={start.subtask} chunk={chunk_size} demo={start.demo}",
                                f"step={step}/{max_steps} success={success}",
                                (
                                    f"distance={metrics['handle_distance']:.3f}m "
                                    f"lift={metrics['lift_height']:.3f}m "
                                    f"grasp={metrics['grasped']} "
                                    f"insert={metrics['insert_distance']:.3f}/"
                                    f"{metrics['insert_threshold']:.3f}m"
                                ),
                            ],
                            size=video_size,
                        )
                    )
                    video_overhead_sec += time.perf_counter() - video_start
                if success or done or step >= max_steps:
                    break
    finally:
        if writer is not None:
            video_start = time.perf_counter()
            writer.close()
            video_overhead_sec += time.perf_counter() - video_start

    wall_time_sec = max(
        0.0,
        time.perf_counter() - wall_start - video_overhead_sec,
    )
    execution_time_sec = step / control_frequency_hz
    computer_time_sec = float(sum(inference_ms)) / 1000.0

    return SubtaskEpisode(
        chunk_size=chunk_size,
        subtask=start.subtask,
        demo=start.demo,
        demo_index=start.demo_index,
        start_index=start.start_index,
        episode=episode_index,
        seed=seed,
        success=success,
        steps=step,
        timeout=not success and step >= max_steps,
        done=bool(done),
        num_policy_calls=policy_calls,
        wall_time_sec=wall_time_sec,
        control_frequency_hz=control_frequency_hz,
        execution_time_sec=execution_time_sec,
        computer_time_sec=computer_time_sec,
        estimated_total_time_sec=execution_time_sec + computer_time_sec,
        video_overhead_sec=video_overhead_sec,
        inference_time_mean_ms=float(np.mean(inference_ms)) if inference_ms else 0.0,
        inference_time_p95_ms=float(np.percentile(inference_ms, 95)) if inference_ms else 0.0,
        final_handle_distance=float(metrics["handle_distance"]),
        final_lift_height=float(metrics["lift_height"]),
        final_grasped=bool(metrics["grasped"]),
        final_insert_distance=float(metrics["insert_distance"]),
        video=str(video_path) if writer is not None else None,
    )


def _plot_chunk_comparison(
    all_chunk_results: dict[int, dict[str, Any]],
    tasks: list[str],
    output_dir: Path,
) -> None:
    """Use the reference plot style with clearer subtask and label spacing."""
    chunk_sizes = sorted(all_chunk_results.keys())
    subtasks = sorted(tasks, key=lambda task: int(task))

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(12, 11),
        sharex=True,
        constrained_layout=True,
    )

    # All bars occupy 0.9 x-units, while adjacent subtasks are 1.8 units
    # apart. This leaves a clear gap between subtask groups regardless of the
    # number of candidate chunk sizes.
    group_spacing = 1.8
    x = np.arange(len(subtasks), dtype=np.float64) * group_spacing
    width = 0.9 / max(1, len(chunk_sizes))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(chunk_sizes))))

    for index, chunk_size in enumerate(chunk_sizes):
        rates = []
        for subtask in subtasks:
            result = all_chunk_results[chunk_size]["subtasks"].get(subtask, {})
            rates.append(result.get("success_rate", 0) * 100)
        offset = (index - len(chunk_sizes) / 2 + 0.5) * width
        bars = ax1.bar(
            x + offset,
            rates,
            width,
            label=f"chunk={chunk_size}",
            color=colors[index],
        )
        for bar, rate in zip(bars, rates):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=True,
            )

    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Chunk Size Comparison: Success Rate")
    ax1.set_ylim(0, 110)
    ax1.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    ax1.grid(axis="y", alpha=0.3)

    all_wall_times = [
        all_chunk_results[chunk_size]["subtasks"]
        .get(subtask, {})
        .get("wall_time", {})
        .get("mean", 0)
        for chunk_size in chunk_sizes
        for subtask in subtasks
    ]
    max_wall_time = max(all_wall_times, default=0.0)
    wall_label_offset = max(max_wall_time * 0.025, 0.05)

    for index, chunk_size in enumerate(chunk_sizes):
        times = []
        for subtask in subtasks:
            result = all_chunk_results[chunk_size]["subtasks"].get(subtask, {})
            times.append(result.get("wall_time", {}).get("mean", 0))
        offset = (index - len(chunk_sizes) / 2 + 0.5) * width
        bars = ax2.bar(
            x + offset,
            times,
            width,
            label=f"chunk={chunk_size}",
            color=colors[index],
        )
        for bar, wall_time in zip(bars, times):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + wall_label_offset,
                f"{wall_time:.1f}s",
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=True,
            )

    ax2.set_ylabel("Avg Wall Time (s)")
    ax2.set_title("Chunk Size Comparison: Wall Time")
    ax2.set_ylim(0, max_wall_time + max(max_wall_time * 0.18, 1.0))
    ax2.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    ax2.grid(axis="y", alpha=0.3)

    all_inference_passes = [
        all_chunk_results[chunk_size]["subtasks"]
        .get(subtask, {})
        .get("inference_passes", {})
        .get("mean", 0)
        for chunk_size in chunk_sizes
        for subtask in subtasks
    ]
    max_inference_passes = max(all_inference_passes, default=0.0)
    inference_label_offset = max(max_inference_passes * 0.025, 0.2)

    for index, chunk_size in enumerate(chunk_sizes):
        passes = []
        for subtask in subtasks:
            result = all_chunk_results[chunk_size]["subtasks"].get(subtask, {})
            passes.append(result.get("inference_passes", {}).get("mean", 0))
        offset = (index - len(chunk_sizes) / 2 + 0.5) * width
        bars = ax3.bar(
            x + offset,
            passes,
            width,
            label=f"chunk={chunk_size}",
            color=colors[index],
        )
        for bar, inference_passes in zip(bars, passes):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + inference_label_offset,
                f"{inference_passes:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=True,
            )

    ax3.set_ylabel("Avg Inference Passes")
    ax3.set_xlabel("Subtask")
    ax3.set_title("Chunk Size Comparison: Inference Passes")
    ax3.set_ylim(
        0,
        max_inference_passes + max(max_inference_passes * 0.18, 1.0),
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Subtask {subtask}" for subtask in subtasks])
    ax3.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    ax3.grid(axis="y", alpha=0.3)

    output_path = output_dir / "chunk_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Comparison chart saved → {output_path}")


def _write_results(
    output_dir: Path,
    metadata: dict[str, Any],
    stage_starts: dict[str, list[StageStart]],
    records: list[SubtaskEpisode],
    tasks: tuple[str, ...],
    chunk_sizes: list[int],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunk_results: dict[int, dict[str, Any]] = {}
    for chunk_size in chunk_sizes:
        chunk_records = [record for record in records if record.chunk_size == chunk_size]
        subtasks: dict[str, Any] = {}
        for task in tasks:
            task_records = [record for record in chunk_records if record.subtask == task]
            successful_records = [record for record in task_records if record.success]
            steps = [record.steps for record in successful_records]
            wall_times = [record.wall_time_sec for record in successful_records]
            inference_passes = [
                record.num_policy_calls for record in successful_records
            ]
            successes = len(successful_records)
            episodes = len(task_records)
            subtasks[SUBTASK_IDS[task]] = {
                "successes": successes,
                "failures": episodes - successes,
                "success_rate": round(successes / episodes, 4) if episodes else 0,
                "steps": {
                    "mean": round(float(np.mean(steps)), 1) if steps else 0,
                    "min": int(np.min(steps)) if steps else 0,
                    "max": int(np.max(steps)) if steps else 0,
                },
                "wall_time": {
                    "mean": round(float(np.mean(wall_times)), 2) if wall_times else 0,
                    "min": round(float(np.min(wall_times)), 2) if wall_times else 0,
                    "max": round(float(np.max(wall_times)), 2) if wall_times else 0,
                },
                "inference_passes": {
                    "mean": (
                        round(float(np.mean(inference_passes)), 1)
                        if inference_passes
                        else 0
                    ),
                    "min": int(np.min(inference_passes)) if inference_passes else 0,
                    "max": int(np.max(inference_passes)) if inference_passes else 0,
                },
            }
        all_chunk_results[chunk_size] = {
            "subtasks": subtasks,
            "overall_successes": sum(int(record.success) for record in chunk_records),
            "overall_total": len(chunk_records),
        }

    comparison_stats = {
        "checkpoint": metadata.get("model_path", metadata.get("checkpoint")),
        "episodes_per_subtask": len(next(iter(stage_starts.values()), [])),
        "seed": metadata.get("rollout_seed"),
        "chunk_sizes": {
            str(chunk_size): result
            for chunk_size, result in all_chunk_results.items()
        },
    }
    (output_dir / "chunk_comparison.json").write_text(
        json.dumps(comparison_stats, indent=2, ensure_ascii=False)
    )

    details = {
        "metadata": metadata,
        "stage_starts": {
            task: [start.summary() for start in starts]
            for task, starts in stage_starts.items()
        },
        "episodes": [asdict(record) for record in records],
    }
    (output_dir / "evaluation_details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False)
    )
    if records:
        with (output_dir / "episodes.csv").open("w", newline="") as file:
            rows = [asdict(record) for record in records]
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    task_ids = [SUBTASK_IDS[task] for task in tasks]
    _plot_chunk_comparison(all_chunk_results, task_ids, output_dir)

    best_by_task = {
        task: min(
            chunk_sizes,
            key=lambda chunk_size: (
                -all_chunk_results[chunk_size]["subtasks"][SUBTASK_IDS[task]][
                    "success_rate"
                ],
                all_chunk_results[chunk_size]["subtasks"][SUBTASK_IDS[task]][
                    "wall_time"
                ]["mean"],
                chunk_size,
            ),
        )
        for task in tasks
    }
    best_overall = min(
        chunk_sizes,
        key=lambda chunk_size: (
            -all_chunk_results[chunk_size]["overall_successes"],
            chunk_size,
        ),
    )
    return {
        "best_chunk_by_subtask": best_by_task,
        "best_overall_chunk": best_overall,
    }


def _parse_tasks(values: list[str]) -> tuple[str, ...]:
    if "all" in values:
        return SUBTASKS
    tasks: list[str] = []
    for value in values:
        task = SUBTASK_ALIASES.get(value, value)
        if task not in SUBTASKS:
            raise ValueError(f"Unknown subtask {value!r}; choose from {SUBTASKS} or all")
        if task not in tasks:
            tasks.append(task)
    return tuple(tasks)


def _default_chunk_sizes(horizon: int) -> list[int]:
    candidates = [
        max(1, round(horizon * percentage / 100))
        for percentage in (20, 40, 60, 80, 100)
    ]
    return sorted(set(candidates))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare candidate chunks on three decoupled Threading subtasks"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset", default="data/threading/threading_d0.hdf5")
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--episodes", type=int, default=20, help="episodes per subtask/chunk")
    parser.add_argument(
        "--candidate-chunks",
        type=int,
        nargs="+",
        default=None,
        help=(
            "exact numbers of generated/executed actions; default: "
            "20%%, 40%%, 60%%, 80%%, and 100%% of the policy horizon"
        ),
    )
    parser.add_argument(
        "--prediction-mode",
        choices=PREDICTION_MODES,
        default="full_then_truncate",
        help=(
            "full_then_truncate predicts the checkpoint horizon and executes a prefix; "
            "required_only predicts exactly the actions that will be executed"
        ),
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="scale the first three OSC translation action dimensions before env.step",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--gmm-eval-mode", choices=("checkpoint", "sample", "mean", "map"), default="checkpoint"
    )
    parser.add_argument("--state-mode", choices=("auto", "joint", "eef"), default="auto")
    parser.add_argument("--split", choices=("validation", "train", "all"), default="validation")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--demo-indices", type=int, nargs="+", default=None)
    parser.add_argument("--rollout-seed", type=int, default=10000)
    parser.add_argument("--approach-distance", type=float, default=0.10)
    parser.add_argument("--lift-height", type=float, default=0.05)
    parser.add_argument(
        "--pick-pregrasp-height",
        type=float,
        default=0.085,
        help=(
            "vertical clearance above the expert grasp pose for pick starts; "
            "default 0.085 m (the previous 0.035 m clearance plus 0.05 m)"
        ),
    )
    parser.add_argument("--approach-max-steps", type=int, default=DEFAULT_MAX_STEPS["approach"])
    parser.add_argument("--pick-max-steps", type=int, default=DEFAULT_MAX_STEPS["pick"])
    parser.add_argument("--insert-max-steps", type=int, default=DEFAULT_MAX_STEPS["insert"])
    parser.add_argument("--output-dir", type=Path, default=Path("threading_subtask_eval"))
    parser.add_argument(
        "--save-videos", choices=("none", "all", "successes", "failures"), default="failures"
    )
    parser.add_argument("--max-videos", type=int, default=10, help="kept videos per subtask/chunk")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-size", type=int, default=512)
    parser.add_argument(
        "--feature-dataset",
        type=Path,
        default=None,
        help="write coarse-labeled shared visual features to this HDF5 file",
    )
    parser.add_argument(
        "--coarse-labels",
        nargs="+",
        default=None,
        metavar="SUBTASK=CHUNK",
        help="required for collection, for example approach=1 pick=4 insert=8",
    )
    parser.add_argument(
        "--selector-candidates",
        type=int,
        nargs="+",
        default=None,
        help=(
            "selector class values; fixed evaluation defaults to powers of two, "
            "coarse collection defaults to label values"
        ),
    )
    parser.add_argument(
        "--collection-chunk",
        type=int,
        default=None,
        help="fixed behavior chunk used for collection; default: policy horizon",
    )
    parser.add_argument("--overwrite-feature-dataset", action="store_true")
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 0.0 < args.translation_scale <= 1.0:
        parser.error("--translation-scale must be in (0, 1]")
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be between 0 and 1")
    if (
        args.approach_distance <= 0.0
        or args.lift_height <= 0.0
        or args.pick_pregrasp_height <= 0.0
    ):
        parser.error(
            "--approach-distance, --lift-height, and --pick-pregrasp-height "
            "must be positive"
        )
    try:
        tasks = _parse_tasks(args.tasks)
    except ValueError as error:
        parser.error(str(error))

    policy = load_policy(
        args.checkpoint,
        device=args.device,
        weights=args.weights,
        use_checkpoint_config=True,
    )
    if not hasattr(policy, "inference_chunk_size"):
        parser.error("Checkpoint policy does not support inference_chunk_size")
    if not hasattr(policy, "set_prediction_mode"):
        parser.error("Checkpoint policy does not support explicit prediction modes")
    policy.set_prediction_mode(args.prediction_mode)
    horizon = int(policy.horizon)
    selector_candidates = sorted(
        set(args.selector_candidates or _default_chunk_sizes(horizon))
    )
    invalid_selector_candidates = [
        value for value in selector_candidates if value <= 0 or value > horizon
    ]
    if invalid_selector_candidates:
        parser.error(
            f"selector candidates must lie in [1, policy.horizon={horizon}]: "
            f"{invalid_selector_candidates}"
        )
    chunk_sizes = sorted(set(args.candidate_chunks or selector_candidates))
    feature_collector = None
    if args.feature_dataset is not None:
        if not args.coarse_labels:
            parser.error("--coarse-labels is required with --feature-dataset")
        behavior_chunk = args.collection_chunk or horizon
        if behavior_chunk <= 0 or behavior_chunk > horizon:
            parser.error(f"--collection-chunk must lie in [1, {horizon}]")
        try:
            coarse_labels = parse_subtask_chunks(args.coarse_labels)
        except ValueError as error:
            parser.error(str(error))
        if args.selector_candidates is None:
            selector_candidates = sorted(set(coarse_labels.values()))
        out_of_range_labels = sorted(
            value for value in set(coarse_labels.values()) if value > horizon
        )
        if out_of_range_labels:
            parser.error(
                f"coarse labels exceed policy horizon {horizon}: "
                f"{out_of_range_labels}"
            )
        missing_labels = [task for task in tasks if task not in coarse_labels]
        if missing_labels:
            parser.error(f"missing coarse labels for tasks: {missing_labels}")
        invalid_labels = sorted(set(coarse_labels.values()) - set(selector_candidates))
        if invalid_labels:
            parser.error(
                f"coarse labels {invalid_labels} are absent from selector candidates "
                f"{selector_candidates}"
            )
        unused_candidates = sorted(
            set(selector_candidates) - set(coarse_labels.values())
        )
        if unused_candidates:
            parser.error(
                "coarse collection would create classes without positive samples: "
                f"{unused_candidates}"
            )
        if args.feature_dataset.exists() and not args.overwrite_feature_dataset:
            parser.error(
                f"{args.feature_dataset} already exists; pass --overwrite-feature-dataset"
            )
        chunk_sizes = [behavior_chunk]
        feature_collector = CoarseChunkFeatureCollector(
            args.feature_dataset,
            candidate_chunks=selector_candidates,
            subtask_chunks=coarse_labels,
            overwrite=args.overwrite_feature_dataset,
            metadata={
                "policy_type": type(policy).__name__,
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "behavior_chunk": behavior_chunk,
            },
        )
    invalid_chunks = [value for value in chunk_sizes if value <= 0 or value > horizon]
    if invalid_chunks:
        parser.error(f"candidate chunks must lie in [1, policy.horizon={horizon}]: {invalid_chunks}")
    if args.gmm_eval_mode != "checkpoint":
        if args.gmm_eval_mode == "map":
            from threading_task.policy import enable_map_gmm_inference

            enable_map_gmm_inference(policy)
        policy.use_sample = {"sample": True, "mean": False, "map": "map"}[
            args.gmm_eval_mode
        ]

    agent_state_dim = int(getattr(policy, "agent_state_dim", 9))
    state_mode = (
        ("eef" if agent_state_dim == 8 else "joint")
        if args.state_mode == "auto"
        else args.state_mode
    )
    expected_dim = 8 if state_mode == "eef" else 9
    if agent_state_dim != expected_dim:
        parser.error(
            f"state_mode={state_mode} produces {expected_dim}D state, checkpoint expects {agent_state_dim}D"
        )
    camera_output_keys = tuple(getattr(policy, "rgb_keys", ("top45", "sideview")))
    camera_keys = getattr(policy, "camera_obs_keys", None)
    if camera_keys is None:
        camera_keys = ("agentview_image", "robot0_eye_in_hand_image")
    camera_keys = tuple(camera_keys)
    image_shapes = getattr(policy, "image_shapes", None)
    if image_shapes is not None:
        camera_sizes = tuple(int(image_shapes[key][-1]) for key in camera_output_keys)
    else:
        image_shape = getattr(policy, "image_shape", (3, 96, 96))
        camera_sizes = (int(image_shape[-1]),) * len(camera_keys)

    print(f"[eval] tasks={tasks} chunks={chunk_sizes} horizon={horizon}")
    print(
        f"[eval] prediction_mode={policy.prediction_mode} "
        f"translation_scale={args.translation_scale}"
    )
    print(f"[eval] paired demos: split={args.split} episodes={args.episodes}")
    print(
        f"[eval] observations: state={state_mode}, "
        f"cameras={dict(zip(camera_output_keys, zip(camera_keys, camera_sizes)))}"
    )
    try:
        stage_starts = build_stage_starts(
            args.dataset,
            tasks,
            args.episodes,
            args.split,
            args.val_ratio,
            args.split_seed,
            args.selection_seed,
            args.demo_indices,
            args.approach_distance,
            args.lift_height,
            args.pick_pregrasp_height,
        )
    except (KeyError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    env, env_meta = create_threading_env(
        args.dataset,
        args.env_name,
        camera_names=tuple(
            key[: -len("_image")] if key.endswith("_image") else key for key in camera_keys
        ),
        camera_size=camera_sizes,
    )
    control_frequency_hz = float(getattr(env.env, "control_freq", 20.0))
    max_steps = {
        "approach": args.approach_max_steps,
        "pick": args.pick_max_steps,
        "insert": args.insert_max_steps,
    }
    records: list[SubtaskEpisode] = []
    try:
        for chunk_size in chunk_sizes:
            policy.inference_chunk_size = chunk_size
            print(f"\n[chunk={chunk_size}]")
            for task in tasks:
                task_records: list[SubtaskEpisode] = []
                kept_videos = 0
                for episode_index, start in enumerate(stage_starts[task]):
                    record_video = (
                        args.save_videos != "none" and kept_videos < args.max_videos
                    )
                    record = run_episode(
                        env=env,
                        policy=policy,
                        start=start,
                        chunk_size=chunk_size,
                        episode_index=episode_index,
                        seed=args.rollout_seed + start.demo_index,
                        max_steps=max_steps[task],
                        approach_distance=args.approach_distance,
                        lift_height=args.lift_height,
                        state_mode=state_mode,
                        camera_keys=camera_keys,
                        camera_output_keys=camera_output_keys,
                        output_dir=args.output_dir,
                        save_video=record_video,
                        video_fps=args.video_fps,
                        video_size=args.video_size,
                        control_frequency_hz=control_frequency_hz,
                        translation_scale=args.translation_scale,
                        feature_collector=feature_collector,
                    )
                    keep_video = (
                        record.video is not None
                        and kept_videos < args.max_videos
                        and _should_keep_video(args.save_videos, record.success)
                    )
                    if keep_video:
                        kept_videos += 1
                    elif record.video is not None:
                        Path(record.video).unlink(missing_ok=True)
                        record.video = None
                    task_records.append(record)
                    records.append(record)
                    mark = "PASS" if record.success else "FAIL"
                    print(
                        f"  {task:8s} {episode_index + 1:02d}/{args.episodes:02d} "
                        f"{start.demo:>10s} {mark} steps={record.steps:3d} "
                        f"exec={record.execution_time_sec:.2f}s "
                        f"compute={record.computer_time_sec:.2f}s "
                        f"total={record.estimated_total_time_sec:.2f}s "
                        f"wall={record.wall_time_sec:.2f}s"
                    )
                aggregate = aggregate_episodes(
                    [record.standard_metrics() for record in task_records]
                )
                print(
                    f"  => {task}: {aggregate['successes']}/{aggregate['num_episodes']} "
                    f"({aggregate['success_rate']:.1%}), avg_steps={aggregate['avg_steps_all']:.1f}"
                )
    finally:
        env.close()
        if feature_collector is not None:
            feature_collector.close()

    metadata = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "dataset": str(Path(args.dataset).expanduser()),
        "env_name": args.env_name or env_meta.get("env_name"),
        "weights": args.weights,
        "prediction_mode": args.prediction_mode,
        "translation_scale": args.translation_scale,
        "gmm_eval_mode": args.gmm_eval_mode,
        "tasks": list(tasks),
        "candidate_chunks": chunk_sizes,
        "policy_horizon": horizon,
        "state_mode": state_mode,
        "camera_keys": list(camera_keys),
        "camera_output_keys": list(camera_output_keys),
        "camera_sizes": list(camera_sizes),
        "control_frequency_hz": control_frequency_hz,
        "timing_definition": {
            "execution_time_sec": "environment steps / control_frequency_hz",
            "computer_time_sec": "sum of policy inference wall-clock durations",
            "estimated_total_time_sec": "execution_time_sec + computer_time_sec",
            "wall_time_sec": "measured simulation rollout wall time excluding video encoding",
        },
        "split": args.split,
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "selection_seed": args.selection_seed,
        "rollout_seed": args.rollout_seed,
        "approach_distance": args.approach_distance,
        "lift_height": args.lift_height,
        "insert_success_criterion": (
            "needle geom center distance to ring center < ring radius (historical deep insertion)"
        ),
        "max_steps": max_steps,
    }
    payload = _write_results(
        args.output_dir, metadata, stage_starts, records, tasks, chunk_sizes
    )
    if args.feature_dataset is not None:
        print("\nCoarse labels written to the feature dataset:")
        for task in tasks:
            print(f"  {task:8s}: {coarse_labels[task]}")
    else:
        print("\nBest chunks:")
        for task, chunk_size in payload["best_chunk_by_subtask"].items():
            print(f"  {task:8s}: {chunk_size}")
        print(f"  overall : {payload['best_overall_chunk']}")
    print(f"Results: {args.output_dir.resolve()}")
    if args.feature_dataset is not None:
        print(f"Feature dataset: {args.feature_dataset.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
