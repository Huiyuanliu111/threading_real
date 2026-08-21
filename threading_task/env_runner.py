"""Online Threading rollout runner used during ARP training."""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from pushbox.diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from pushbox.diffusion_policy.policy.base_image_policy import BaseImagePolicy

from .chunk_labels import OnlineSpatialChunkRule, SpatialChunkDecision
from .env import agent_state_from_obs, create_threading_env, image_from_obs, seed_env, success_from_env
from .metrics import EpisodeMetrics, aggregate_episodes, write_metrics
from .visualization import VideoWriter, compose_camera_views


class ThreadingImageRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir: str,
        dataset_path: str,
        env_name: str | None = None,
        n_eval_episodes: int = 10,
        max_steps: int = 1000,
        test_start_seed: int = 10000,
        n_video_episodes: int = 3,
        video_fps: int = 20,
        video_size: int = 256,
        joint_key: str = "robot0_joint_pos",
        gripper_key: str = "robot0_gripper_qpos",
        eef_pos_key: str = "robot0_eef_pos",
        eef_quat_key: str = "robot0_eef_quat",
        state_mode: str = "joint",
        camera_keys: tuple[str, ...] = ("agentview_image", "robot0_eye_in_hand_image"),
        camera_output_keys: tuple[str, ...] | None = None,
        camera_size: int = 96,
        camera_sizes: tuple[int, ...] | None = None,
        translation_scale: float = 0.25,
        spatial_chunk_rule: dict[str, Any] | None = None,
        save_chunk_trace: bool = False,
        chunk_trace_spatial_thresholds: dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__(output_dir)
        self.dataset_path = dataset_path
        self.env_name = env_name
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.test_start_seed = test_start_seed
        self.n_video_episodes = n_video_episodes
        self.video_fps = video_fps
        self.video_size = video_size
        self.joint_key = joint_key
        self.gripper_key = gripper_key
        self.eef_pos_key = eef_pos_key
        self.eef_quat_key = eef_quat_key
        self.state_mode = state_mode
        self.camera_keys = tuple(camera_keys)
        if camera_output_keys is None:
            if len(self.camera_keys) != 2:
                raise ValueError("camera_output_keys is required when using other than two cameras")
            camera_output_keys = ("top45", "sideview")
        self.camera_output_keys = tuple(camera_output_keys)
        if len(self.camera_output_keys) != len(self.camera_keys):
            raise ValueError("camera_keys and camera_output_keys must have equal length")
        self.camera_size = int(camera_size)
        self.camera_sizes = (
            tuple(int(size) for size in camera_sizes)
            if camera_sizes is not None
            else (self.camera_size,) * len(self.camera_keys)
        )
        if len(self.camera_sizes) != len(self.camera_keys):
            raise ValueError("camera_sizes must contain one value per camera")
        if not 0.0 < translation_scale <= 1.0:
            raise ValueError("translation_scale must be in (0, 1]")
        self.translation_scale = float(translation_scale)
        self.spatial_chunk_rule = (
            None
            if spatial_chunk_rule is None
            else OnlineSpatialChunkRule(**spatial_chunk_rule)
        )
        self.save_chunk_trace = bool(save_chunk_trace)
        thresholds = chunk_trace_spatial_thresholds or {}
        self.chunk_trace_rule = (
            OnlineSpatialChunkRule(
                small_chunk=4,
                full_chunk=10,
                grasp_distance=float(thresholds.get("grasp_distance", 0.10)),
                lift_threshold=float(thresholds.get("lift_threshold", 0.05)),
                insert_approach_distance=float(
                    thresholds.get("insert_approach_distance", 0.20)
                ),
            )
            if self.save_chunk_trace
            else None
        )
        self._env = None
        self._env_meta = None
        self._run_index = 0

    def _get_env(self):
        if self._env is None:
            self._env, self._env_meta = create_threading_env(
                self.dataset_path,
                self.env_name,
                camera_names=tuple(
                    key[: -len("_image")] if key.endswith("_image") else key
                    for key in self.camera_keys
                ),
                camera_size=self.camera_sizes,
            )
        return self._env

    def run(self, policy: BaseImagePolicy) -> Dict:
        env = self._get_env()
        device = next(policy.parameters()).device
        n_obs_steps = int(policy.n_obs_steps)
        output = Path(self.output_dir) / "threading_rollouts" / f"run_{self._run_index:04d}"
        videos_dir = output / "videos"
        episodes: list[EpisodeMetrics] = []
        video_paths: list[Path] = []
        view_weight_history: dict[str, list[float]] = {}
        selected_chunk_counts: Counter[int] = Counter()
        selected_region_counts: Counter[str] = Counter()
        chunk_trace: list[dict[str, Any]] = []

        for episode_index in range(self.n_eval_episodes):
            seed = self.test_start_seed + episode_index
            seed_env(env, seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            obs = env.reset()
            policy.reset()
            if self.spatial_chunk_rule is not None:
                self.spatial_chunk_rule.reset(obs)
            if self.chunk_trace_rule is not None:
                self.chunk_trace_rule.reset(obs)
            episode_trace: list[dict[str, Any]] = []
            state_buf: list[np.ndarray] = []
            image_buffers: dict[str, list[np.ndarray]] = {
                key: [] for key in self.camera_output_keys
            }
            rewards: list[float] = []
            inference_ms: list[float] = []
            policy_calls = 0
            step = 0
            success = False
            wall_start = time.perf_counter()
            video_path = videos_dir / f"episode_{episode_index:03d}.mp4"
            writer = VideoWriter(video_path, fps=self.video_fps) if episode_index < self.n_video_episodes else None

            try:
                while step < self.max_steps and not success:
                    spatial_decision: SpatialChunkDecision | None = None
                    trace_spatial_decision: SpatialChunkDecision | None = None
                    if self.spatial_chunk_rule is not None:
                        spatial_decision = self.spatial_chunk_rule.select(obs)
                        policy.inference_chunk_size = spatial_decision.chunk_size
                    if self.chunk_trace_rule is not None:
                        trace_spatial_decision = self.chunk_trace_rule.select(obs)
                    camera_images = {
                        output_key: image_from_obs(obs, camera_key)
                        for output_key, camera_key in zip(
                            self.camera_output_keys, self.camera_keys
                        )
                    }
                    state = agent_state_from_obs(
                        obs,
                        joint_key=self.joint_key,
                        gripper_key=self.gripper_key,
                        eef_pos_key=self.eef_pos_key,
                        eef_quat_key=self.eef_quat_key,
                        state_mode=self.state_mode,
                    )
                    if not state_buf:
                        state_buf = [state.copy() for _ in range(n_obs_steps)]
                        image_buffers = {
                            key: [image.copy() for _ in range(n_obs_steps)]
                            for key, image in camera_images.items()
                        }
                    else:
                        state_buf = (state_buf + [state])[-n_obs_steps:]
                        image_buffers = {
                            key: (image_buffers[key] + [image])[-n_obs_steps:]
                            for key, image in camera_images.items()
                        }

                    policy_obs = {
                        "agent_pos": torch.from_numpy(np.stack(state_buf))[None].to(device),
                        **{
                            key: torch.from_numpy(np.stack(buffer))[None].to(device)
                            for key, buffer in image_buffers.items()
                        },
                    }
                    infer_start = time.perf_counter()
                    with torch.no_grad():
                        action_result = policy.predict_action(policy_obs)
                    for key, value in getattr(policy, "last_view_weights", {}).items():
                        view_weight_history.setdefault(key, []).append(float(value))
                    inference_ms.append((time.perf_counter() - infer_start) * 1000)
                    policy_calls += 1
                    if spatial_decision is not None:
                        selected_chunk_counts[spatial_decision.chunk_size] += 1
                        selected_region_counts[spatial_decision.region] += 1
                    elif "chunk_size" in action_result:
                        chunks = action_result["chunk_size"].detach().cpu().reshape(-1).tolist()
                        selected_chunk_counts.update(int(value) for value in chunks)
                    actions = action_result["action"][0].detach().cpu().numpy()
                    trace_row: dict[str, Any] | None = None
                    if self.save_chunk_trace and "chunk_size" in action_result:
                        probabilities = (
                            action_result["chunk_probabilities"][0]
                            .detach()
                            .cpu()
                            .reshape(-1)
                            .tolist()
                        )
                        candidates = tuple(
                            int(value)
                            for value in policy.chunk_selector.config.candidate_chunks
                        )
                        trace_row = {
                            "episode": episode_index,
                            "seed": seed,
                            "policy_call": policy_calls - 1,
                            "step": step,
                            "selected_chunk": int(
                                action_result["chunk_size"].detach().cpu().item()
                            ),
                            "confidence": float(
                                action_result["chunk_confidence"].detach().cpu().item()
                            ),
                            **{
                                f"prob_chunk_{chunk}": float(probability)
                                for chunk, probability in zip(candidates, probabilities)
                            },
                            "spatial_region": (
                                trace_spatial_decision.region
                                if trace_spatial_decision is not None
                                else None
                            ),
                            "handle_distance": (
                                trace_spatial_decision.handle_distance
                                if trace_spatial_decision is not None
                                else None
                            ),
                            "lift_height": (
                                trace_spatial_decision.lift_height
                                if trace_spatial_decision is not None
                                else None
                            ),
                            "insert_distance": (
                                trace_spatial_decision.insert_distance
                                if trace_spatial_decision is not None
                                else None
                            ),
                        }
                        episode_trace.append(trace_row)

                    executed_steps = 0
                    for action in actions:
                        scaled_action = np.array(action, copy=True)
                        scaled_action[:3] *= self.translation_scale
                        obs, reward, done, _ = env.step(scaled_action)
                        reward = float(reward)
                        rewards.append(reward)
                        step += 1
                        executed_steps += 1
                        success = success_from_env(env)
                        if writer is not None:
                            video_images = [
                                np.moveaxis(image_from_obs(obs, key), 0, -1)
                                for key in self.camera_keys
                            ]
                            view_line = "views=" + ", ".join(
                                f"{key}:{value:.2f}"
                                for key, value in getattr(
                                    policy, "last_view_weights", {}
                                ).items()
                            )
                            spatial_line = ""
                            if spatial_decision is not None:
                                spatial_line = (
                                    f"region={spatial_decision.region} "
                                    f"chunk={spatial_decision.chunk_size} "
                                    f"pick_d={spatial_decision.handle_distance:.3f} "
                                    f"lift={spatial_decision.lift_height:.3f} "
                                    f"insert_d={spatial_decision.insert_distance:.3f}"
                                )
                            writer.append(
                                compose_camera_views(
                                    video_images,
                                    list(self.camera_keys),
                                    [
                                        f"episode={episode_index} seed={seed} step={step}",
                                        f"reward={reward:.3f} success={success}",
                                        view_line,
                                        spatial_line,
                                    ],
                                    size=self.video_size,
                                )
                            )
                        if success or done or step >= self.max_steps:
                            break
                    if trace_row is not None:
                        trace_row["executed_steps"] = executed_steps
            finally:
                if writer is not None:
                    writer.close()

            if writer is not None:
                video_paths.append(video_path)
            episodes.append(
                EpisodeMetrics(
                    episode=episode_index,
                    seed=seed,
                    success=success,
                    steps=step,
                    episode_return=float(sum(rewards)),
                    max_reward=float(max(rewards, default=0.0)),
                    timeout=not success and step >= self.max_steps,
                    num_policy_calls=policy_calls,
                    wall_time_sec=time.perf_counter() - wall_start,
                    inference_time_mean_ms=float(np.mean(inference_ms)) if inference_ms else 0.0,
                    inference_time_p95_ms=float(np.percentile(inference_ms, 95)) if inference_ms else 0.0,
                    video=str(video_path) if writer is not None else None,
                )
            )
            for row in episode_trace:
                row["episode_steps"] = step
                row["normalized_progress"] = (
                    float(row["step"]) / step if step else 0.0
                )
                row["budget_progress"] = float(row["step"]) / self.max_steps
                row["success"] = success
                row["timeout"] = not success and step >= self.max_steps
            chunk_trace.extend(episode_trace)

        if chunk_trace:
            trace_json = output / "chunk_trace.json"
            trace_csv = output / "chunk_trace.csv"
            output.mkdir(parents=True, exist_ok=True)
            trace_json.write_text(json.dumps(chunk_trace, indent=2))
            with trace_csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(chunk_trace[0]))
                writer.writeheader()
                writer.writerows(chunk_trace)

        payload = write_metrics(
            output,
            episodes,
            metadata={
                "dataset_path": self.dataset_path,
                "env_name": self.env_name,
                "translation_scale": self.translation_scale,
                "chunk_trace": (
                    "chunk_trace.csv" if chunk_trace else None
                ),
                "chunk_trace_spatial_rule": (
                    self.chunk_trace_rule.config()
                    if self.chunk_trace_rule is not None
                    else None
                ),
                "selected_chunk_counts": {
                    str(chunk): count
                    for chunk, count in sorted(selected_chunk_counts.items())
                },
                "selected_region_counts": dict(sorted(selected_region_counts.items())),
                "spatial_chunk_rule": (
                    None
                    if self.spatial_chunk_rule is None
                    else self.spatial_chunk_rule.config()
                ),
            },
        )
        aggregate = payload["aggregate"]
        result: Dict = {
            "test_mean_score": aggregate["success_rate"],
            "test_success_rate": aggregate["success_rate"],
            "test_timeout_rate": aggregate["timeout_rate"],
            "test_avg_steps": aggregate["avg_steps_all"],
            "test_avg_success_steps": aggregate["avg_steps_success"] or 0.0,
            "test_avg_policy_calls": aggregate["avg_policy_calls"],
            "test_inference_time_ms": aggregate["inference_time_mean_ms"],
        }
        for key, values in view_weight_history.items():
            result[f"test_view_weight_{key}"] = float(np.mean(values))
        for chunk, count in sorted(selected_chunk_counts.items()):
            result[f"test_selected_chunk_{chunk}_count"] = count
        for region, count in sorted(selected_region_counts.items()):
            result[f"test_selected_region_{region}_count"] = count
        try:
            import wandb

            for index, video_path in enumerate(video_paths):
                result[f"test_video_{index}"] = wandb.Video(str(video_path), format="mp4")
        except (ImportError, OSError):
            pass
        if self.spatial_chunk_rule is not None:
            policy.inference_chunk_size = None
        self._run_index += 1
        return result

    def __del__(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
