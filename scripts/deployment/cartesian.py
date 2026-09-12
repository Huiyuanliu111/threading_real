#!/usr/bin/env python3
"""Deploy a 7D base-frame TCP-delta ThreadingReal policy through TrackC."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import select
import signal
import sys
import termios
import time
from typing import Any, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "remote_controller" / "src"))

from scripts.deployment.joint import (  # noqa: E402
    DEFAULT_FRONTVIEW_SERIAL,
    DEFAULT_SIDEVIEW_SERIAL,
    DEFAULT_WRIST_SERIAL,
    PANDA_LOWER,
    PANDA_UPPER,
    TRAINING_START_Q,
    GripperController,
    RealSenseRig,
    diagonal_stiffness,
    make_observation,
    stack_observations,
)
from scripts.deployment.endpoint_schedule import EndpointExecutionSchedule
def _live_rays(intrinsic: dict[str, float]) -> np.ndarray:
    u, v = np.meshgrid(np.arange(int(intrinsic["width"])), np.arange(int(intrinsic["height"])))
    return np.stack((
        (u - intrinsic["ppx"]) / intrinsic["fx"],
        (v - intrinsic["ppy"]) / intrinsic["fy"],
        np.ones_like(u),
    ), axis=-1).astype(np.float32)


def make_live_pointcloud_observation(
    rgbd_frames, cameras: RealSenseRig, calibration: dict[str, Any],
    q: Sequence[float], gripper_width: float, tcp_pose: np.ndarray,
    num_points: int, bounds: np.ndarray, rng: np.random.Generator,
    *, mvt_mode: bool = False, axis_length: float = 0.04,
) -> dict[str, torch.Tensor]:
    from build_vla_pointcloud_dataset import crop_and_sample, depth_to_base_points
    from threading_task.pointcloud_dataset import rasterize_bev

    points, colors, camera_ids = [], [], []
    frame_by_view = dict(zip(("sideview", "wrist", "frontview"), rgbd_frames, strict=True))
    for calibration_key, source_id in (("sideview", 0), ("frontview", 1)):
        rig_index = cameras.view_names.index(calibration_key)
        rgb, depth = frame_by_view[calibration_key]
        xyz, valid = depth_to_base_points(
            depth, _live_rays(cameras.color_intrinsics[rig_index]),
            cameras.depth_scales[rig_index],
            np.asarray(calibration["cameras"][calibration_key]["camera_from_base"]),
        )
        points.append(xyz)
        colors.append(rgb[valid])
        camera_ids.append(np.full(len(xyz), source_id, dtype=np.uint8))
    state = np.concatenate((np.asarray(q, np.float32), [gripper_width])).astype(np.float32)
    if mvt_mode:
        from build_threading_mvt_dataset import deterministic_surface_points
        xyz, rgb, valid_count, _ = deterministic_surface_points(
            points, colors, bounds, num_points, voxel_m=0.001)
        pose = np.asarray(tcp_pose, dtype=np.float32)
        control = np.stack((pose[:3, 3],
                            pose[:3, 3] + pose[:3, 0] * axis_length,
                            pose[:3, 3] + pose[:3, 1] * axis_length)).astype(np.float32)
        return {
            "points": torch.from_numpy(xyz),
            "colors": torch.from_numpy(rgb.astype(np.float32) / 255),
            "valid_points": torch.tensor(valid_count, dtype=torch.int64),
            "agent_pos": torch.from_numpy(state),
            "control_points": torch.from_numpy(control),
        }
    xyz, rgb, source, _ = crop_and_sample(points, colors, camera_ids, bounds, num_points, rng)
    bev = rasterize_bev(xyz, rgb.astype(np.float32) / 255, source, bounds, 64)
    return {
        "points": torch.from_numpy(xyz), "colors": torch.from_numpy(rgb.astype(np.float32) / 255),
        "camera_id": torch.from_numpy(source.astype(np.int64)),
        "bev": torch.from_numpy(bev),
        "agent_pos": torch.from_numpy(state), "tcp_pos": torch.from_numpy(np.asarray(tcp_pose[:3, 3], np.float32)),
    }


def stack_pointcloud_observations(history, device: str) -> dict[str, torch.Tensor]:
    return {key: torch.stack([frame[key] for frame in history]).unsqueeze(0).to(device) for key in history[0]}


def cartesian_stiffness(values: Sequence[float]) -> list[list[float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("cartesian stiffness must contain 6 finite non-negative values")
    return np.diag(values).tolist()


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit else vector * (limit / norm)


def integrate_cartesian_delta_chunk(
    deltas: np.ndarray,
    current_T: np.ndarray,
    current_width: float,
    *,
    max_first_translation: float,
    max_step_translation: float,
    max_first_rotation: float,
    max_step_rotation: float,
    workspace_min: np.ndarray | None,
    workspace_max: np.ndarray | None,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, float]]:
    """Integrate base-frame ``[dxyz, drotvec, dwidth]`` actions safely."""
    deltas = np.asarray(deltas, dtype=np.float64)
    T = np.asarray(current_T, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[1] != 7 or len(deltas) == 0:
        raise ValueError(f"expected non-empty Nx7 Cartesian-delta chunk, got {deltas.shape}")
    if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not np.all(np.isfinite(deltas)):
        raise ValueError("Cartesian actions and current TCP pose must be finite")
    raw_translation = np.linalg.norm(deltas[:, :3], axis=1)
    raw_rotation = np.linalg.norm(deltas[:, 3:6], axis=1)
    poses: list[np.ndarray] = []
    widths: list[float] = []
    pose = T.copy()
    width = float(current_width)
    for index, delta in enumerate(deltas):
        translation_limit = max_first_translation if index == 0 else max_step_translation
        rotation_limit = max_first_rotation if index == 0 else max_step_rotation
        dxyz = _clip_norm(delta[:3], translation_limit)
        drotvec = _clip_norm(delta[3:6], rotation_limit)
        pose = pose.copy()
        pose[:3, 3] += dxyz
        # Dataset convention: translation and rotation deltas are in base frame.
        pose[:3, :3] = Rotation.from_rotvec(drotvec).as_matrix() @ pose[:3, :3]
        if workspace_min is not None and (
            np.any(pose[:3, 3] < workspace_min) or np.any(pose[:3, 3] > workspace_max)
        ):
            raise RuntimeError(
                f"TCP target {pose[:3, 3].round(4).tolist()} leaves configured workspace"
            )
        poses.append(pose)
        width = float(np.clip(width + delta[6], 0.0, 0.08))
        widths.append(width)
    return poses, np.asarray(widths), {
        "raw_first_translation": float(raw_translation[0]),
        "raw_step_translation": float(np.max(raw_translation[1:]) if len(deltas) > 1 else 0.0),
        "raw_first_rotation": float(raw_rotation[0]),
        "raw_step_rotation": float(np.max(raw_rotation[1:]) if len(deltas) > 1 else 0.0),
        "safe_first_translation": float(np.linalg.norm(poses[0][:3, 3] - T[:3, 3])),
    }


def cartesian_pose_error(current_T: np.ndarray, target_T: np.ndarray) -> tuple[float, float]:
    """Return TCP translation (m) and rotation (rad) error."""
    current_T = np.asarray(current_T, dtype=np.float64)
    target_T = np.asarray(target_T, dtype=np.float64)
    if current_T.shape != (4, 4) or target_T.shape != (4, 4):
        raise ValueError("current and target TCP poses must be 4x4 matrices")
    translation = float(np.linalg.norm(target_T[:3, 3] - current_T[:3, 3]))
    rotation = float(
        Rotation.from_matrix(target_T[:3, :3] @ current_T[:3, :3].T).magnitude()
    )
    return translation, rotation


class SmolVLADeploymentPolicy:
    """Adapt a LeRobot SmolVLA checkpoint to the existing TrackC runner."""

    action_mode = "cartesian_delta"
    action_dim = 7
    uses_pointcloud = False
    rgb_keys = ("sideview", "wrist", "frontview")
    image_shape = (3, 224, 224)
    n_obs_steps = 1

    def __init__(self, checkpoint: Path, device: str, task: str):
        from lerobot.policies.factory import make_pre_post_processors
        try:
            from lerobot.policies.smolvla import SmolVLAPolicy
        except ImportError:
            try:
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            except ImportError as exc:
                raise ImportError(
                    "SmolVLA deployment requires the project .venv-smolvla environment; "
                    "run `source /home/huiyuan/teleoperation/.venv-smolvla/bin/activate`"
                ) from exc

        self.device = device
        self.task = task
        self.model = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.horizon = int(self.model.config.chunk_size)
        self.n_action_steps = int(self.model.config.n_action_steps)
        action_feature = self.model.config.output_features["action"]
        if self.horizon != 10 or int(action_feature.shape[0]) != 7:
            raise ValueError("SmolVLA checkpoint must emit a 10x7 action chunk")

    def predict_action(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        expected = {"sideview", "wrist", "frontview", "agent_pos"}
        missing = expected.difference(observations)
        if missing:
            raise KeyError(f"missing live SmolVLA observations: {sorted(missing)}")
        frame = {
            # Feed the original dataset keys. The saved processor renames them
            # to camera1/2/3 exactly as it did during training.
            "observation.images.exterior_image_2_right": observations["sideview"][0, -1],
            "observation.images.wrist_image_left": observations["wrist"][0, -1],
            "observation.images.exterior_image_1_left": observations["frontview"][0, -1],
            "observation.state": observations["agent_pos"][0, -1],
            "task": self.task,
        }
        batch = self.preprocess(frame)
        chunk = self.postprocess(self.model.predict_action_chunk(batch))
        return {"action": chunk[:, : self.n_action_steps]}


class PI05DeploymentPolicy:
    """Adapt the two-view LeRobot pi0.5 checkpoint to the TrackC runner."""

    action_mode = "cartesian_delta"
    action_dim = 7
    uses_pointcloud = False
    rgb_keys = ("sideview", "frontview")
    image_shape = (3, 224, 224)
    n_obs_steps = 1

    def __init__(
        self,
        checkpoint: Path,
        device: str,
        task: str,
        *,
        selector: Path | None = None,
        prediction_mode: str = "full_then_truncate",
    ):
        self.device = device
        self.task = task
        self.prediction_mode = prediction_mode
        self.last_prediction_diagnostics: dict[str, Any] | None = None
        self.engine = None
        if selector is not None:
            from pi05.chunk_selector.inference import PI05SelectorInference

            self.engine = PI05SelectorInference(checkpoint, selector, device=device)
            self.model = self.engine.policy
            self.horizon = self.engine.full_horizon
            self.n_action_steps = self.horizon
            return

        from lerobot.policies.factory import make_pre_post_processors
        try:
            from lerobot.policies.pi05 import PI05Policy
        except ImportError:
            try:
                from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            except ImportError as exc:
                raise ImportError(
                    "pi0.5 deployment requires the project .venv-pi05 environment; "
                    "run `source /home/huiyuan/teleoperation/.venv-pi05/bin/activate`"
                ) from exc

        self.model = PI05Policy.from_pretrained(checkpoint).to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.horizon = int(self.model.config.chunk_size)
        self.n_action_steps = int(self.model.config.n_action_steps)
        action_feature = self.model.config.output_features["action"]
        if self.horizon != 10 or int(action_feature.shape[0]) != 7:
            raise ValueError("pi0.5 checkpoint must emit a 10x7 action chunk")

    @property
    def adaptive(self) -> bool:
        return self.engine is not None

    def set_prediction_mode(self, mode: str) -> None:
        if mode not in {"required_only", "full_then_truncate"}:
            raise ValueError(f"unknown pi0.5 prediction mode: {mode}")
        self.prediction_mode = mode

    def predict_action(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        expected = {"sideview", "frontview", "agent_pos"}
        missing = expected.difference(observations)
        if missing:
            raise KeyError(f"missing live pi0.5 observations: {sorted(missing)}")
        frame = {
            "observation.images.exterior_image_2_right": observations["sideview"][0, -1],
            "observation.images.exterior_image_1_left": observations["frontview"][0, -1],
            "observation.state": observations["agent_pos"][0, -1],
            "task": self.task,
        }
        if self.engine is not None:
            prediction = self.engine.predict(
                self.engine.prepare(frame), mode=self.prediction_mode
            )
            self.last_prediction_diagnostics = {
                "prediction_mode": self.prediction_mode,
                "execution_chunk": prediction.execution_chunk,
                "predicted_chunk": prediction.predicted_chunk,
                "continuous_chunk": prediction.continuous_chunk,
                "probability_4": prediction.probabilities[0],
                "probability_10": prediction.probabilities[1],
                "vision_seconds": prediction.vision_seconds,
                "selector_seconds": prediction.selector_seconds,
                "action_seconds": prediction.action_seconds,
                "total_seconds": prediction.total_seconds,
            }
            actions = prediction.actions.clone()
            actions[..., 6] = 0.0
            return {"action": actions}
        batch = self.preprocess(frame)
        chunk = self.postprocess(self.model.predict_action_chunk(batch))
        chunk = chunk.clone()
        chunk[..., 6] = 0.0
        return {"action": chunk[:, : self.n_action_steps]}


def load_deployment_policy(
    checkpoint: Path,
    *,
    device: str,
    weights: str,
    policy_kind: str,
    task: str,
    selector: Path | None = None,
    prediction_mode: str = "full_then_truncate",
):
    checkpoint = checkpoint.expanduser().resolve()
    detected_kind = policy_kind
    config_path = checkpoint / "config.json"
    if detected_kind == "auto" and config_path.is_file():
        config = json.loads(config_path.read_text())
        config_type = config.get("type")
        detected_kind = config_type if config_type in {"smolvla", "pi05"} else "pushbox"
    elif detected_kind == "auto":
        detected_kind = "pushbox"
    if detected_kind == "smolvla":
        print(f"[runner] loading SmolVLA checkpoint: {checkpoint}")
        return SmolVLADeploymentPolicy(checkpoint, device, task)
    if detected_kind == "pi05":
        print(f"[runner] loading pi0.5 checkpoint: {checkpoint}")
        return PI05DeploymentPolicy(
            checkpoint,
            device,
            task,
            selector=selector,
            prediction_mode=prediction_mode,
        )

    from scripts.arp.policy_loader import load_policy

    return load_policy(
        str(checkpoint), device=device, weights=weights, use_checkpoint_config=True
    )


def wait_for_trackc_segment(
    streamer: Any,
    client: Any,
    robot_model: Any,
    target_T: np.ndarray,
    *,
    position_tolerance: float,
    rotation_tolerance: float,
    timeout: float,
    settle_samples: int,
    poll_hz: float,
    stop_requested: Any,
) -> dict[str, float | bool]:
    """Wait until TrackC has sent the segment, then sample its actual tracking error."""
    started = time.monotonic()
    deadline = started + timeout
    completed_samples = 0
    last_translation = float("inf")
    last_rotation = float("inf")
    while not stop_requested():
        with streamer.manager.lock:
            segment_completed = streamer.manager.completed
        state, info = client.get_latest_state(allow_stale=False)
        if state is not None:
            if state["arm_state"] == "ERROR":
                raise RuntimeError("remote controller reports arm ERROR during synchronous wait")
            current_T = client.get_tcp_pose_from_q(
                robot_model, state["q"], frame_name="panda_hand_tcp"
            )
            last_translation, last_rotation = cartesian_pose_error(current_T, target_T)
            completed_samples = completed_samples + 1 if segment_completed else 0
            if completed_samples >= settle_samples:
                return {
                    "stopped": False,
                    "elapsed": time.monotonic() - started,
                    "translation_error": last_translation,
                    "rotation_error": last_rotation,
                    "within_tolerance": (
                        last_translation <= position_tolerance
                        and last_rotation <= rotation_tolerance
                    ),
                }
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "synchronous TrackC segment wait timed out: "
                f"segment_completed={segment_completed}, robot_state={info}"
            )
        time.sleep(1.0 / poll_hz)
    return {
        "stopped": True,
        "elapsed": time.monotonic() - started,
        "translation_error": last_translation,
        "rotation_error": last_rotation,
        "within_tolerance": False,
    }


def wait_for_episode_enter(prompt: str, stop_requested: Any) -> bool:
    """Wait for one Enter key while still honoring SIGINT/SIGTERM."""
    if not sys.stdin.isatty():
        raise RuntimeError("interactive episodes require a terminal on stdin")
    terminal_fd = sys.stdin.fileno()
    # A pasted command or repeated Enter must not start/stop the next episode.
    termios.tcflush(terminal_fd, termios.TCIFLUSH)
    print(prompt, flush=True)
    while not stop_requested():
        ready, _, _ = select.select([terminal_fd], [], [], 0.1)
        if ready:
            # Use the descriptor consistently: TextIOWrapper.readline can read
            # ahead, hiding queued lines from select and terminal flushing.
            if os.read(terminal_fd, 4096) == b"":
                raise RuntimeError("terminal input closed during interactive episodes")
            termios.tcflush(terminal_fd, termios.TCIFLUSH)
            return True
    return False


def episode_end_requested() -> bool:
    """Consume one pending Enter key without blocking the control loop."""
    terminal_fd = sys.stdin.fileno()
    ready, _, _ = select.select([terminal_fd], [], [], 0.0)
    if not ready:
        return False
    if os.read(terminal_fd, 4096) == b"":
        raise RuntimeError("terminal input closed during interactive episodes")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy a Cartesian-delta ThreadingReal checkpoint via TrackC")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-kind", choices=("auto", "pushbox", "smolvla", "pi05"), default="auto")
    parser.add_argument(
        "--chunk-selector",
        type=Path,
        help="trained pi0.5 chunk selector sidecar; enables adaptive 4-10 step execution",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=("required_only", "full_then_truncate"),
        default="full_then_truncate",
        help="generate only selected steps, or generate 10 and execute the selected prefix",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="append one JSON record per inference cycle",
    )
    parser.add_argument("--task", default="pick up the block")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    # Lab topology documented in doc/old/THREADING_REAL.md: the controller
    # runs on the follower, while this workstation receives state over UDP.
    parser.add_argument("--server-url", default="http://10.157.175.22:8008/RPC2",
                        help="controller XML-RPC URL (default: http://10.157.175.22:8008/RPC2)")
    parser.add_argument("--server-ip", default="10.157.175.22",
                        help="TrackC command destination (default: 10.157.175.22)")
    parser.add_argument("--udp-ip", default="10.157.175.211",
                        help="local state receiver address (default: 10.157.175.211)")
    parser.add_argument("--udp-port", type=int, default=9000)
    parser.add_argument("--command-port", type=int, default=9200)
    parser.add_argument("--udp-frequency", type=int, default=500)
    parser.add_argument("--stream-hz", type=int, default=500)
    parser.add_argument(
        "--policy-hz",
        type=float,
        default=6.0,
        help="policy/control rate; stride-5 block-grasp checkpoints are trained at 6 Hz",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=1,
        help="number of predicted deltas executed before replanning; start real tests with 1",
    )
    parser.add_argument(
        "--execution-schedule", type=Path,
        help="MVT endpoint-based execution YAML; overrides fixed --execute-steps",
    )
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help=(
            "number of episodes to run without reloading the policy; every episode waits "
            "for Enter to start and can be ended with Enter"
        ),
    )
    parser.add_argument(
        "--synchronous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "wait until TrackC has completely sent the action chunk before the next "
            "observation/inference (default: enabled)"
        ),
    )
    parser.add_argument("--sync-position-tolerance", type=float, default=0.001)
    parser.add_argument("--sync-rotation-tolerance", type=float, default=0.02)
    parser.add_argument("--sync-timeout", type=float, default=2.0)
    parser.add_argument("--sync-settle-samples", type=int, default=3)
    parser.add_argument("--sync-poll-hz", type=float, default=50.0)
    parser.add_argument("--sideview-serial", default=DEFAULT_SIDEVIEW_SERIAL)
    parser.add_argument("--wrist-serial", default=DEFAULT_WRIST_SERIAL)
    parser.add_argument("--frontview-serial", default=DEFAULT_FRONTVIEW_SERIAL)
    parser.add_argument(
        "--pointcloud-calibration",
        type=Path,
        default=PROJECT_ROOT / "calibration" / "block_grasp_spatial.json",
    )
    parser.add_argument("--pointcloud-num-points", type=int, default=None,
                        help="defaults to the checkpoint's training point count")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="policy input size; defaults to the checkpoint image shape",
    )
    parser.add_argument(
        "--pre-resize-image-size",
        type=int,
        default=None,
        help="optionally downsample live RGB here before resizing to policy input size",
    )
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--state-timeout", type=float, default=2.0)
    parser.add_argument("--cartesian-stiffness", type=float, nargs=6, default=[200, 200, 200, 15, 15, 15])
    parser.add_argument("--nullspace-stiffness", type=float, default=1.0)
    # The policy runs on data downsampled from 30 Hz to 6 Hz, so each predicted
    # delta spans five source frames. Scale the former 30 Hz execution limits by
    # the same factor while retaining the workspace guard below.
    parser.add_argument("--max-first-translation", type=float, default=0.040)
    parser.add_argument("--max-step-translation", type=float, default=0.025)
    parser.add_argument("--max-first-rotation", type=float, default=0.40)
    parser.add_argument("--max-step-rotation", type=float, default=0.25)
    parser.add_argument("--abort-translation", type=float, default=0.15)
    parser.add_argument("--abort-rotation", type=float, default=1.50)
    parser.add_argument("--joint-stiffness", type=float, nargs=7, default=[200, 200, 200, 200, 100, 100, 50])
    parser.add_argument("--training-start-q", type=float, nargs=7, default=TRAINING_START_Q.tolist())
    parser.add_argument("--move-to-training-start", action="store_true")
    parser.add_argument("--recover-before-init", action="store_true")
    parser.add_argument("--gmm-eval-mode", choices=("map", "mean", "sample"), default="map")
    parser.add_argument("--gripper-close-threshold", type=float, default=0.035)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.055)
    parser.add_argument("--gripper-speed", type=float, default=0.05)
    parser.add_argument("--gripper-force", type=float, default=20.0)
    parser.add_argument("--gripper-epsilon", type=float, default=0.01)
    parser.add_argument(
        "--grasp-before-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "close the gripper at the current manually positioned pose before camera warmup "
            "during real execution (default: enabled)"
        ),
    )
    parser.add_argument(
        "--initial-grasp-width",
        type=float,
        default=0.02,
        help="target object width for the initial Franka grasp (must be > 0; default: 0.02 m)",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-real-robot", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    from remote_controller.RemoteControllerClient import RemoteControllerClient
    try:
        from remote_controller.robot_kinematics import RobotModel
    except ImportError as exc:
        raise ImportError("Cartesian deployment requires pinocchio; install it in pushbox with `conda install -c conda-forge pinocchio`") from exc
    if args.execute != args.confirm_real_robot:
        raise ValueError("real execution requires both --execute and --confirm-real-robot")
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if not 0.0 < args.initial_grasp_width <= 0.08:
        raise ValueError("--initial-grasp-width must be in (0, 0.08] meters")
    if args.policy_hz <= 0 or args.stream_hz <= 0:
        raise ValueError("--policy-hz and --stream-hz must be positive")
    if (
        args.sync_position_tolerance <= 0
        or args.sync_rotation_tolerance <= 0
        or args.sync_timeout <= 0
        or args.sync_settle_samples <= 0
        or args.sync_poll_hz <= 0
    ):
        raise ValueError("synchronous wait settings must be positive")
    workspace_min = workspace_max = None
    policy = load_deployment_policy(
        args.checkpoint,
        device=args.device,
        weights=args.weights,
        policy_kind=args.policy_kind,
        task=args.task,
        selector=args.chunk_selector,
        prediction_mode=args.prediction_mode,
    )
    if getattr(policy, "action_mode", None) != "cartesian_delta" or int(policy.action_dim) != 7:
        raise ValueError("checkpoint must be a 7D cartesian_delta policy")
    pointcloud_mode = bool(getattr(policy, "uses_pointcloud", False))
    rgb_keys = tuple(getattr(policy, "rgb_keys", ()))
    supported_rgb_keys = {
        ("sideview", "frontview"),
        ("sideview", "wrist", "frontview"),
    }
    if not pointcloud_mode and rgb_keys not in supported_rgb_keys:
        raise ValueError(f"unsupported checkpoint RGB inputs: {rgb_keys}")
    if not pointcloud_mode and args.image_size is None:
        args.image_size = int(policy.image_shape[-1])
    if not pointcloud_mode and args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.pre_resize_image_size is not None and args.pre_resize_image_size <= 0:
        raise ValueError("--pre-resize-image-size must be positive")
    if args.execute_steps <= 0 or args.execute_steps > int(policy.horizon):
        raise ValueError("--execute-steps must be in [1, checkpoint horizon]")
    adaptive_pi05 = isinstance(policy, PI05DeploymentPolicy) and policy.adaptive
    execution_schedule = None
    if args.execution_schedule is not None:
        if not getattr(policy, "uses_mvt", False):
            raise ValueError("--execution-schedule currently requires an MVT point-cloud policy")
        execution_schedule = EndpointExecutionSchedule.from_file(args.execution_schedule)
        if execution_schedule.coarse_steps > int(policy.horizon):
            raise ValueError("execution schedule coarse_steps exceeds checkpoint horizon")
    max_execution_steps = execution_schedule.coarse_steps if execution_schedule else args.execute_steps
    max_period = (policy.horizon if adaptive_pi05 else max_execution_steps) / args.policy_hz
    if args.synchronous and args.sync_timeout <= max_period:
        raise ValueError("--sync-timeout must exceed the maximum executed chunk / policy_hz")
    if not adaptive_pi05:
        policy.n_action_steps = max_execution_steps
    if hasattr(policy, "set_prediction_mode"):
        policy.set_prediction_mode(args.prediction_mode)
    if args.chunk_selector is not None and not adaptive_pi05:
        raise ValueError("--chunk-selector is only supported by a pi0.5 policy")
    if args.trace_output is not None:
        args.trace_output = args.trace_output.expanduser().resolve()
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(policy, (SmolVLADeploymentPolicy, PI05DeploymentPolicy)):
        if args.gmm_eval_mode != "map":
            raise ValueError("--gmm-eval-mode only applies to PushBox policies")
    elif args.gmm_eval_mode == "map":
        from threading_task.policy import enable_map_gmm_inference
        enable_map_gmm_inference(policy)
        policy.use_sample = "map"
    else:
        policy.use_sample = args.gmm_eval_mode == "sample"
    calibration = None
    pointcloud_rng = np.random.default_rng(42)
    pointcloud_bounds = None
    mvt_mode = bool(getattr(policy, "uses_mvt", False))
    if pointcloud_mode:
        if args.pointcloud_num_points is None:
            args.pointcloud_num_points = int(getattr(policy, "pointcloud_max_points", 4096))
        if args.pointcloud_num_points <= 0:
            raise ValueError("--pointcloud-num-points must be positive")
        calibration = json.loads(args.pointcloud_calibration.expanduser().read_text())
        expected = {
            "sideview": args.sideview_serial,
            "frontview": args.frontview_serial,
        }
        for key, serial in expected.items():
            calibrated = str(calibration["cameras"][key].get("serial", ""))
            if calibrated and calibrated != serial:
                raise ValueError(f"{key} calibration serial {calibrated} != live serial {serial}")
        bounds_tensor = policy.renderer.scene_bounds if mvt_mode else policy.point_bounds
        pointcloud_bounds = bounds_tensor.detach().cpu().numpy().reshape(-1)
    client = RemoteControllerClient(args.server_url, capacity=max(16, int(policy.n_obs_steps) + 2), horizon_prev=int(policy.n_obs_steps), sensor_size=29)
    cameras = streamer = None
    stop = False
    def request_stop(*_args: Any) -> None:
        nonlocal stop
        stop = True
    old_int, old_term = signal.signal(signal.SIGINT, request_stop), signal.signal(signal.SIGTERM, request_stop)
    try:
        result = client.init(udp_ip=args.udp_ip, udp_port=args.udp_port, udp_frequency_hz=args.udp_frequency, recover_before_init=args.recover_before_init)
        if result != 0: raise RuntimeError(f"remote controller init failed: {result}")
        state = client.wait_for_first_udp(timeout=args.state_timeout)
        if args.move_to_training_start:
            q_start = np.asarray(args.training_start_q, dtype=float)
            if q_start.shape != (7,) or np.any(q_start < PANDA_LOWER) or np.any(q_start > PANDA_UPPER): raise ValueError("invalid training-start-q")
            result = client.movej(q_start.tolist(), stiffness=diagonal_stiffness(args.joint_stiffness), dq_max=[.10]*7, ddq_max=[.20]*7, queue=False)
            if result != 0: raise RuntimeError(f"move to start failed: {result}")
            if client.wait_until_arm_moving_finished(timeout=30.0) != "IDLE": raise RuntimeError("move to start did not finish IDLE")
            client.gripper_release(args.gripper_speed, queue=True)
            state = client.wait_for_first_udp(timeout=args.state_timeout)
        robot_model = RobotModel()
        cameras = RealSenseRig(
            args.sideview_serial, args.wrist_serial, args.frontview_serial,
            fps=30, enable_depth=pointcloud_mode,
            enabled_views=("sideview", "frontview") if pointcloud_mode else rgb_keys,
        )
        samples = max(1, round(args.stream_hz / args.policy_hz))
        if args.execute:
            streamer = client.create_trackc_streamer(command_ip=args.server_ip, command_port=args.command_port, stream_hz=args.stream_hz, samples_per_segment=samples)
        execution_text = "selector[4,10]" if adaptive_pi05 else str(args.execute_steps)
        if execution_schedule:
            execution_text = f"endpoint[{execution_schedule.coarse_steps}->{execution_schedule.fine_steps}]"
            print(f"[runner] endpoint_xyz_m={execution_schedule.endpoint_xyz_m.tolist()} "
                  f"fine_radius_m={execution_schedule.fine_radius_m}; fine mode stays active until episode reset")
        print(
            f"[runner] policy loaded; mode={'EXECUTE' if args.execute else 'DRY-RUN'} "
            f"synchronous={args.synchronous} episodes={args.episodes} "
            f"n_obs_steps={policy.n_obs_steps} execute_steps={execution_text} "
            f"prediction_mode={args.prediction_mode} samples_per_segment={samples}"
        )

        for episode in range(1, args.episodes + 1):
            if stop:
                break
            if execution_schedule:
                execution_schedule.reset()
            prompt = (
                f"[episode {episode}/{args.episodes}] Model is loaded. Press Enter to start."
                if episode == 1
                else f"[episode {episode}/{args.episodes}] Reset the scene, then press Enter to start."
            )
            if not wait_for_episode_enter(prompt, lambda: stop):
                break

            reset_target = getattr(policy, "model", policy)
            if hasattr(reset_target, "reset"):
                reset_target.reset()
            if args.execute and args.grasp_before_inference:
                result = client.grasp(
                    args.initial_grasp_width,
                    args.gripper_speed,
                    args.gripper_force,
                    args.gripper_epsilon,
                    args.gripper_epsilon,
                    queue=False,
                )
                if result != 0:
                    raise RuntimeError(
                        f"initial grasp failed: {result} ({client.decode_rpc_result(result)})"
                    )
                gripper_state = client.wait_until_gripper_moving_finished(timeout=15.0)
                if gripper_state == "ERROR":
                    raise RuntimeError("gripper entered ERROR during initial grasp")
                print(
                    f"[gripper] episode={episode} initial grasp complete: "
                    f"state={gripper_state}, width={client.get_gripper_width():.4f} m"
                )

            gripper = GripperController(client, args, client.get_gripper_width())
            history: deque[Any] = deque(maxlen=int(policy.n_obs_steps))
            for _ in range(int(policy.n_obs_steps)):
                camera_data = cameras.read_rgbd(args.camera_timeout_ms) if pointcloud_mode else cameras.read(args.camera_timeout_ms)
                state, info = client.get_latest_state(allow_stale=False)
                if state is None: raise RuntimeError(f"robot state unavailable during warmup: {info}")
                T_observation = client.get_tcp_pose_from_q(robot_model, state["q"], frame_name="panda_hand_tcp")
                if pointcloud_mode:
                    history.append(make_live_pointcloud_observation(
                        camera_data, cameras, calibration, state["q"], client.get_gripper_width(),
                        T_observation, args.pointcloud_num_points, pointcloud_bounds, pointcloud_rng,
                        mvt_mode=mvt_mode, axis_length=float(getattr(policy, "axis_length", 0.04)),
                    ))
                else:
                    side, wrist, front = camera_data
                    history.append(make_observation(side, wrist, front, state["q"], client.get_gripper_width(), args.image_size, args.pre_resize_image_size, tcp_position=T_observation[:3, 3]))
                time.sleep(1 / args.policy_hz)

            if streamer is not None:
                T_start = client.get_tcp_pose_from_q(robot_model, state["q"], frame_name="panda_hand_tcp")
                streamer.start(T_start, cartesian_stiffness(args.cartesian_stiffness), args.nullspace_stiffness)
                print(f"[runner] episode={episode} TrackC UDP target={args.server_ip}:{args.command_port} stream_hz={args.stream_hz}")
            print(f"[episode {episode}/{args.episodes}] Running; press Enter to end this episode.")

            cycle = 0
            next_cycle = time.monotonic()
            episode_stop = False

            def episode_stop_requested() -> bool:
                nonlocal episode_stop
                if not episode_stop and episode_end_requested():
                    episode_stop = True
                    print(f"[episode {episode}/{args.episodes}] End requested.")
                return stop or episode_stop

            while not stop and not episode_stop:
                if episode_stop_requested():
                    break
                camera_data = cameras.read_rgbd(args.camera_timeout_ms) if pointcloud_mode else cameras.read(args.camera_timeout_ms)
                state, info = client.get_latest_state(allow_stale=False)
                if state is None: raise RuntimeError(f"fresh robot state unavailable: {info}")
                if state["arm_state"] == "ERROR": raise RuntimeError("remote controller reports arm ERROR")
                width = client.get_gripper_width()
                T_observation = client.get_tcp_pose_from_q(robot_model, state["q"], frame_name="panda_hand_tcp")
                if pointcloud_mode:
                    history.append(make_live_pointcloud_observation(
                        camera_data, cameras, calibration, state["q"], width, T_observation,
                        args.pointcloud_num_points, pointcloud_bounds, pointcloud_rng,
                        mvt_mode=mvt_mode, axis_length=float(getattr(policy, "axis_length", 0.04)),
                    ))
                    policy_obs = stack_pointcloud_observations(history, args.device)
                else:
                    side, wrist, front = camera_data
                    history.append(make_observation(side, wrist, front, state["q"], width, args.image_size, args.pre_resize_image_size, tcp_position=T_observation[:3, 3]))
                    policy_obs = stack_observations(history, args.device, rgb_keys=rgb_keys)
                with torch.inference_mode(): raw = policy.predict_action(policy_obs)["action"][0].detach().cpu().numpy()
                # Inference itself is synchronous, so consume an Enter pressed while it
                # was running before sending any newly predicted command to the robot.
                if episode_stop_requested():
                    break
                prediction_diagnostics = getattr(policy, "last_prediction_diagnostics", None)
                raw_translation = np.linalg.norm(raw[:, :3], axis=1)
                raw_rotation = np.linalg.norm(raw[:, 3:6], axis=1)
                max_translation_step = int(np.argmax(raw_translation))
                max_rotation_step = int(np.argmax(raw_rotation))
                if (
                    raw_translation[max_translation_step] > args.abort_translation
                    or raw_rotation[max_rotation_step] > args.abort_rotation
                ):
                    raise RuntimeError(
                        "unsafe raw Cartesian action in predicted chunk: "
                        f"translation={raw_translation[max_translation_step]:.4f} m "
                        f"at step {max_translation_step + 1}/{len(raw)}, "
                        f"rotation={raw_rotation[max_rotation_step]:.4f} rad "
                        f"at step {max_rotation_step + 1}/{len(raw)}; "
                        f"limits={args.abort_translation:.4f} m/{args.abort_rotation:.4f} rad"
                    )
                T_now = client.get_tcp_pose_from_q(robot_model, state["q"], frame_name="panda_hand_tcp")
                poses, widths, stats = integrate_cartesian_delta_chunk(raw, T_now, width, max_first_translation=args.max_first_translation, max_step_translation=args.max_step_translation, max_first_rotation=args.max_first_rotation, max_step_rotation=args.max_step_rotation, workspace_min=workspace_min, workspace_max=workspace_max)
                execution_diagnostics = {}
                if execution_schedule:
                    steps, execution_diagnostics = execution_schedule.select(T_now[:3, 3], poses)
                    execution_diagnostics["predicted_action_count"] = len(raw)
                    raw, poses, widths = raw[:steps], poses[:steps], widths[:steps]
                sync_result = None
                if streamer is not None:
                    streamer.update_waypoints(poses, merge_mode="replace")
                    gripper.update(float(widths[-1]))
                    if args.synchronous:
                        sync_result = wait_for_trackc_segment(
                            streamer,
                            client,
                            robot_model,
                            poses[-1],
                            position_tolerance=args.sync_position_tolerance,
                            rotation_tolerance=args.sync_rotation_tolerance,
                            timeout=args.sync_timeout,
                            settle_samples=args.sync_settle_samples,
                            poll_hz=args.sync_poll_hz,
                            stop_requested=episode_stop_requested,
                        )
                        if sync_result["stopped"]:
                            break
                cycle += 1
                if args.trace_output is not None:
                    trace_record = {
                        "episode": episode,
                        "cycle": cycle,
                        "timestamp": time.time(),
                        "executed": bool(args.execute),
                        "action_count": int(len(raw)),
                        **(prediction_diagnostics or {}),
                        **execution_diagnostics,
                    }
                    with args.trace_output.open("a") as trace_file:
                        trace_file.write(json.dumps(trace_record) + "\n")
                sync_text = ""
                if sync_result is not None:
                    sync_text = (
                        f" sync_time={sync_result['elapsed']:.3f}s"
                        f" sync_xyz_err={sync_result['translation_error'] * 1000:.2f}mm"
                        f" sync_rot_err={np.degrees(sync_result['rotation_error']):.2f}deg"
                        f" sync_target={sync_result['within_tolerance']}"
                    )
                spatial_text = ""
                diagnostics = getattr(policy, "last_spatial_diagnostics", None)
                if diagnostics:
                    goal = diagnostics["goal_xyz"][0].detach().cpu().numpy()
                    confidence = diagnostics["confidence"][0].detach().cpu().numpy()
                    trustworthy = bool(diagnostics["trustworthy"][0].item())
                    spatial_text = (
                        f" goal_xyz={np.round(goal, 4).tolist()}"
                        f" heatmap_conf={np.round(confidence, 4).tolist()}"
                        f" spatial_ok={trustworthy}"
                    )
                prediction_text = ""
                if prediction_diagnostics:
                    prediction_text = (
                        f" chunk={prediction_diagnostics['execution_chunk']}"
                        f" predicted={prediction_diagnostics['predicted_chunk']}"
                        f" soft={prediction_diagnostics['continuous_chunk']:.2f}"
                        f" infer={prediction_diagnostics['total_seconds'] * 1000:.1f}ms"
                        f" action={prediction_diagnostics['action_seconds'] * 1000:.1f}ms"
                    )
                execution_text = ""
                if execution_diagnostics:
                    execution_text = (f" phase={execution_diagnostics['execution_phase']}"
                                      f" execute_steps={len(raw)}"
                                      f" endpoint_distance={execution_diagnostics['endpoint_distance_m'] * 1000:.1f}mm")
                print(f"[runner] episode={episode} cycle={cycle} state_age={info['age']:.4f}s raw_dxyz={stats['raw_first_translation']*1000:.2f}mm safe_dxyz={stats['safe_first_translation']*1000:.2f}mm raw_drot={np.degrees(stats['raw_first_rotation']):.2f}deg gripper={widths[-1]:.4f}m{prediction_text}{spatial_text}{execution_text}{sync_text}")
                if episode_stop_requested():
                    break
                if args.max_cycles > 0 and cycle >= args.max_cycles:
                    print(f"[episode {episode}/{args.episodes}] Reached max_cycles={args.max_cycles}.")
                    break
                if args.synchronous:
                    continue
                period = len(raw) / args.policy_hz
                next_cycle += period; remaining = next_cycle-time.monotonic()
                if remaining > 0: time.sleep(remaining)
                else: print(f"[runner] warning: inference overran replan period by {-remaining:.3f}s"); next_cycle=time.monotonic()

            if streamer is not None and streamer.started:
                streamer.stop()
                arm_state = client.wait_until_arm_moving_finished(timeout=5.0)
                if arm_state != "IDLE":
                    raise RuntimeError(
                        f"arm did not become IDLE after stopping TrackC: {arm_state}"
                    )
                print(
                    f"[episode {episode}/{args.episodes}] TrackC stopped and arm is IDLE. "
                    "Use the Franka hand-guiding controls for manual reset; this runner "
                    "does not enable freedrive."
                )
            print(f"[episode {episode}/{args.episodes}] Finished after {cycle} cycles.")
        return 0
    finally:
        if streamer is not None: streamer.close()
        if cameras is not None: cameras.close()
        client.close(); signal.signal(signal.SIGINT, old_int); signal.signal(signal.SIGTERM, old_term)


def main() -> int:
    args = build_parser().parse_args()
    try: return run(args)
    except KeyboardInterrupt: return 130
    except Exception as exc: print(f"[runner] fatal: {exc}", file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
