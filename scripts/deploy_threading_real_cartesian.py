#!/usr/bin/env python3
"""Deploy a 7D base-frame TCP-delta ThreadingReal policy through TrackC."""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import signal
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT / "remote_controller" / "src"))

from scripts.deploy_threading_real import (  # noqa: E402
    DEFAULT_SIDEVIEW_SERIAL,
    DEFAULT_WRIST_SERIAL,
    PANDA_LOWER,
    PANDA_UPPER,
    TRAINING_START_Q,
    GripperController,
    RealSensePair,
    diagonal_stiffness,
    make_observation,
    stack_observations,
)
from scripts.eval_policy import load_policy  # noqa: E402


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy a Cartesian-delta ThreadingReal checkpoint via TrackC")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--server-url", default="http://localhost:8008/RPC2")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--udp-ip", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=9000)
    parser.add_argument("--command-port", type=int, default=9200)
    parser.add_argument("--udp-frequency", type=int, default=500)
    parser.add_argument("--stream-hz", type=int, default=500)
    parser.add_argument("--policy-hz", type=float, default=30.0)
    parser.add_argument("--execute-steps", type=int, default=20)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--sideview-serial", default=DEFAULT_SIDEVIEW_SERIAL)
    parser.add_argument("--wrist-serial", default=DEFAULT_WRIST_SERIAL)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--state-timeout", type=float, default=2.0)
    parser.add_argument("--cartesian-stiffness", type=float, nargs=6, default=[200, 200, 200, 15, 15, 15])
    parser.add_argument("--nullspace-stiffness", type=float, default=1.0)
    parser.add_argument("--max-first-translation", type=float, default=0.008)
    parser.add_argument("--max-step-translation", type=float, default=0.005)
    parser.add_argument("--max-first-rotation", type=float, default=0.08)
    parser.add_argument("--max-step-rotation", type=float, default=0.05)
    parser.add_argument("--abort-translation", type=float, default=0.03)
    parser.add_argument("--abort-rotation", type=float, default=0.30)
    parser.add_argument("--workspace-min", type=float, nargs=3)
    parser.add_argument("--workspace-max", type=float, nargs=3)
    parser.add_argument(
        "--allow-unbounded-workspace",
        action="store_true",
        help="explicitly allow real execution without TCP workspace bounds",
    )
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
    if (args.workspace_min is None) != (args.workspace_max is None):
        raise ValueError("provide both --workspace-min and --workspace-max, or neither")
    workspace_min = None if args.workspace_min is None else np.asarray(args.workspace_min, dtype=float)
    workspace_max = None if args.workspace_max is None else np.asarray(args.workspace_max, dtype=float)
    if workspace_min is not None and np.any(workspace_min >= workspace_max):
        raise ValueError("workspace-min must be strictly smaller than workspace-max")
    if args.execute and workspace_min is None and not args.allow_unbounded_workspace:
        raise ValueError(
            "real Cartesian execution requires --workspace-min/--workspace-max, "
            "or the explicit --allow-unbounded-workspace override"
        )
    if args.allow_unbounded_workspace and workspace_min is not None:
        raise ValueError("--allow-unbounded-workspace cannot be combined with workspace bounds")
    if args.execute and args.allow_unbounded_workspace:
        print("[runner] WARNING: executing without TCP workspace bounds")
    policy = load_policy(str(args.checkpoint.expanduser()), device=args.device, weights=args.weights, use_checkpoint_config=True)
    if getattr(policy, "action_mode", None) != "cartesian_delta" or int(policy.action_dim) != 7:
        raise ValueError("checkpoint must be a 7D cartesian_delta policy")
    if args.execute_steps <= 0 or args.execute_steps > int(policy.horizon):
        raise ValueError("--execute-steps must be in [1, checkpoint horizon]")
    policy.n_action_steps = args.execute_steps
    if hasattr(policy, "set_prediction_mode"):
        policy.set_prediction_mode("full_then_truncate")
    if args.gmm_eval_mode == "map":
        from threading_task.policy import enable_map_gmm_inference
        enable_map_gmm_inference(policy)
    else:
        policy.use_sample = args.gmm_eval_mode == "sample"
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
        cameras = RealSensePair(args.sideview_serial, args.wrist_serial, fps=30)
        gripper = GripperController(client, args, client.get_gripper_width())
        history: deque[Any] = deque(maxlen=int(policy.n_obs_steps))
        for _ in range(int(policy.n_obs_steps)):
            side, wrist = cameras.read(args.camera_timeout_ms); state, info = client.get_latest_state(allow_stale=False)
            if state is None: raise RuntimeError(f"robot state unavailable during warmup: {info}")
            history.append(make_observation(side, wrist, state["q"], client.get_gripper_width(), args.image_size)); time.sleep(1 / args.policy_hz)
        samples = max(1, round(args.stream_hz / args.policy_hz))
        T_start = client.get_tcp_pose_from_q(robot_model, state["q"], frame_name="panda_hand_tcp")
        if args.execute:
            streamer = client.create_trackc_streamer(command_ip=args.server_ip, command_port=args.command_port, stream_hz=args.stream_hz, samples_per_segment=samples)
            streamer.start(T_start, cartesian_stiffness(args.cartesian_stiffness), args.nullspace_stiffness)
            print(f"[runner] TrackC UDP target={args.server_ip}:{args.command_port} stream_hz={args.stream_hz}")
        print(f"[runner] mode={'EXECUTE' if args.execute else 'DRY-RUN'} n_obs_steps={policy.n_obs_steps} execute_steps={args.execute_steps} samples_per_segment={samples}")
        cycle = 0; next_cycle = time.monotonic(); period = args.execute_steps / args.policy_hz
        while not stop and (args.max_cycles == 0 or cycle < args.max_cycles):
            side, wrist = cameras.read(args.camera_timeout_ms); state, info = client.get_latest_state(allow_stale=False)
            if state is None: raise RuntimeError(f"fresh robot state unavailable: {info}")
            if state["arm_state"] == "ERROR": raise RuntimeError("remote controller reports arm ERROR")
            width = client.get_gripper_width(); history.append(make_observation(side, wrist, state["q"], width, args.image_size))
            with torch.inference_mode(): raw = policy.predict_action(stack_observations(history, args.device))["action"][0].detach().cpu().numpy()
            if (
                np.max(np.linalg.norm(raw[:, :3], axis=1)) > args.abort_translation
                or np.max(np.linalg.norm(raw[:, 3:6], axis=1)) > args.abort_rotation
            ):
                raise RuntimeError("unsafe raw Cartesian action in predicted chunk")
            T_now = client.get_tcp_pose_from_q(robot_model, state["q"], frame_name="panda_hand_tcp")
            poses, widths, stats = integrate_cartesian_delta_chunk(raw, T_now, width, max_first_translation=args.max_first_translation, max_step_translation=args.max_step_translation, max_first_rotation=args.max_first_rotation, max_step_rotation=args.max_step_rotation, workspace_min=workspace_min, workspace_max=workspace_max)
            if streamer is not None: streamer.update_waypoints(poses, merge_mode="replace"); gripper.update(float(widths[-1]))
            cycle += 1; print(f"[runner] cycle={cycle} state_age={info['age']:.4f}s raw_dxyz={stats['raw_first_translation']*1000:.2f}mm safe_dxyz={stats['safe_first_translation']*1000:.2f}mm raw_drot={np.degrees(stats['raw_first_rotation']):.2f}deg gripper={widths[-1]:.4f}m")
            next_cycle += period; remaining = next_cycle-time.monotonic()
            if remaining > 0: time.sleep(remaining)
            else: print(f"[runner] warning: inference overran replan period by {-remaining:.3f}s"); next_cycle=time.monotonic()
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
