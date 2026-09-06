#!/usr/bin/env python3
"""Run a trained ThreadingReal policy against one Franka follower.

The policy process talks to ``remote_controller_server`` over XML-RPC/UDP.
By default this script is observation-only: it opens the cameras, receives
robot state, runs inference, and prints the proposed action chunk. Real robot
commands require both ``--execute`` and ``--confirm-real-robot``.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import signal
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT / "remote_controller" / "src"))

from scripts.eval_policy import load_policy  # noqa: E402


PANDA_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float64,
)
PANDA_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float64,
)
DEFAULT_SIDEVIEW_SERIAL = "233722072293"
DEFAULT_WRIST_SERIAL = "233622071984"
# Default evaluation start captured from the follower on 2026-09-04.
# Pass --training-start-q explicitly to use a different safe start posture.
TRAINING_START_Q = np.array(
    [0.307272, 0.323924, -0.112529, -2.501686, -0.012559, 2.764401, 0.833281],
    dtype=np.float64,
)


def diagonal_stiffness(values: Sequence[float]) -> list[list[float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("joint stiffness must contain 7 finite non-negative values")
    return np.diag(values).tolist()


def image_to_policy_tensor(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    """Convert one HWC RGB frame to a float CHW policy tensor in [0, 1]."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected an HxWx3 RGB image, got {image.shape}")
    image = image[..., :3]
    if image.shape[:2] != (image_size, image_size):
        interpolation = cv2.INTER_AREA if max(image.shape[:2]) > image_size else cv2.INTER_LINEAR
        image = cv2.resize(image, (image_size, image_size), interpolation=interpolation)
    image = np.ascontiguousarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(np.moveaxis(image, -1, 0))


def sanitize_action_chunk(
    raw_actions: np.ndarray,
    current_q: Sequence[float],
    *,
    joint_limit_margin: float,
    max_first_delta: float,
    max_step_delta: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Validate and rate-limit an Nx8 ``[q, width]`` policy chunk.

    Joint positions outside the software limit are clipped. Each target is
    then clipped relative to the preceding safe target. The returned gripper
    predictions are only range-clipped; they are not sent by this function.
    """
    actions = np.asarray(raw_actions, dtype=np.float64)
    q_now = np.asarray(current_q, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] == 0:
        raise ValueError(f"expected a non-empty Nx8 action chunk, got {actions.shape}")
    if q_now.shape != (7,):
        raise ValueError(f"expected current_q shape (7,), got {q_now.shape}")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(q_now)):
        raise ValueError("action chunk and current_q must be finite")
    if not 0.0 <= joint_limit_margin < 0.2:
        raise ValueError("joint_limit_margin must be in [0, 0.2)")
    if max_first_delta <= 0.0 or max_step_delta <= 0.0:
        raise ValueError("joint delta limits must be positive")

    lower = PANDA_LOWER + joint_limit_margin
    upper = PANDA_UPPER - joint_limit_margin
    if np.any(q_now < lower) or np.any(q_now > upper):
        raise RuntimeError("current robot state is outside the configured software joint limits")

    predicted_q = actions[:, :7]
    bounded_q = np.clip(predicted_q, lower, upper)
    safe_q = np.empty_like(bounded_q)
    previous = q_now
    for index, target in enumerate(bounded_q):
        delta_limit = max_first_delta if index == 0 else max_step_delta
        safe_q[index] = previous + np.clip(target - previous, -delta_limit, delta_limit)
        previous = safe_q[index]

    widths = np.clip(actions[:, 7], 0.0, 0.08)
    stats = {
        "raw_first_delta": float(np.max(np.abs(predicted_q[0] - q_now))),
        # Keep this separate from current_q -> first target, which is checked
        # by raw_first_delta above. This metric only describes continuity
        # inside the predicted action chunk.
        "raw_step_delta": float(
            np.max(np.abs(np.diff(predicted_q, axis=0))) if len(predicted_q) > 1 else 0.0
        ),
        "applied_first_delta": float(np.max(np.abs(safe_q[0] - q_now))),
        "joint_limit_clip": float(np.max(np.abs(predicted_q - bounded_q))),
        "rate_limit_clip": float(np.max(np.abs(bounded_q - safe_q))),
    }
    return safe_q, widths, stats


def integrate_delta_actions(deltas: np.ndarray, current_state: Sequence[float]) -> np.ndarray:
    """Convert an Nx8 predicted delta chunk into absolute state targets."""
    deltas = np.asarray(deltas, dtype=np.float64)
    current_state = np.asarray(current_state, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[1] != 8:
        raise ValueError(f"expected Nx8 deltas, got {deltas.shape}")
    if current_state.shape != (8,):
        raise ValueError(f"expected current_state shape (8,), got {current_state.shape}")
    if not np.all(np.isfinite(deltas)) or not np.all(np.isfinite(current_state)):
        raise ValueError("deltas and current_state must be finite")
    return current_state[None, :] + np.cumsum(deltas, axis=0)


def choose_gripper_transition(
    current_mode: str,
    predicted_width: float,
    *,
    close_threshold: float,
    open_threshold: float,
) -> str | None:
    """Return ``close``/``open`` only when a hysteresis boundary is crossed."""
    if current_mode not in {"open", "closed"}:
        raise ValueError(f"unknown gripper mode: {current_mode}")
    if not 0.0 <= close_threshold < open_threshold <= 0.08:
        raise ValueError("gripper thresholds must satisfy 0 <= close < open <= 0.08")
    if current_mode == "open" and predicted_width <= close_threshold:
        return "close"
    if current_mode == "closed" and predicted_width >= open_threshold:
        return "open"
    return None


@dataclass
class ObservationFrame:
    sideview: torch.Tensor
    wrist: torch.Tensor
    agent_pos: torch.Tensor
    timestamp: float


class RealSensePair:
    """Two serial-pinned RealSense color streams matching data collection."""

    def __init__(
        self,
        sideview_serial: str,
        wrist_serial: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        if sideview_serial == wrist_serial:
            raise ValueError("sideview and wrist cameras must use different serial numbers")
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is required for live RealSense input; install the "
                "Intel RealSense Python bindings in the runner environment"
            ) from exc

        self.rs = rs
        self.pipelines: list[Any] = []
        try:
            for serial in (sideview_serial, wrist_serial):
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
                pipeline.start(config)
                self.pipelines.append(pipeline)
            # Discard auto-exposure startup frames.
            for _ in range(10):
                self.read(timeout_ms=2000)
        except Exception:
            self.close()
            raise

    def read(self, timeout_ms: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        frames = []
        for pipeline in self.pipelines:
            frameset = pipeline.wait_for_frames(timeout_ms)
            color = frameset.get_color_frame()
            if not color:
                raise RuntimeError("RealSense frameset has no color frame")
            frames.append(np.asanyarray(color.get_data()).copy())
        return frames[0], frames[1]

    def close(self) -> None:
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass
        self.pipelines.clear()


class GripperController:
    def __init__(self, client: Any, args: argparse.Namespace, current_width: float):
        self.client = client
        self.args = args
        midpoint = (args.gripper_close_threshold + args.gripper_open_threshold) / 2.0
        self.mode = "open" if current_width >= midpoint else "closed"

    def update(self, predicted_width: float) -> None:
        transition = choose_gripper_transition(
            self.mode,
            predicted_width,
            close_threshold=self.args.gripper_close_threshold,
            open_threshold=self.args.gripper_open_threshold,
        )
        if transition is None or self.client.get_gripper_state() == "MOVING":
            return
        if transition == "open":
            result = self.client.gripper_release(self.args.gripper_speed, queue=True)
        else:
            target = float(np.clip(predicted_width, 0.0, self.args.gripper_close_threshold))
            result = self.client.grasp(
                target,
                self.args.gripper_speed,
                self.args.gripper_force,
                self.args.gripper_epsilon,
                self.args.gripper_epsilon,
                queue=True,
            )
        if result != 0:
            raise RuntimeError(
                f"gripper {transition} failed: {result} ({self.client.decode_rpc_result(result)})"
            )
        self.mode = "closed" if transition == "close" else "open"
        print(f"[gripper] {transition}, predicted_width={predicted_width:.4f} m")


def make_observation(
    sideview_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    q: Sequence[float],
    gripper_width: float,
    image_size: int,
    pre_resize_image_size: int | None = None,
) -> ObservationFrame:
    state = np.concatenate(
        [np.asarray(q, dtype=np.float32), np.asarray([gripper_width], dtype=np.float32)]
    )
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError(f"invalid 8D robot state: {state}")
    if pre_resize_image_size is not None:
        pre_resize_image_size = int(pre_resize_image_size)
        if pre_resize_image_size <= 0:
            raise ValueError("pre_resize_image_size must be positive")
        sideview_rgb = cv2.resize(
            sideview_rgb,
            (pre_resize_image_size, pre_resize_image_size),
            interpolation=cv2.INTER_AREA,
        )
        wrist_rgb = cv2.resize(
            wrist_rgb,
            (pre_resize_image_size, pre_resize_image_size),
            interpolation=cv2.INTER_AREA,
        )
    return ObservationFrame(
        sideview=image_to_policy_tensor(sideview_rgb, image_size),
        wrist=image_to_policy_tensor(wrist_rgb, image_size),
        agent_pos=torch.from_numpy(state),
        timestamp=time.monotonic(),
    )


def stack_observations(
    history: Sequence[ObservationFrame], device: str
) -> dict[str, torch.Tensor]:
    if not history:
        raise ValueError("observation history is empty")
    return {
        "sideview": torch.stack([frame.sideview for frame in history]).unsqueeze(0).to(device),
        "wrist": torch.stack([frame.wrist for frame in history]).unsqueeze(0).to(device),
        "agent_pos": torch.stack([frame.agent_pos for frame in history]).unsqueeze(0).to(device),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy a ThreadingReal checkpoint on one follower via remote_controller"
    )
    parser.add_argument("checkpoint", type=Path, help="checkpoint .ckpt file or directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--server-url", default="http://localhost:8008/RPC2")
    parser.add_argument("--server-ip", default="127.0.0.1", help="TrackJ command destination")
    parser.add_argument("--udp-ip", default="127.0.0.1", help="local state receiver bind/address")
    parser.add_argument("--udp-port", type=int, default=9000)
    parser.add_argument("--command-port", type=int, default=9100)
    parser.add_argument("--udp-frequency", type=int, default=500)
    parser.add_argument("--stream-hz", type=int, default=500)
    parser.add_argument("--policy-hz", type=float, default=30.0)
    parser.add_argument("--execute-steps", type=int, default=5)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 runs until interrupted")
    parser.add_argument("--sideview-serial", default=DEFAULT_SIDEVIEW_SERIAL)
    parser.add_argument("--wrist-serial", default=DEFAULT_WRIST_SERIAL)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--camera-timeout-ms", type=int, default=1000)
    parser.add_argument("--state-timeout", type=float, default=2.0)
    parser.add_argument("--joint-limit-margin", type=float, default=0.05)
    parser.add_argument("--max-first-delta", type=float, default=0.05)
    parser.add_argument("--max-step-delta", type=float, default=0.025)
    parser.add_argument(
        "--abort-first-delta",
        type=float,
        default=0.15,
        help="stop if the raw first target differs from current q by more than this",
    )
    parser.add_argument(
        "--abort-step-delta",
        type=float,
        default=0.15,
        help="stop if adjacent raw policy targets differ by more than this",
    )
    parser.add_argument(
        "--gmm-eval-mode",
        choices=("map", "mean", "sample"),
        default="map",
        help="MAP is deterministic and is the default for hardware deployment",
    )
    parser.add_argument(
        "--stiffness",
        type=float,
        nargs=7,
        default=[200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 50.0],
        metavar=("K1", "K2", "K3", "K4", "K5", "K6", "K7"),
    )
    parser.add_argument("--gripper-close-threshold", type=float, default=0.035)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.055)
    parser.add_argument("--gripper-speed", type=float, default=0.05)
    parser.add_argument("--gripper-force", type=float, default=20.0)
    parser.add_argument("--gripper-epsilon", type=float, default=0.01)
    parser.add_argument(
        "--move-to-training-start",
        action="store_true",
        help="after explicit execution confirmation, move follower to recorded start q",
    )
    parser.add_argument(
        "--training-start-q",
        type=float,
        nargs=7,
        default=TRAINING_START_Q.tolist(),
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help="override the recorded training start posture (radians)",
    )
    parser.add_argument("--recover-before-init", action="store_true")
    parser.add_argument("--execute", action="store_true", help="enable arm/gripper commands")
    parser.add_argument(
        "--confirm-real-robot",
        action="store_true",
        help="second explicit acknowledgement required together with --execute",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.execute != args.confirm_real_robot:
        parser.error("real execution requires both --execute and --confirm-real-robot")
    if args.policy_hz <= 0 or args.stream_hz <= 0 or args.udp_frequency <= 0:
        parser.error("control and UDP frequencies must be positive")
    if args.execute_steps <= 0:
        parser.error("--execute-steps must be positive")
    if args.move_to_training_start and not args.execute:
        parser.error("--move-to-training-start requires --execute --confirm-real-robot")
    if args.image_size <= 0 or args.camera_timeout_ms <= 0 or args.state_timeout <= 0:
        parser.error("image size and timeouts must be positive")
    if args.abort_first_delta <= 0 or args.abort_step_delta <= 0:
        parser.error("raw-action abort thresholds must be positive")
    if args.max_cycles < 0:
        parser.error("--max-cycles cannot be negative")
    if not 0.0 < args.gripper_speed <= 0.1:
        parser.error("--gripper-speed must be in (0, 0.1]")
    if not 0.0 < args.gripper_force <= 70.0:
        parser.error("--gripper-force must be in (0, 70]")
    try:
        diagonal_stiffness(args.stiffness)
        start_q = np.asarray(args.training_start_q, dtype=np.float64)
        if start_q.shape != (7,) or not np.all(np.isfinite(start_q)):
            raise ValueError("training-start-q must contain 7 finite values")
        if np.any(start_q < PANDA_LOWER + args.joint_limit_margin) or np.any(
            start_q > PANDA_UPPER - args.joint_limit_margin
        ):
            raise ValueError("training-start-q is outside the configured software limits")
        choose_gripper_transition(
            "open",
            0.04,
            close_threshold=args.gripper_close_threshold,
            open_threshold=args.gripper_open_threshold,
        )
    except ValueError as exc:
        parser.error(str(exc))


def run(args: argparse.Namespace) -> int:
    try:
        # Import the client module directly. Package-level ``remote_controller``
        # also imports the optional Pinocchio kinematics helper, which this
        # deployment runner does not need.
        from remote_controller.RemoteControllerClient import RemoteControllerClient
    except ImportError as exc:
        raise ImportError(
            "remote_controller is unavailable; run `python -m pip install -e "
            "../remote_controller` in the policy environment"
        ) from exc

    policy = load_policy(
        str(args.checkpoint.expanduser()),
        device=args.device,
        weights=args.weights,
        use_checkpoint_config=True,
    )
    if int(getattr(policy, "agent_state_dim", -1)) != 8:
        raise ValueError("checkpoint must expect 8D [q1..q7, gripper_width] state")
    if int(getattr(policy, "action_dim", -1)) != 8:
        raise ValueError("checkpoint must predict 8D [q1..q7, gripper_width] actions")
    n_obs_steps = int(policy.n_obs_steps)
    if args.execute_steps > int(policy.horizon):
        raise ValueError("--execute-steps cannot exceed the checkpoint horizon")
    policy.n_action_steps = args.execute_steps
    if hasattr(policy, "set_prediction_mode"):
        policy.set_prediction_mode("full_then_truncate")
    if args.gmm_eval_mode == "map":
        from threading_task.policy import enable_map_gmm_inference

        enable_map_gmm_inference(policy)
    else:
        policy.use_sample = args.gmm_eval_mode == "sample"

    client = RemoteControllerClient(
        args.server_url,
        capacity=max(16, n_obs_steps + 2),
        horizon_prev=n_obs_steps,
        sensor_size=29,
    )
    cameras: RealSensePair | None = None
    streamer: Any | None = None
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        result = client.init(
            udp_ip=args.udp_ip,
            udp_port=args.udp_port,
            udp_frequency_hz=args.udp_frequency,
            recover_before_init=args.recover_before_init,
        )
        if result != 0:
            raise RuntimeError(
                f"remote controller init failed: {result} ({client.decode_rpc_result(result)})"
            )
        state = client.wait_for_first_udp(timeout=args.state_timeout)
        if args.move_to_training_start:
            target_q = np.asarray(args.training_start_q, dtype=np.float64)
            print(f"[runner] moving to training start q={target_q.round(4).tolist()}")
            result = client.movej(
                target_q.tolist(),
                stiffness=diagonal_stiffness(args.stiffness),
                dq_max=[0.10] * 7,
                ddq_max=[0.20] * 7,
                queue=False,
            )
            if result != 0:
                raise RuntimeError(
                    f"move to training start failed: {result} ({client.decode_rpc_result(result)})"
                )
            final_arm_state = client.wait_until_arm_moving_finished(timeout=30.0)
            if final_arm_state != "IDLE":
                raise RuntimeError(f"training-start move ended in arm state {final_arm_state}")
            result = client.gripper_release(args.gripper_speed, queue=True)
            if result != 0:
                raise RuntimeError(
                    f"opening gripper failed: {result} ({client.decode_rpc_result(result)})"
                )
            final_gripper_state = client.wait_until_gripper_moving_finished(timeout=15.0)
            if final_gripper_state not in {"IDLE", "HOLDING"}:
                raise RuntimeError(
                    f"opening gripper ended in state {final_gripper_state}"
                )
            state = client.wait_for_first_udp(timeout=args.state_timeout)
        cameras = RealSensePair(args.sideview_serial, args.wrist_serial, fps=30)
        current_width = client.get_gripper_width()
        gripper = GripperController(client, args, current_width)
        history: deque[ObservationFrame] = deque(maxlen=n_obs_steps)

        warmup_period = 1.0 / args.policy_hz
        for _ in range(n_obs_steps):
            sideview, wrist = cameras.read(timeout_ms=args.camera_timeout_ms)
            state, info = client.get_latest_state(allow_stale=False)
            if state is None:
                raise RuntimeError(f"robot state unavailable during warmup: {info}")
            current_width = client.get_gripper_width()
            history.append(
                make_observation(sideview, wrist, state["q"], current_width, args.image_size)
            )
            time.sleep(warmup_period)

        samples_per_segment = max(1, round(args.stream_hz / args.policy_hz))
        if args.execute:
            streamer = client.create_trackj_streamer(
                command_ip=args.server_ip,
                command_port=args.command_port,
                stream_hz=args.stream_hz,
                samples_per_segment=samples_per_segment,
            )
            streamer.start(state["q"], diagonal_stiffness(args.stiffness))
            print(
                f"[runner] TrackJ UDP target={args.server_ip}:{args.command_port} "
                f"stream_hz={args.stream_hz}"
            )
            mode = "EXECUTE"
        else:
            mode = "DRY-RUN"
        print(
            f"[runner] mode={mode} n_obs_steps={n_obs_steps} "
            f"execute_steps={args.execute_steps} samples_per_segment={samples_per_segment}"
        )

        cycle = 0
        replan_period = args.execute_steps / args.policy_hz
        next_cycle = time.monotonic()
        while not stop_requested and (args.max_cycles == 0 or cycle < args.max_cycles):
            sideview, wrist = cameras.read(timeout_ms=args.camera_timeout_ms)
            state, info = client.get_latest_state(allow_stale=False)
            if state is None:
                raise RuntimeError(f"fresh robot state unavailable: {info}")
            if state["arm_state"] == "ERROR":
                raise RuntimeError("remote controller reports arm ERROR")
            current_width = client.get_gripper_width()
            history.append(
                make_observation(sideview, wrist, state["q"], current_width, args.image_size)
            )
            obs = stack_observations(history, args.device)
            with torch.inference_mode():
                prediction = policy.predict_action(obs)["action"]
            raw_actions = prediction[0].detach().cpu().numpy()
            if getattr(policy, "action_mode", "absolute") == "delta":
                raw_actions = integrate_delta_actions(
                    raw_actions,
                    np.r_[np.asarray(state["q"], dtype=np.float64), current_width],
                )
            safe_q, widths, stats = sanitize_action_chunk(
                raw_actions,
                state["q"],
                joint_limit_margin=args.joint_limit_margin,
                max_first_delta=args.max_first_delta,
                max_step_delta=args.max_step_delta,
            )
            if stats["raw_first_delta"] > args.abort_first_delta:
                raise RuntimeError(
                    "unsafe raw first target: "
                    f"{stats['raw_first_delta']:.4f} rad exceeds "
                    f"{args.abort_first_delta:.4f} rad; "
                    f"current_q={np.asarray(state['q']).round(4).tolist()}, "
                    f"raw_q={np.asarray(raw_actions[0, :7]).round(4).tolist()}"
                )
            if stats["raw_step_delta"] > args.abort_step_delta:
                raise RuntimeError(
                    "unsafe raw action step: "
                    f"{stats['raw_step_delta']:.4f} rad exceeds "
                    f"{args.abort_step_delta:.4f} rad; "
                    f"raw_q0={np.asarray(raw_actions[0, :7]).round(4).tolist()}, "
                    f"raw_q1={np.asarray(raw_actions[min(1, len(raw_actions) - 1), :7]).round(4).tolist()}"
                )
            if streamer is not None:
                streamer.update_waypoints(safe_q.tolist(), merge_mode="replace")
                gripper.update(float(widths[-1]))
            cycle += 1
            print(
                f"[runner] cycle={cycle} state_age={info['age']:.4f}s "
                f"raw_dq={stats['raw_first_delta']:.4f} "
                f"safe_dq={stats['applied_first_delta']:.4f} "
                f"gripper={widths[-1]:.4f}m"
            )
            next_cycle += replan_period
            remaining = next_cycle - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                print(f"[runner] warning: inference overran replan period by {-remaining:.3f}s")
                next_cycle = time.monotonic()
        return 0
    finally:
        if streamer is not None:
            try:
                streamer.close()
            except Exception as exc:
                print(f"[runner] warning: TrackJ stop failed: {exc}", file=sys.stderr)
        if cameras is not None:
            cameras.close()
        client.close()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[runner] fatal: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
