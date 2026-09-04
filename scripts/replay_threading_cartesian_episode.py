#!/usr/bin/env python3
"""Directly replay one recorded Cartesian-delta LeRobot episode through TrackC."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent / "remote_controller" / "src"))
from scripts.deploy_threading_real import GripperController, PANDA_LOWER, PANDA_UPPER, diagonal_stiffness  # noqa: E402
from scripts.deploy_threading_real_cartesian import cartesian_stiffness  # noqa: E402


def load_episode(dataset: Path, episode_index: int, start_frame: int, max_frames: int) -> tuple[np.ndarray, np.ndarray, float]:
    import json
    import pyarrow.parquet as pq
    files = sorted((dataset / "data").rglob("*.parquet"))
    if not files: raise FileNotFoundError(f"no data parquet files under {dataset}")
    tables = [pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"]) for path in files]
    import pyarrow as pa
    table = pa.concat_tables(tables)
    episodes = np.asarray(table.column("episode_index"), dtype=np.int64)
    frames = np.asarray(table.column("frame_index"), dtype=np.int64)
    rows = np.flatnonzero(episodes == episode_index)
    if not len(rows): raise ValueError(f"episode {episode_index} is not in {dataset}")
    rows = rows[np.argsort(frames[rows], kind="stable")]
    rows = rows[frames[rows] >= start_frame]
    if max_frames > 0: rows = rows[:max_frames]
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)[rows]
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float64)[rows]
    if states.shape[1] != 8 or actions.shape[1] != 7: raise ValueError(f"expected 8D state and 7D Cartesian action, got {states.shape}, {actions.shape}")
    fps = float(json.loads((dataset / "meta" / "info.json").read_text())["fps"])
    return states, actions, fps


def integrate(T0: np.ndarray, actions: np.ndarray, max_translation: float, max_rotation: float) -> list[np.ndarray]:
    T = np.asarray(T0, dtype=float).copy(); poses = []
    for i, action in enumerate(actions):
        if np.linalg.norm(action[:3]) > max_translation or np.linalg.norm(action[3:6]) > max_rotation:
            raise RuntimeError(f"recorded action at replay frame {i} exceeds safety bound")
        T = T.copy(); T[:3, 3] += action[:3]; T[:3, :3] = Rotation.from_rotvec(action[3:6]).as_matrix() @ T[:3, :3]
        poses.append(T)
    return poses


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT.parent / "data" / "threading_lerobot_v3_cartesian")
    p.add_argument("--episode-index", type=int, required=True); p.add_argument("--start-frame", type=int, default=0); p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--server-url", default="http://10.157.175.22:8008/RPC2"); p.add_argument("--server-ip", default="10.157.175.22"); p.add_argument("--udp-ip", default="10.157.175.211")
    p.add_argument("--udp-port", type=int, default=9000); p.add_argument("--command-port", type=int, default=9200); p.add_argument("--stream-hz", type=int, default=500); p.add_argument("--replay-hz", type=float, default=0.0)
    p.add_argument("--max-translation", type=float, default=0.008); p.add_argument("--max-rotation", type=float, default=0.08)
    p.add_argument("--joint-stiffness", type=float, nargs=7, default=[200,200,200,200,100,100,50]); p.add_argument("--cartesian-stiffness", type=float, nargs=6, default=[200,200,200,15,15,15])
    p.add_argument("--execute", action="store_true"); p.add_argument("--confirm-real-robot", action="store_true"); args = p.parse_args()
    if args.execute != args.confirm_real_robot: p.error("real replay requires --execute --confirm-real-robot")
    states, actions, fps = load_episode(args.dataset.resolve(), args.episode_index, args.start_frame, args.max_frames)
    replay_hz = args.replay_hz or fps
    if replay_hz <= 0 or args.stream_hz <= 0: p.error("frequencies must be positive")
    from remote_controller.RemoteControllerClient import RemoteControllerClient
    try: from remote_controller.robot_kinematics import RobotModel
    except ImportError as exc: raise RuntimeError("replay requires pinocchio in the pushbox environment") from exc
    client = RemoteControllerClient(args.server_url, capacity=32, horizon_prev=2, sensor_size=29); streamer = None
    try:
        if client.init(udp_ip=args.udp_ip, udp_port=args.udp_port, udp_frequency_hz=500, recover_before_init=False) != 0: raise RuntimeError("initSession failed")
        client.wait_for_first_udp(timeout=2.0)
        q0 = states[0, :7]
        if np.any(q0 < PANDA_LOWER) or np.any(q0 > PANDA_UPPER): raise RuntimeError("recorded start q violates limits")
        if args.execute:
            print(f"[replay] moving to episode={args.episode_index} frame={args.start_frame} q={q0.round(4).tolist()}")
            if client.movej(q0.tolist(), stiffness=diagonal_stiffness(args.joint_stiffness), dq_max=[.10]*7, ddq_max=[.20]*7, queue=False) != 0: raise RuntimeError("move to recorded start failed")
            if client.wait_until_arm_moving_finished(timeout=30.0) != "IDLE": raise RuntimeError("move to recorded start did not finish IDLE")
        state = client.wait_for_first_udp(timeout=2.0); T0 = client.get_tcp_pose_from_q(RobotModel(), state["q"], frame_name="panda_hand_tcp")
        poses = integrate(T0, actions, args.max_translation, args.max_rotation)
        print(f"[replay] episode={args.episode_index} frames={len(poses)} fps={replay_hz:.2f} duration={len(poses)/replay_hz:.2f}s mode={'EXECUTE' if args.execute else 'DRY-RUN'}")
        if not args.execute: return 0
        streamer = client.create_trackc_streamer(command_ip=args.server_ip, command_port=args.command_port, stream_hz=args.stream_hz, samples_per_segment=max(1, round(args.stream_hz/replay_hz)))
        streamer.start(T0, cartesian_stiffness(args.cartesian_stiffness)); streamer.update_waypoints(poses, merge_mode="replace")
        gripper_args = SimpleNamespace(gripper_close_threshold=.035, gripper_open_threshold=.055, gripper_speed=.05, gripper_force=20., gripper_epsilon=.01)
        gripper = GripperController(client, gripper_args, client.get_gripper_width())
        width = float(states[0, 7]); gripper.update(width)
        replay_start = time.monotonic()
        for index, action in enumerate(actions):
            width = float(np.clip(width + action[6], 0.0, .08))
            target_time = replay_start + (index + 1) / replay_hz
            time.sleep(max(0.0, target_time - time.monotonic()))
            gripper.update(width)
        time.sleep(.5); return 0
    finally:
        if streamer is not None: streamer.close()
        client.close()

if __name__ == "__main__": raise SystemExit(main())
