#!/usr/bin/env python3
"""Fail-fast checks for the local pi0.5 dataset and training configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


CAMERAS = {
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_right",
}
ACTION_NAMES = ["dx", "dy", "dz", "drotvec_x", "drotvec_y", "drotvec_z", "dgripper"]
STATE_FEATURES = {
    "joint": ((8,), ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "gripper_width"]),
    "tcp_pose": ((8,), ["tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw", "gripper_width"]),
    "tcp_pose_6d": (
        (10,),
        [
            "tcp_x", "tcp_y", "tcp_z",
            "tcp_rotation_col0_x", "tcp_rotation_col0_y", "tcp_rotation_col0_z",
            "tcp_rotation_col1_x", "tcp_rotation_col1_y", "tcp_rotation_col1_z",
            "gripper_width",
        ],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        default="threading_real/threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d",
    )
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--expected-episodes", type=int, default=80)
    parser.add_argument("--expected-fps", type=int, default=15)
    parser.add_argument(
        "--state-representation",
        choices=tuple(STATE_FEATURES),
        default="tcp_pose_6d",
    )
    parser.add_argument(
        "--allow-dropped-frames",
        action="store_true",
        help="permit datasets whose preparation report records zero-action row removal",
    )
    args = parser.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root.expanduser().resolve(), video_backend="pyav")
    errors: list[str] = []
    if ds.fps != args.expected_fps:
        errors.append(f"fps must be {args.expected_fps}, got {ds.fps}")
    if set(ds.meta.camera_keys) != CAMERAS:
        errors.append(f"camera keys mismatch: {ds.meta.camera_keys}")
    if ds.num_episodes != args.expected_episodes:
        errors.append(f"expected {args.expected_episodes} episodes, got {ds.num_episodes}")
    state_shape, state_names = STATE_FEATURES[args.state_representation]
    state_feature = ds.meta.features["observation.state"]
    if tuple(state_feature["shape"]) != state_shape:
        errors.append(f"state shape must be {state_shape}, got {state_feature['shape']}")
    if state_feature.get("names") != state_names:
        errors.append(f"state names mismatch: {state_feature.get('names')}")
    if ds.meta.features["action"]["shape"] != (7,):
        errors.append(f"action shape must be (7,), got {ds.meta.features['action']['shape']}")
    if ds.meta.features["action"].get("names") != ACTION_NAMES:
        errors.append(f"action names mismatch: {ds.meta.features['action'].get('names')}")

    preparation_path = ds.root / "meta" / "pi05_preparation_report.json"
    preparation = None
    if preparation_path.is_file():
        preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
        removed_frames = int(preparation.get("removed_zero_actions", 0))
        dropped_frames = bool(preparation.get("drop_zero_actions", removed_frames > 0))
        if not args.allow_dropped_frames and (dropped_frames or removed_frames):
            errors.append(
                "preparation report records removed timeline rows: "
                f"drop_zero_actions={dropped_frames}, removed_zero_actions={removed_frames}"
            )
    elif not args.allow_dropped_frames:
        errors.append("missing pi05_preparation_report.json; cannot verify that all timeline rows were retained")
    for key in ("observation.state", "action"):
        missing = {"q01", "q99"} - set(ds.meta.stats[key])
        if missing:
            errors.append(f"{key} lacks quantile stats: {sorted(missing)}")
    shortest = min(int(ep["length"]) for ep in ds.meta.episodes)
    if shortest < args.chunk_size:
        errors.append(f"shortest episode ({shortest}) is shorter than chunk size ({args.chunk_size})")

    timeline = ds.hf_dataset.select_columns(
        ["episode_index", "frame_index", "timestamp", "index"]
    ).with_format("numpy")[:]
    episode_indices = np.asarray(timeline["episode_index"], dtype=np.int64)
    frame_indices = np.asarray(timeline["frame_index"], dtype=np.int64)
    timestamps = np.asarray(timeline["timestamp"], dtype=np.float64)
    global_indices = np.asarray(timeline["index"], dtype=np.int64)
    if not np.array_equal(global_indices, np.arange(ds.num_frames)):
        errors.append("global index is not contiguous from 0 to num_frames - 1")
    for episode in ds.meta.episodes:
        episode_index = int(episode["episode_index"])
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        length = stop - start
        if not np.all(episode_indices[start:stop] == episode_index):
            errors.append(f"episode {episode_index} rows are not contiguous in the data table")
        if not np.array_equal(frame_indices[start:stop], np.arange(length)):
            errors.append(f"episode {episode_index} frame_index is not contiguous from zero")
        expected_timestamps = np.arange(length, dtype=np.float64) / ds.fps
        if not np.allclose(timestamps[start:stop], expected_timestamps, rtol=0.0, atol=1e-5):
            max_error = float(np.max(np.abs(timestamps[start:stop] - expected_timestamps)))
            errors.append(
                f"episode {episode_index} timestamps are not fixed-rate {ds.fps} Hz "
                f"(max error {max_error:.3g}s)"
            )

    sample = ds[0]
    summary = {
        "dataset_root": str(ds.root),
        "episodes": ds.num_episodes,
        "frames": ds.num_frames,
        "fps": ds.fps,
        "cameras": ds.meta.camera_keys,
        "state_shape": list(sample["observation.state"].shape),
        "state_representation": args.state_representation,
        "state_names": state_feature.get("names"),
        "action_shape": list(sample["action"].shape),
        "task": sample["task"],
        "chunk_size": args.chunk_size,
        "chunk_seconds": args.chunk_size / ds.fps,
        "timeline_checked": True,
        "frame_removal_checked": preparation is not None,
        "drop_zero_actions": None if preparation is None else preparation.get("drop_zero_actions"),
        "removed_zero_actions": None if preparation is None else preparation.get("removed_zero_actions"),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
