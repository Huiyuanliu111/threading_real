#!/usr/bin/env python3
"""Raw recordings -> OpenPI-compatible v2.1; full images, no trajectory smoothing."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import io
import json
from pathlib import Path
import sys

import cv2
import datasets
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from convert_vla_to_lerobot_v3 import (
    SequentialVideoReader, nearest_timestamp_indices, read_camera_timestamp_csv, read_follower_matrix,
)
from convert_lerobot_v3_to_cartesian import UrdfForwardKinematics
from export_dataset import statistics, write_episode_parquet
from transforms import ACTION_NAMES, FRONT, SIDE, STATE_NAMES


def letterbox(rgb, size=224):
    height, width = rgb.shape[:2]
    ratio = size / max(height, width)
    h, w = round(height * ratio), round(width * ratio)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    output = np.zeros((size, size, 3), dtype=np.uint8)
    output[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = resized
    return output


def make_features():
    features = {
        "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
        **{key: {"dtype": "image", "shape": [224, 224, 3], "names": ["height", "width", "channels"]}
           for key in (SIDE, FRONT)},
        **{key: {"dtype": dtype, "shape": [1], "names": None} for key, dtype in [
            ("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"),
            ("index", "int64"), ("task_index", "int64")]},
    }
    hf = {key: datasets.Image() if feature["dtype"] == "image" else
          datasets.Value(feature["dtype"]) if feature["shape"] == [1] else
          datasets.Sequence(datasets.Value(feature["dtype"]), length=feature["shape"][0])
          for key, feature in features.items()}
    return features, hf


def planned_length(robot_path):
    matrix = read_follower_matrix(robot_path)
    if matrix.shape[1] != 30:
        raise ValueError(f"Host timestamps required: {robot_path}")
    timestamps = [read_camera_timestamp_csv(robot_path.parent / f"{cam}_timestamps.csv", 2**63-1)
                  for cam in ("cam1", "cam3")]
    _, robot_errors = nearest_timestamp_indices(matrix[:, 0].astype(np.int64), timestamps[0])
    _, camera_errors = nearest_timestamp_indices(timestamps[1], timestamps[0])
    valid = np.flatnonzero((robot_errors <= 5_000_000) & (camera_errors <= 25_000_000))
    if len(valid) < 3 or np.any(np.diff(valid) != 1):
        raise ValueError(f"Alignment gaps or too few frames: {robot_path}")
    return len(valid)-1


def build_episode(job):
    ep, robot_path, total_frames, output, urdf, task = job
    fk = UrdfForwardKinematics(urdf)
    features, hf = make_features()
    meta = output / "meta"
    trial = robot_path.parent
    manifest = json.loads((trial / "recording_manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError(f"Incomplete recording: {trial}")
    matrix = read_follower_matrix(robot_path)
    if matrix.shape[1] != 30:
        raise ValueError(f"Host timestamped robot recording required: {trial}")
    captures = [cv2.VideoCapture(str(trial / f"{cam}.mp4")) for cam in ("cam1", "cam3")]
    try:
        if not all(cap.isOpened() for cap in captures):
            raise ValueError(f"Cannot open videos: {trial}")
        counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]
        timestamps = [read_camera_timestamp_csv(trial / f"{cam}_timestamps.csv", count)
                      for cam, count in zip(("cam1", "cam3"), counts, strict=True)]
        if any(len(ts) != count for ts, count in zip(timestamps, counts, strict=True)):
            raise ValueError(f"Video/timestamp count mismatch: {trial}")
        robot_rows, robot_errors = nearest_timestamp_indices(matrix[:, 0].astype(np.int64), timestamps[0])
        side_rows, camera_errors = nearest_timestamp_indices(timestamps[1], timestamps[0])
        valid = (robot_errors <= 5_000_000) & (camera_errors <= 25_000_000)
        valid_rows = np.flatnonzero(valid)
        if len(valid_rows) < 3 or np.any(np.diff(valid_rows) != 1):
            raise ValueError(f"Alignment has interior gaps or too few frames: {trial}; refusing to bridge gaps")
        # Retain every cam1 frame on the contiguous valid interval (no decimation).
        selected = valid_rows
        current, future = selected[:-1], selected[1:]
        real_dt = np.diff(timestamps[0][selected]) / 1e9
        if np.any((real_dt < 0.02) | (real_dt > 0.05)):
            raise ValueError(f"Unexpected 30 Hz interval in {trial}; refusing to hide dropped frames")
        selected_robot = robot_rows[selected]
        # Raw layout: host_time, teleop_active, measured q[7], measured width, ...
        joints = matrix[selected_robot, 2:9]
        positions, rotations = fk.poses(joints)
        states = np.concatenate((positions, rotations[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)), axis=1)
        actions = np.concatenate((np.diff(positions, axis=0),
            Rotation.from_matrix(rotations[1:] @ rotations[:-1].transpose(0, 2, 1)).as_rotvec()), axis=1)
        readers = [SequentialVideoReader(cap, trial / f"{cam}.mp4")
                   for cap, cam in zip(captures, ("cam1", "cam3"), strict=True)]
        rows, values = [], {key: [] for key in features}
        source_sizes = {}
        for frame, raw_frame in enumerate(current):
            row = {"observation.state": states[frame].astype(np.float32),
                   "action": actions[frame].astype(np.float32), "timestamp": np.float32(frame / 30),
                   "frame_index": frame, "episode_index": ep, "index": total_frames + frame, "task_index": 0}
            for reader, source_frame, key in zip(readers, (raw_frame, side_rows[raw_frame]), (SIDE, FRONT), strict=True):
                rgb = cv2.cvtColor(reader.read(int(source_frame)), cv2.COLOR_BGR2RGB)
                source_sizes[key] = list(rgb.shape)
                rgb = letterbox(rgb)
                values[key].append(rgb)
                buffer = io.BytesIO()
                Image.fromarray(rgb).save(buffer, format="PNG")
                row[key] = {"bytes": buffer.getvalue(), "path": None}
            for key in features:
                if key not in (SIDE, FRONT):
                    values[key].append(row[key])
            rows.append(row)
        path = output / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_episode_parquet(rows, hf, path)
        episode = {"episode_index": ep, "tasks": [task], "length": len(rows)}
        episode_stats = {"episode_index": ep, "stats": {key: statistics(value, key in (SIDE, FRONT))
                                                     for key, value in values.items()}}
        # Preserve exact source associations and real time intervals separately
        # from LeRobot's regular 30 Hz indexing timestamps.
        np.savez_compressed(meta / "alignment" / f"episode_{ep:06d}.npz",
            cam1_frame=current, cam3_frame=side_rows[current], robot_row=robot_rows[current],
            future_cam1_frame=future, future_robot_row=robot_rows[future],
            host_timestamp_ns=timestamps[0][current], action_dt_seconds=real_dt,
            robot_skew_ns=robot_errors[current], camera_skew_ns=camera_errors[current])
        report = {"episode_index": ep, "source": str(trial), "raw_video_frames": counts,
            "frames": len(rows), "source_image_shapes": source_sizes,
            "alignment_rejected_boundary_frames": int((~valid).sum()),
            "valid_aligned_frames": len(valid_rows), "selected_pose_samples": len(selected),
            "decimated_frames": len(valid_rows) - len(selected), "terminal_pose_without_action": 1,
            "robot_skew_max_ms": float(robot_errors[selected].max()/1e6),
            "camera_skew_max_ms": float(camera_errors[selected].max()/1e6),
            "action_dt_min_max_seconds": [float(real_dt.min()), float(real_dt.max())]}
        return episode, episode_stats, report
    finally:
        for capture in captures:
            capture.release()


def build(args):
    if args.workers < 1:
        raise ValueError("workers must be positive")
    trials = sorted(args.raw_root.resolve().glob("threading_new_*/episode_*/DATA_follower.m"))
    if len(trials) != args.expected_episodes:
        raise ValueError(f"Expected {args.expected_episodes} episodes, found {len(trials)}")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    urdf = ROOT / "remote_controller/src/remote_controller/assets/panda/panda_arm.urdf"
    features, hf = make_features()
    meta = output / "meta"
    meta.mkdir(parents=True)
    (meta / "alignment").mkdir()
    episodes, stats, reports, total_frames = [], [], [], 0
    jobs, expected_lengths = [], []
    for ep, robot_path in enumerate(trials):
        length = planned_length(robot_path)
        expected_lengths.append(length)
        jobs.append((ep, robot_path, total_frames, output, urdf, args.task))
        total_frames += length
    # Workers write separate episode files. Global row indices are planned
    # beforehand, so results do not depend on worker scheduling.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for ep, (episode, episode_stats, report) in enumerate(pool.map(build_episode, jobs)):
            if episode["length"] != expected_lengths[ep]:
                raise ValueError(f"Source changed during build: {trials[ep]}")
            episodes.append(episode)
            stats.append(episode_stats)
            reports.append(report)
            (meta / "build_progress.json").write_text(json.dumps(reports, indent=2))
            print(f"[{ep+1}/{len(trials)}] {Path(report['source']).parent.name}/{Path(report['source']).name}: "
                  f"{episode['length']} frames", flush=True)
    for name, records in [("episodes", episodes), ("episodes_stats", stats),
                          ("tasks", [{"task_index": 0, "task": args.task}])]:
        (meta / f"{name}.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    processing = {"raw_root": str(args.raw_root.resolve()), "repo_id": args.repo_id,
        "episodes": len(trials), "frames": total_frames, "fps": 30, "source_fps_nominal": 30,
        "smoothing": None, "roi": None, "resize": "full RGB letterbox 224; INTER_AREA; black padding",
        "depth_used": False, "static_action_filter": False, "active_span_filter": False,
        "gripper_zeroed": False, "gripper_removed": True, "physical_action_dim": 6, "physical_state_dim": 9, "image_encoding": "PNG embedded in parquet; no video re-encoding",
        "alignment": "cam1 reference; nearest host timestamp; camera <=25ms; robot <=5ms; reject interior gaps",
        "downsample": False,
        "timestamp": "nominal frame_index/30 in parquet; real timestamps/intervals in meta/alignment",
        "state": "measured q -> Panda FK -> xyz, rotation columns 0 and 1",
        "action": "next selected measured TCP pose minus current; base-frame rotation log",
        "urdf": str(urdf), "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(), "per_episode": reports}
    (meta / "raw_processing.json").write_text(json.dumps(processing, indent=2))
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v2.1", "robot_type": "franka",
        "fps": 30, "total_episodes": len(trials), "total_frames": total_frames, "total_tasks": 1,
        "total_videos": 0, "total_chunks": (len(trials)+999)//1000, "chunks_size": 1000,
        "splits": {"train": f"0:{len(trials)}"}, "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", "video_path": None}, indent=2))
    print(f"Completed: {output}; {total_frames} frames", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/home/huiyuan/threading_new"))
    parser.add_argument("--output", type=Path, default=HERE / "data/threading_tcp6_nosmooth_30hz")
    parser.add_argument("--repo-id", default="threading_real/threading_tcp6_nosmooth_30hz")
    parser.add_argument("--workers", type=int, default=2, help="Independent episode workers; does not change data processing")
    parser.add_argument("--expected-episodes", type=int, default=80)
    parser.add_argument("--task", default="insert the grasped block through the needle")
    build(parser.parse_args())
