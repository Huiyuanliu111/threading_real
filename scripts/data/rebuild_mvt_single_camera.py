#!/usr/bin/env python3
"""Rebuild an MVT HDF5 point cloud from one recorded RGB-D camera.

The source MVT file supplies the exact episode/frame selection, robot state,
actions, and camera frame indices. Only points, colors, and point-count
metadata are regenerated from the selected physical camera.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from build_threading_mvt_dataset import deterministic_surface_points  # noqa: E402
from build_vla_pointcloud_dataset import (  # noqa: E402
    _camera_geometry,
    _video_count,
    depth_to_base_points,
)
from convert_vla_to_lerobot_v3 import (  # noqa: E402
    SequentialVideoReader,
    close_depth_source,
)


REGENERATED_DATASETS = {
    "points",
    "colors",
    "valid_points",
    "raw_valid_points",
    "cropped_points",
    "voxel_points",
    "camera_frame_index",
}


def copy_preserved_datasets(source: h5py.Group, target: h5py.Group) -> None:
    for name in source:
        if name not in REGENERATED_DATASETS:
            source.copy(name, target, name=name)


def rebuild(args: argparse.Namespace) -> None:
    source_path = args.source.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    partial_path = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or partial_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path} or {partial_path}")

    calibration_path = args.calibration.expanduser().resolve()
    calibration = json.loads(calibration_path.read_text())
    raw_root = args.raw_root.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []

    with h5py.File(source_path, "r") as source, h5py.File(partial_path, "w") as target:
        if source.attrs.get("format") != "threading-mvt-pointcloud-v1":
            raise ValueError(f"unsupported source MVT dataset: {source_path}")
        for key, value in source.attrs.items():
            target.attrs[key] = value
        bounds = np.asarray(source.attrs["bounds_m"], dtype=np.float32)
        max_points = int(source.attrs["max_points"])
        voxel_m = float(source.attrs["voxel_m"])
        target.attrs["raw_root"] = str(raw_root)
        target.attrs["calibration_path"] = str(calibration_path)
        target.attrs["source_pointcloud_dataset"] = str(source_path)
        target.attrs["camera_mapping_json"] = json.dumps(
            [[args.camera_file, args.calibration_key]]
        )
        target.attrs["physical_views_json"] = json.dumps([args.calibration_key])
        target.attrs["pointcloud_source_count"] = 1

        episode_keys = sorted(source.keys())
        for episode_number, episode_key in enumerate(episode_keys):
            source_group = source[episode_key]
            if "source_trial" not in source_group.attrs:
                raise KeyError(f"{episode_key} has no source_trial attribute")
            frame_indices = np.asarray(source_group["camera_frame_index"])
            if frame_indices.ndim != 2 or args.camera_column >= frame_indices.shape[1]:
                raise ValueError(
                    f"{episode_key} camera_frame_index has shape {frame_indices.shape}; "
                    f"cannot select column {args.camera_column}"
                )
            selected_frames = frame_indices[:, args.camera_column].astype(np.int64)
            if np.any(selected_frames < 0) or np.any(np.diff(selected_frames) < 0):
                raise ValueError(f"{episode_key} selected camera frames must be nonnegative and monotone")

            trial = raw_root / str(source_group.attrs["source_trial"])
            video_path = trial / args.camera_file
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise ValueError(f"cannot open {video_path}")
            geometry = None
            try:
                committed_frames = _video_count(capture, video_path)
                if len(selected_frames) and selected_frames[-1] >= committed_frames:
                    raise IndexError(
                        f"{episode_key} requests frame {selected_frames[-1]} but "
                        f"{video_path} has {committed_frames} frames"
                    )
                geometry = _camera_geometry(
                    trial,
                    args.camera_file,
                    args.calibration_key,
                    calibration,
                    committed_frames,
                )
                depth, metadata, rays, camera_from_base = geometry
                reader = SequentialVideoReader(capture, video_path)

                target_group = target.create_group(episode_key)
                for key, value in source_group.attrs.items():
                    target_group.attrs[key] = value
                copy_preserved_datasets(source_group, target_group)
                count = len(selected_frames)
                point_ds = target_group.create_dataset(
                    "points",
                    (count, max_points, 3),
                    dtype="f2",
                    chunks=(1, max_points, 3),
                    compression="lzf",
                )
                color_ds = target_group.create_dataset(
                    "colors",
                    (count, max_points, 3),
                    dtype="u1",
                    chunks=(1, max_points, 3),
                    compression="lzf",
                )
                valid_counts = np.empty(count, dtype=np.int32)
                cropped_counts = np.empty(count, dtype=np.int32)

                for destination_row, source_frame in enumerate(selected_frames):
                    bgr = reader.read(int(source_frame))
                    xyz, valid_depth = depth_to_base_points(
                        depth[int(source_frame)],
                        rays,
                        float(metadata["depth_scale_m"]),
                        camera_from_base,
                    )
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[valid_depth]
                    points, colors, valid, cropped = deterministic_surface_points(
                        [xyz], [rgb], bounds, max_points, voxel_m
                    )
                    point_ds[destination_row] = points.astype(np.float16)
                    color_ds[destination_row] = colors
                    valid_counts[destination_row] = valid
                    cropped_counts[destination_row] = cropped

                target_group.create_dataset("valid_points", data=valid_counts)
                target_group.create_dataset("cropped_points", data=cropped_counts)
                target_group.create_dataset("camera_frame_index", data=selected_frames[:, None])
                report = {
                    "episode": episode_number,
                    "source_trial": str(source_group.attrs["source_trial"]),
                    "frames": count,
                    "min_valid_points": int(valid_counts.min()),
                    "min_cropped_points": int(cropped_counts.min()),
                    "max_cropped_points": int(cropped_counts.max()),
                }
                reports.append(report)
                print(json.dumps(report), file=sys.stderr, flush=True)
            finally:
                if geometry is not None:
                    close_depth_source(geometry[0])
                capture.release()

        target.attrs["report_json"] = json.dumps(reports)
        target.flush()
    partial_path.replace(output_path)
    print(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing MVT HDF5 supplying frame selection")
    parser.add_argument("output", type=Path, help="new single-camera MVT HDF5")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera-file", default="cam1.mp4")
    parser.add_argument("--calibration-key", default="sideview")
    parser.add_argument("--camera-column", type=int, default=0)
    return parser


if __name__ == "__main__":
    rebuild(build_parser().parse_args())
