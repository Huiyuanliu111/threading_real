"""Read trajectory and video provenance from both real-task MVT HDF5 formats."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def read_trajectories(dataset: Path, *, task: str, urdf: Path | None = None):
    with h5py.File(dataset, "r") as source:
        expected_format = f"{'threading' if task == 'threading' else 'maze'}-mvt-pointcloud-v1"
        if source.attrs.get("format") != expected_format:
            raise ValueError(f"expected dataset format {expected_format}")
        trajectories = {}
        if task == "threading":
            from convert_lerobot_v3_to_cartesian import UrdfForwardKinematics
            if urdf is None:
                raise ValueError("Threading trajectories require the Panda URDF")
            fk = UrdfForwardKinematics(urdf.expanduser().resolve())
        for key in sorted(source):
            group = source[key]
            if task == "threading":
                trajectories[key] = fk.poses(group["observation_state"][:, :7])[0]
            else:
                xy = group["tcp_xy"][:]
                trajectories[key] = np.column_stack((xy, np.full(len(xy), source.attrs["fixed_z_m"])))
        return trajectories


def video_provenance(source, episode_key, *, raw_root: Path | None = None):
    group = source[episode_key]
    root = raw_root if raw_root is not None else Path(source.attrs["raw_root"])
    trial = group.attrs.get("source_trial")
    if trial is None:
        reports = json.loads(source.attrs.get("report_json", "[]"))
        episode = int(episode_key.rsplit("_", 1)[-1])
        report = next((item for item in reports if int(item["episode"]) == episode), {})
        trial = report.get("source_trial", report.get("trial"))
    if trial is None:
        raise ValueError(f"no video trial provenance for {episode_key}")
    if "camera_mapping_json" in source.attrs:
        mapping = json.loads(source.attrs["camera_mapping_json"])
        filenames = [item[0] for item in mapping]
    elif source.attrs["format"] == "maze-mvt-pointcloud-v1":
        filenames = ["cam3.mp4"]
    else:
        raise ValueError("dataset has no camera mapping")
    indices = group["camera_frame_index"][:]
    if indices.ndim != 2 or indices.shape[1] != len(filenames):
        raise ValueError("camera frame indices do not match camera mapping")
    return [root.expanduser() / trial / name for name in filenames], indices
