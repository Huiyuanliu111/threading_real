#!/usr/bin/env python3
"""Fail-fast checks for the 6 Hz Cartesian dataset used by SmolVLA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


CAMERAS = {
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_right",
    "observation.images.wrist_image_left",
}
ACTION_NAMES = ["dx", "dy", "dz", "drotvec_x", "drotvec_y", "drotvec_z", "dgripper"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="threading_real/block_grasp_minimal_pi05_6hz")
    parser.add_argument("--chunk-size", type=int, default=10)
    args = parser.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root.expanduser().resolve(), video_backend="pyav")
    errors: list[str] = []
    if ds.fps != 6:
        errors.append(f"fps must be 6, got {ds.fps}")
    if set(ds.meta.camera_keys) != CAMERAS:
        errors.append(f"camera keys mismatch: {ds.meta.camera_keys}")
    if ds.meta.features["observation.state"]["shape"] != (8,):
        errors.append(f"state shape must be (8,), got {ds.meta.features['observation.state']['shape']}")
    if ds.meta.features["action"]["shape"] != (7,):
        errors.append(f"action shape must be (7,), got {ds.meta.features['action']['shape']}")
    if ds.meta.features["action"].get("names") != ACTION_NAMES:
        errors.append(f"action names mismatch: {ds.meta.features['action'].get('names')}")
    for key in ("observation.state", "action"):
        missing = {"q01", "q99"} - set(ds.meta.stats[key])
        if missing:
            errors.append(f"{key} lacks quantile stats: {sorted(missing)}")
    shortest = min(int(ep["length"]) for ep in ds.meta.episodes)
    if shortest < args.chunk_size:
        errors.append(f"shortest episode ({shortest}) is shorter than chunk size ({args.chunk_size})")

    sample = ds[0]
    summary = {
        "dataset_root": str(ds.root),
        "episodes": ds.num_episodes,
        "frames": ds.num_frames,
        "fps": ds.fps,
        "cameras": ds.meta.camera_keys,
        "state_shape": list(sample["observation.state"].shape),
        "action_shape": list(sample["action"].shape),
        "task": sample["task"],
        "chunk_size": args.chunk_size,
        "chunk_seconds": args.chunk_size / ds.fps,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
