#!/usr/bin/env python3
"""Build a filtered, temporally downsampled two-camera dataset for pi0.5.

The input actions are already Cartesian deltas from frame t to t+stride. The
source dataset is still stored at its camera rate, so we keep one frame out of
every stride frames. This makes adjacent actions in a pi0.5 chunk non-overlapping.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


META_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
EXPECTED_CAMERAS = {
    "observation.images.exterior_image_1_left",
    "observation.images.exterior_image_2_right",
}


def _as_frame(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return array


def _as_array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _is_zero_action(
    action: np.ndarray,
    translation_threshold: float,
    rotation_threshold: float,
    gripper_threshold: float,
) -> bool:
    return bool(
        np.linalg.norm(action[:3]) < translation_threshold
        and np.linalg.norm(action[3:6]) < rotation_threshold
        and abs(float(action[6])) < gripper_threshold
    )


def convert(
    source: Path,
    output: Path,
    repo_id: str,
    stride: int,
    overwrite: bool,
    translation_threshold: float = 1e-3,
    rotation_threshold: float = 1e-2,
    gripper_threshold: float = 5e-4,
) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir() or not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {source}")
    if stride < 1:
        raise ValueError("--stride must be positive")
    if min(translation_threshold, rotation_threshold, gripper_threshold) < 0:
        raise ValueError("zero-action thresholds must be non-negative")
    if output == source or output in source.parents:
        raise ValueError("Output must not be the source dataset or one of its parents")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)

    src = LeRobotDataset(repo_id=f"source/{repo_id}", root=source, video_backend="pyav")
    if src.fps % stride:
        raise ValueError(f"Source fps {src.fps} is not divisible by stride {stride}")
    cameras = set(src.meta.camera_keys)
    if cameras != EXPECTED_CAMERAS:
        raise ValueError(f"Expected cameras {sorted(EXPECTED_CAMERAS)}, got {sorted(cameras)}")
    action_names = src.meta.features["action"].get("names")
    if action_names != ["dx", "dy", "dz", "drotvec_x", "drotvec_y", "drotvec_z", "dgripper"]:
        raise ValueError(f"Expected Cartesian stride actions, got names={action_names}")

    features = {key: value for key, value in src.meta.features.items() if key not in META_KEYS}
    dst = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=src.fps // stride,
        robot_type=src.meta.robot_type,
        features=features,
        use_videos=True,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec="libsvtav1", video_backend="pyav"),
    )

    total_frames = 0
    episode_lengths: list[int] = []
    removed_zero_actions_per_episode: list[int] = []
    for episode in src.meta.episodes:
        start = int(episode["dataset_from_index"])
        stop = int(episode["dataset_to_index"])
        kept = 0
        removed = 0
        for index in range(start, stop, stride):
            sample = src[index]
            action = _as_array(sample["action"]).astype(np.float32, copy=False)
            if _is_zero_action(
                action,
                translation_threshold,
                rotation_threshold,
                gripper_threshold,
            ):
                removed += 1
                continue
            frame = {
                key: (_as_frame(sample[key]) if key in cameras else _as_array(sample[key]))
                for key in features
            }
            frame["task"] = str(sample["task"])
            dst.add_frame(frame)
            kept += 1
        if kept == 0:
            raise ValueError(
                f"episode {int(episode['episode_index'])} has no non-zero actions after filtering"
            )
        dst.save_episode()
        episode_lengths.append(kept)
        removed_zero_actions_per_episode.append(removed)
        total_frames += kept
    dst.finalize()

    report = {
        "source": str(source),
        "output": str(output),
        "repo_id": repo_id,
        "source_fps": src.fps,
        "stride": stride,
        "output_fps": dst.fps,
        "episodes": len(episode_lengths),
        "frames": total_frames,
        "episode_lengths": episode_lengths,
        "removed_zero_actions_per_episode": removed_zero_actions_per_episode,
        "removed_zero_actions": int(sum(removed_zero_actions_per_episode)),
        "zero_action_thresholds": {
            "translation_m": translation_threshold,
            "rotation_rad": rotation_threshold,
            "gripper_m": gripper_threshold,
        },
        "cameras": sorted(cameras),
        "state_dim": int(src.meta.features["observation.state"]["shape"][0]),
        "action_dim": int(src.meta.features["action"]["shape"][0]),
    }
    report_path = output / "meta" / "pi05_preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="threading_real/threading_combined_pi05_15hz_sg5_nozero")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--zero-translation", type=float, default=1e-3)
    parser.add_argument("--zero-rotation", type=float, default=1e-2)
    parser.add_argument("--zero-gripper", type=float, default=5e-4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(convert(
        args.source, args.output, args.repo_id, args.stride, args.overwrite,
        args.zero_translation, args.zero_rotation, args.zero_gripper,
    ), indent=2))


if __name__ == "__main__":
    main()
