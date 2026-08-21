#!/usr/bin/env python3
"""Validate real-robot LeRobot data before Threading ARP training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_REAL_CAMERA_KEYS = (
    "observation.images.exterior_image_1_left",
    "observation.images.wrist_image_left",
    "observation.images.exterior_image_2_right",
)


def _read_vector_column(path: Path, name: str) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to validate LeRobot parquet files") from exc
    table = pq.read_table(str(path), columns=[name])
    return np.asarray(table.column(name).to_pylist(), dtype=np.float32)


def _video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"exists": path.exists(), "open": False, "frames": 0, "width": 0, "height": 0}
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = capture.read()
        finite = bool(ok and np.isfinite(frame).all())
    finally:
        capture.release()
    return {
        "exists": path.exists(),
        "open": True,
        "frames": frames,
        "width": width,
        "height": height,
        "first_frame_finite": finite,
    }


def validate(
    dataset_path: Path,
    camera_keys: tuple[str, ...],
    state_key: str,
    action_key: str,
    expected_state_dim: int | None,
    expected_action_dim: int | None,
    max_episodes: int | None,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    root = dataset_path.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing {info_path}")
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    for key in (state_key, action_key, *camera_keys):
        if key not in features:
            errors.append(f"missing feature {key!r}")

    parquet_files = sorted((root / "data").rglob("episode_*.parquet"))
    if not parquet_files:
        parquet_files = sorted((root / "data").rglob("*.parquet"))
    if max_episodes is not None:
        parquet_files = parquet_files[:max_episodes]
    if not parquet_files:
        errors.append("no parquet files under data/")

    chunks_size = int(info.get("chunks_size", 1000))
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    episodes = []
    total_frames = 0
    for episode_index, parquet_path in enumerate(parquet_files):
        try:
            states = _read_vector_column(parquet_path, state_key)
            actions = _read_vector_column(parquet_path, action_key)
        except Exception as exc:
            errors.append(f"{parquet_path}: cannot read state/action: {exc}")
            continue
        if len(states) != len(actions):
            errors.append(f"{parquet_path}: state/action length mismatch {len(states)} != {len(actions)}")
        if expected_state_dim is not None and states.shape[1:] != (expected_state_dim,):
            errors.append(f"{parquet_path}: state shape {states.shape}, expected (*,{expected_state_dim})")
        if expected_action_dim is not None and actions.shape[1:] != (expected_action_dim,):
            errors.append(f"{parquet_path}: action shape {actions.shape}, expected (*,{expected_action_dim})")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            errors.append(f"{parquet_path}: state/action contains NaN or Inf")
        episode_report = {
            "episode_index": episode_index,
            "parquet": str(parquet_path.relative_to(root)),
            "frames": int(min(len(states), len(actions))),
            "videos": {},
        }
        total_frames += episode_report["frames"]
        for camera_key in camera_keys:
            video_rel = video_template.format(
                episode_chunk=episode_index // chunks_size,
                video_key=camera_key,
                episode_index=episode_index,
            )
            video_path = root / video_rel
            video = _video_info(video_path)
            episode_report["videos"][camera_key] = video
            if not video["open"]:
                errors.append(f"{video_path}: cannot open video")
            elif video["frames"] < episode_report["frames"]:
                errors.append(
                    f"{video_path}: {video['frames']} video frames < {episode_report['frames']} data rows"
                )
            if video.get("width", 0) <= 0 or video.get("height", 0) <= 0:
                errors.append(f"{video_path}: invalid video size")
        episodes.append(episode_report)

    if info.get("fps") != 30:
        warnings.append(f"expected Record_layer 30 FPS, got {info.get('fps')}")
    return {
        "dataset": str(root),
        "codebase_version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "episodes_checked": len(episodes),
        "total_frames_checked": total_frames,
        "state_feature": features.get(state_key),
        "action_feature": features.get(action_key),
        "camera_keys": list(camera_keys),
        "episodes": episodes,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--camera-keys", nargs="+", default=list(DEFAULT_REAL_CAMERA_KEYS))
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--expected-state-dim", type=int, default=8)
    parser.add_argument("--expected-action-dim", type=int, default=8)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(
        args.dataset,
        tuple(args.camera_keys),
        args.state_key,
        args.action_key,
        args.expected_state_dim,
        args.expected_action_dim,
        args.max_episodes,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
