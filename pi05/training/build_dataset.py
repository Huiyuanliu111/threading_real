#!/usr/bin/env python3
"""Build one continuous two-view pi0.5 dataset from multiple raw recording roots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_DATASET_NAME = "threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d"
DEFAULT_REPO_ID = f"threading_real/{DEFAULT_DATASET_NAME}"


def run(command: list[str]) -> None:
    print("[pipeline]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def stage_raw_roots(raw_roots: list[Path], stage: Path) -> int:
    stage.mkdir(parents=True)
    count = 0
    for source_index, raw_root in enumerate(raw_roots, start=1):
        trials = sorted(raw_root.expanduser().resolve().glob("episode_*/DATA_follower.m"))
        if not trials:
            raise FileNotFoundError(f"no episodes found under {raw_root}")
        for robot_file in trials:
            trial = robot_file.parent
            destination = stage / f"source_{source_index:02d}_{trial.name}"
            destination.mkdir()
            for source_file in trial.iterdir():
                (destination / source_file.name).symlink_to(source_file.resolve())
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, action="append", required=True,
        help="repeat for each raw recording root",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DATA_ROOT / DEFAULT_DATASET_NAME,
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", default="insert the grasped block through the needle")
    parser.add_argument("--expected-episodes", type=int, default=80)
    parser.add_argument("--source-fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--smooth-polyorder", type=int, default=2)
    parser.add_argument("--zero-translation", type=float, default=1e-3)
    parser.add_argument("--zero-rotation", type=float, default=1e-2)
    parser.add_argument("--zero-gripper", type=float, default=5e-4)
    parser.add_argument(
        "--drop-zero-actions",
        action="store_true",
        help="optional ablation; disabled by default to preserve the fixed-rate visual trajectory",
    )
    parser.add_argument(
        "--state-representation",
        choices=("joint", "tcp_pose", "tcp_pose_6d"),
        default="tcp_pose_6d",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=REPO_ROOT / "remote_controller/src/remote_controller/assets/panda/panda_arm.urdf",
    )
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()

    for name in ("expected_episodes", "source_fps", "image_size", "action_stride", "chunk_size"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.source_fps % args.action_stride:
        raise ValueError("--source-fps must be divisible by --action-stride")
    expected_fps = args.source_fps // args.action_stride

    output = args.output.expanduser().resolve()
    stage = output.with_name(output.name + ".raw_staging")
    joint = output.with_name(output.name + ".joint_source")
    cartesian = output.with_name(output.name + ".cartesian_source")
    stride = output.with_name(output.name + ".cartesian_stride")
    generated = [stage, joint, cartesian, stride, output]
    existing = [str(path) for path in generated if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite pipeline outputs: " + ", ".join(existing))

    episode_count = 0
    try:
        episode_count = stage_raw_roots(args.raw_root, stage)
        if episode_count != args.expected_episodes:
            raise ValueError(
                f"expected {args.expected_episodes} raw episodes, found {episode_count}"
            )
        run([
            sys.executable, "convert_vla_to_lerobot_v3.py", str(stage), str(joint),
            "--repo-id", args.repo_id,
            "--task", args.task,
            "--fps", str(args.source_fps),
            "--image-size", str(args.image_size), "--skip-depth",
        ])
        run([
            sys.executable, "convert_lerobot_v3_to_cartesian.py",
            str(joint), str(cartesian), "--urdf", str(args.urdf),
        ])
        run([
            sys.executable, "threading_real/scripts/data/cartesian_stride.py",
            str(cartesian), str(stride), "--stride", str(args.action_stride),
            "--smooth-window", str(args.smooth_window),
            "--smooth-polyorder", str(args.smooth_polyorder),
        ])
        prepare_command = [
            sys.executable, "threading_real/pi05/training/prepare_dataset.py",
            "--source", str(stride), "--output", str(output),
            "--repo-id", args.repo_id, "--stride", str(args.action_stride),
            "--state-representation", args.state_representation,
            "--urdf", str(args.urdf),
            "--zero-translation", str(args.zero_translation),
            "--zero-rotation", str(args.zero_rotation),
            "--zero-gripper", str(args.zero_gripper),
        ]
        if args.drop_zero_actions:
            prepare_command.append("--drop-zero-actions")
        run(prepare_command)
        run([
            sys.executable, "threading_real/pi05/training/preflight.py",
            "--dataset-root", str(output), "--repo-id", args.repo_id,
            "--chunk-size", str(args.chunk_size),
            "--expected-episodes", str(args.expected_episodes),
            "--expected-fps", str(expected_fps),
            "--state-representation", args.state_representation,
        ])
        summary = {
            "raw_roots": [str(path.expanduser().resolve()) for path in args.raw_root],
            "episodes": episode_count,
            "output": str(output),
            "repo_id": args.repo_id,
            "task": args.task,
            "source_fps": args.source_fps,
            "fps": expected_fps,
            "image_size": args.image_size,
            "action_stride": args.action_stride,
            "chunk_size": args.chunk_size,
            "smooth_window": args.smooth_window,
            "smooth_polyorder": args.smooth_polyorder,
            "zero_action_thresholds": {
                "translation_m": args.zero_translation,
                "rotation_rad": args.zero_rotation,
                "gripper_m": args.zero_gripper,
            },
            "drop_zero_actions": args.drop_zero_actions,
            "state_representation": args.state_representation,
            "urdf": str(args.urdf.expanduser().resolve()),
            "cameras": [
                "observation.images.exterior_image_2_right",
                "observation.images.exterior_image_1_left",
            ],
        }
        (output / "meta" / "combined_pipeline_report.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if not args.keep_intermediates and output.exists():
            for path in (joint, cartesian, stride):
                shutil.rmtree(path, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
