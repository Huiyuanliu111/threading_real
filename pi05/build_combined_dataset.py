#!/usr/bin/env python3
"""Build one filtered two-view pi0.5 dataset from multiple raw recording roots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_REPO_ID = "threading_real/threading_combined_pi05_15hz_sg5_nozero"


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
        default=DATA_ROOT / "threading_combined_pi05_15hz_sg5_nozero",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--expected-episodes", type=int, default=80)
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    prefix = output.with_name(output.name.removesuffix("_pi05_15hz_sg5_nozero"))
    stage = prefix.with_name(prefix.name + "_raw_staging")
    joint = prefix.with_name(prefix.name + "_lerobot_v3_joint_30hz")
    cartesian = prefix.with_name(prefix.name + "_lerobot_v3_cartesian_30hz")
    stride = prefix.with_name(prefix.name + "_lerobot_v3_cartesian_stride2_sg5_30hz")
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
            "--task", "insert the grasped block through the needle",
            "--fps", "30", "--image-size", "224", "--skip-depth",
        ])
        run([
            sys.executable, "convert_lerobot_v3_to_cartesian.py",
            str(joint), str(cartesian),
        ])
        run([
            sys.executable, "threading_real/scripts/make_cartesian_stride_actions.py",
            str(cartesian), str(stride), "--stride", "2",
            "--smooth-window", "5", "--smooth-polyorder", "2",
        ])
        run([
            sys.executable, "threading_real/pi05/prepare_dataset.py",
            "--source", str(stride), "--output", str(output),
            "--repo-id", args.repo_id, "--stride", "2",
        ])
        run([
            sys.executable, "threading_real/pi05/preflight.py",
            "--dataset-root", str(output), "--repo-id", args.repo_id,
            "--chunk-size", "10", "--expected-episodes", str(args.expected_episodes),
            "--expected-fps", "15",
        ])
        summary = {
            "raw_roots": [str(path.expanduser().resolve()) for path in args.raw_root],
            "episodes": episode_count,
            "output": str(output),
            "repo_id": args.repo_id,
            "fps": 15,
            "chunk_size": 10,
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
