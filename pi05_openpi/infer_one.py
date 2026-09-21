#!/usr/bin/env python3
"""Offline inference on exported dataset frames; never sends robot commands."""

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np

from run import bind_dataset, parser as training_parser, setup
from transforms import FRONT, SIDE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-config", type=Path, help="Required for fine-tuned checkpoints")
    # Keep cloud URIs as strings: Path would turn gs:// into gs:/.
    p.add_argument("--checkpoint", help="Local directory or gs:// URI; defaults to pi05_base only with --base")
    p.add_argument("--base", action="store_true", help="Use base weights with this dataset's statistics for a smoke test")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--openpi-root", type=Path)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--num-steps", type=int, default=10, help="Flow-matching denoising steps")
    p.add_argument("--warmup", type=int, default=0, help="Extra inference calls before measuring the final call")
    args = p.parse_args()
    if args.checkpoint is None:
        if not args.base:
            p.error("--checkpoint is required for a fine-tuned policy")
        args.checkpoint = "gs://openpi-assets/checkpoints/pi05_base"
    if args.num_steps < 1 or args.warmup < 0:
        p.error("num-steps must be positive and warmup must be nonnegative")
    if args.run_config is None:
        if not args.base:
            p.error("Use --run-config for trained weights, or explicitly select --base for a base checkpoint")
        defaults = training_parser().parse_args(["check"])
        settings = {key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
                    for key, value in vars(defaults).items() if key not in ("command", "print_config")}
        settings["assets_dir"] = str(Path(settings["assets_dir"]) / settings["exp_name"])
    else:
        settings = json.loads(args.run_config.read_text())
    for key in ("openpi_root", "dataset_root"):
        if getattr(args, key) is not None:
            settings[key] = str(getattr(args, key).expanduser().resolve())
    config = setup(settings)
    bind_dataset(settings)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies.policy_config import create_trained_policy
    dataset = LeRobotDataset(settings["repo_id"])
    if not 0 <= args.frame < len(dataset):
        raise ValueError("frame outside dataset")
    sample = dataset[args.frame]
    norm_stats = None
    if args.base:
        norm_stats = config.data.create(config.assets_dirs, config.model).norm_stats
        if norm_stats is None:
            raise FileNotFoundError("Base checkpoint smoke test requires the dataset's computed norm stats")
        print("BASE CHECKPOINT SMOKE TEST: dataset statistics are supplied explicitly; this is not a fine-tuned threading policy.", flush=True)
    logging.basicConfig(level=logging.INFO)
    load_start = time.monotonic()
    policy = create_trained_policy(config, args.checkpoint, norm_stats=norm_stats,
                                   sample_kwargs={"num_steps": args.num_steps})
    load_seconds = time.monotonic() - load_start
    inputs = {"front": sample[FRONT].numpy(), "side": sample[SIDE].numpy(),
              "state": sample["observation.state"].numpy(), "prompt": sample["task"]}
    warmup_seconds = []
    for _ in range(args.warmup):
        start = time.monotonic()
        policy.infer(inputs)
        warmup_seconds.append(time.monotonic() - start)
    start = time.monotonic()
    output = policy.infer(inputs)
    # Conversion to numpy synchronizes JAX; elapsed time includes device execution.
    actions = np.asarray(output["actions"])
    infer_seconds = time.monotonic() - start
    if actions.shape != (settings["horizon"], 6) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid model output: {actions.shape}")
    print(actions)
    report = {"checkpoint": args.checkpoint, "base_checkpoint_smoke_test": args.base,
              "dataset_root": settings["dataset_root"], "fps": dataset.fps, "frame": args.frame,
              "num_steps": args.num_steps, "warmup_calls": args.warmup,
              "warmup_seconds": warmup_seconds,
              "load_seconds": load_seconds, "inference_seconds": infer_seconds,
              "actions_shape": list(actions.shape), "finite": bool(np.isfinite(actions).all()),
              "norm_stats_source": "dataset" if args.base else "checkpoint"}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Use a handle so the exact requested path is respected.
        with args.output.open("wb") as stream:
            np.save(stream, actions)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
