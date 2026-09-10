#!/usr/bin/env python3
"""Run one checkpoint on one dataset frame and print an unnormalized action chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="threading_real/threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    args = parser.parse_args()

    if args.warmup_runs < 0 or args.benchmark_runs < 1:
        parser.error("--warmup-runs must be >= 0 and --benchmark-runs must be >= 1")

    checkpoint = args.checkpoint.expanduser().resolve()
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root.expanduser().resolve(), video_backend="pyav")
    policy = PI05Policy.from_pretrained(checkpoint).to(args.device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    frame = dict(dataset[args.frame])
    batch = preprocess(frame)

    def predict() -> torch.Tensor:
        with torch.inference_mode():
            return postprocess(policy.predict_action_chunk(batch))

    def synchronize() -> None:
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize(args.device)

    for _ in range(args.warmup_runs):
        predict()
    synchronize()

    timings_ms = []
    actions = None
    for _ in range(args.benchmark_runs):
        started = time.perf_counter()
        actions = predict()
        synchronize()
        timings_ms.append((time.perf_counter() - started) * 1000.0)

    assert actions is not None
    sorted_timings = sorted(timings_ms)
    p95_index = min(len(sorted_timings) - 1, int(0.95 * len(sorted_timings)))
    mean_ms = statistics.fmean(timings_ms)
    array = actions.detach().cpu().numpy()
    chunk_steps = int(array.shape[1])
    print(json.dumps({
        "frame": args.frame,
        "task": frame["task"],
        "dataset_action_hz": float(dataset.fps),
        "action_chunk_steps": chunk_steps,
        "action_chunk_seconds": chunk_steps / float(dataset.fps),
        "benchmark": {
            "warmup_runs": args.warmup_runs,
            "runs": args.benchmark_runs,
            "mean_inference_ms": mean_ms,
            "median_inference_ms": statistics.median(timings_ms),
            "p95_inference_ms": sorted_timings[p95_index],
            "chunk_inferences_per_second": 1000.0 / mean_ms,
        },
        "actions": array[0].tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
