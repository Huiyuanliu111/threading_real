#!/usr/bin/env python3
"""Run a SmolVLA checkpoint on one dataset frame and print its Cartesian action chunk."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="threading_real/block_grasp_smolvla_6hz")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root.expanduser().resolve(),
        video_backend="pyav",
    )
    policy = SmolVLAPolicy.from_pretrained(checkpoint).to(args.device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    frame = dict(dataset[args.frame])
    batch = preprocess(frame)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        actions = postprocess(policy.predict_action_chunk(batch))
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    array = actions.detach().cpu().numpy()
    result = {
        "frame": args.frame,
        "task": frame["task"],
        "shape": list(array[0].shape),
        "latency_seconds": elapsed,
        "actions": array[0].tolist(),
    }
    if args.device.startswith("cuda"):
        result["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
