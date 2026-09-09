#!/usr/bin/env python3
"""Run one checkpoint on one dataset frame and print an unnormalized action chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="threading_real/threading_combined_pi05_15hz_sg5_nozero")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

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
    with torch.inference_mode():
        actions = postprocess(policy.predict_action_chunk(batch))
    array = actions.detach().cpu().numpy()
    print(json.dumps({"frame": args.frame, "task": frame["task"], "actions": array[0].tolist()}, indent=2))


if __name__ == "__main__":
    main()
