#!/usr/bin/env python3
"""Measure whether a ThreadingReal policy changes actions when images change.

The diagnostic fixes one recorded q/gripper history and substitutes image
histories from multiple episodes. A near-zero action spread means the policy
is effectively ignoring vision for that state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_policy import load_policy  # noqa: E402
from threading_task.dataset import ThreadingRealLeRobotDataset  # noqa: E402


def sample_index(dataset: ThreadingRealLeRobotDataset, episode: int, frame: int) -> int:
    candidates = [
        index for index, (ep, start) in enumerate(dataset.sample_indices)
        if int(ep) == episode and int(start) >= frame
    ]
    if not candidates:
        raise ValueError(f"no valid sample for episode={episode}, frame>={frame}")
    return candidates[0]


def first_action(policy: torch.nn.Module, obs: dict[str, torch.Tensor], device: str) -> np.ndarray:
    batch = {key: value.unsqueeze(0).to(device) for key, value in obs.items()}
    with torch.inference_mode():
        return policy.predict_action(batch)["action"][0, 0].detach().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "data" / "threading_lerobot_v3_cartesian")
    parser.add_argument("--base-episode", type=int, default=0)
    parser.add_argument("--base-frame", type=int, default=0)
    parser.add_argument("--num-image-episodes", type=int, default=10)
    parser.add_argument("--image-frame", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.num_image_episodes < 2: parser.error("--num-image-episodes must be at least 2")

    policy = load_policy(str(args.checkpoint), device=args.device, weights="ema", use_checkpoint_config=True)
    if getattr(policy, "action_mode", None) != "cartesian_delta" or int(policy.action_dim) != 7:
        parser.error("checkpoint must be a 7D cartesian_delta policy")
    from threading_task.policy import enable_map_gmm_inference
    enable_map_gmm_inference(policy)
    dataset = ThreadingRealLeRobotDataset(
        dataset_path=str(args.dataset), horizon=max(21, int(policy.horizon) + 1),
        pad_before=max(1, int(policy.n_obs_steps) - 1), pad_after=7,
        n_obs_steps=int(policy.n_obs_steps), action_mode="cartesian_delta", image_size=96,
    )
    base = dataset[sample_index(dataset, args.base_episode, args.base_frame)]["obs"]
    available = np.unique([int(ep) for ep, _ in dataset.sample_indices])
    selected = np.unique(np.linspace(0, len(available) - 1, args.num_image_episodes, dtype=int))
    records: list[dict[str, object]] = []
    for index in selected:
        episode = int(available[index])
        image_sample = dataset[sample_index(dataset, episode, args.image_frame)]
        image_obs = image_sample["obs"]
        # Fixed q/gripper; only RGB histories are replaced.
        obs = {**image_obs, "agent_pos": base["agent_pos"]}
        action = first_action(policy, obs, args.device)
        label = image_sample["action"][0]
        records.append({
            "image_episode": episode,
            "predicted_action": action.astype(float).tolist(),
            "recorded_action": label.detach().cpu().numpy().astype(float).tolist(),
        })
    actions = np.asarray([record["predicted_action"] for record in records], dtype=float)
    labels = np.asarray([record["recorded_action"] for record in records], dtype=float)
    translation = actions[:, :3]
    rotation = actions[:, 3:6]
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "dataset": str(args.dataset.resolve()),
        "fixed_state_episode": args.base_episode, "fixed_state_frame": args.base_frame,
        "image_frame": args.image_frame, "records": records,
        "translation_std_mm": (translation.std(axis=0) * 1000).tolist(),
        "translation_spread_mm": float(np.linalg.norm(translation.max(axis=0) - translation.min(axis=0)) * 1000),
        "recorded_translation_spread_mm": float(
            np.linalg.norm(labels[:, :3].max(axis=0) - labels[:, :3].min(axis=0)) * 1000
        ),
        "visual_response_to_recorded_spread_ratio": float(
            np.linalg.norm(translation.max(axis=0) - translation.min(axis=0)) /
            max(np.linalg.norm(labels[:, :3].max(axis=0) - labels[:, :3].min(axis=0)), 1e-12)
        ),
        "rotation_std_deg": np.degrees(rotation.std(axis=0)).tolist(),
        "mean_translation_mm": (translation.mean(axis=0) * 1000).tolist(),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
