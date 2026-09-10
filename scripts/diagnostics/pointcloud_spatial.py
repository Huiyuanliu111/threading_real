#!/usr/bin/env python3
"""Held-out accuracy and visual counterfactual checks for point-cloud Spatial ARP."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pushbox.diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from scripts.arp.policy_loader import load_policy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["cfg"]
    train_dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataset = train_dataset.get_validation_dataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    policy = load_policy(str(args.checkpoint), args.device, args.weights, True)
    if not getattr(policy, "uses_pointcloud", False):
        raise TypeError("checkpoint is not a point-cloud spatial policy")

    errors, predictions, targets, episode_errors = [], [], [], defaultdict(list)
    saved = []
    saved_episodes = set()
    with torch.inference_mode():
        for batch in loader:
            batch = dict_apply(batch, lambda value: value.to(args.device))
            output = policy.predict_action(batch["obs"])
            error = (output["spatial_goal_xyz"] - batch["spatial_goal_xyz"]).norm(dim=-1)
            errors.append(error.cpu())
            predictions.append(output["spatial_goal_xyz"].cpu())
            targets.append(batch["spatial_goal_xyz"].cpu())
            for episode, value in zip(batch["episode_index"].cpu().tolist(), error.cpu().tolist()):
                episode_errors[int(episode)].append(value)
            for index, episode in enumerate(batch["episode_index"].cpu().tolist()):
                if episode not in saved_episodes:
                    saved_episodes.add(episode)
                    saved.append({key: value[index:index+1].clone() for key, value in batch["obs"].items()})

        error = torch.cat(errors)
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        train_mean = torch.from_numpy(np.asarray(train_dataset.goals)[train_dataset.selected_episodes].mean(axis=0))
        baseline = (target - train_mean).norm(dim=-1)
        print(f"train_episodes={train_dataset.selected_episodes}")
        print(f"val_episodes={dataset.selected_episodes} samples={len(error)}")
        print(f"xyz_error_mean={error.mean()*1000:.2f}mm median={error.median()*1000:.2f}mm p90={error.quantile(.9)*1000:.2f}mm")
        print(f"mean_goal_baseline={baseline.mean()*1000:.2f}mm improvement={(1-error.mean()/baseline.mean())*100:.1f}%")
        for episode, values in sorted(episode_errors.items()):
            print(f"episode={episode:02d} mean={np.mean(values)*1000:.2f}mm n={len(values)}")

        # Hold TCP/proprioception fixed and replace only point cloud input. A
        # non-zero spread is direct evidence that vision changes the command.
        fixed = saved[0]
        swapped = []
        for source in saved:
            obs = dict(fixed)
            for key in ("points", "colors", "camera_id", "bev"):
                obs[key] = source[key]
            swapped.append(policy.predict_action(obs)["spatial_goal_xyz"].cpu())
        swapped = torch.cat(swapped)
        spread = (swapped - swapped.mean(dim=0)).norm(dim=-1).mean()
        unique = torch.unique(swapped, dim=0).shape[0]
        print(f"fixed_state_pointcloud_swap_goal_spread={spread*1000:.2f}mm unique_bins={unique}/{len(swapped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
