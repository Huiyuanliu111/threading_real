#!/usr/bin/env python3
"""Offline accuracy and counterfactual checks for a spatial checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pushbox.diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from scripts.eval_policy import load_policy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg")
    if cfg is None:
        raise KeyError("checkpoint does not contain its Hydra config")
    dataset = hydra.utils.instantiate(cfg.task.dataset).get_validation_dataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    policy = load_policy(
        str(args.checkpoint),
        device=args.device,
        weights=args.weights,
        use_checkpoint_config=True,
    )
    if not hasattr(policy, "projection_matrices"):
        raise TypeError("checkpoint is not a spatial policy")

    pixel_errors = []
    xyz_errors = []
    observations = []
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            batch = dict_apply(batch, lambda value: value.to(args.device))
            result = policy.predict_action(batch["obs"])
            scale = policy.image_shape[-1] / policy.heatmap_size
            predicted_pixels = result["spatial_goal_pixels"] * scale
            pixel_error = (predicted_pixels - batch["spatial_goal_pixels"]).norm(dim=-1)
            pixel_errors.append(pixel_error.cpu())
            xyz_errors.append(
                (result["spatial_goal_xyz"] - batch["spatial_goal_xyz"]).norm(dim=-1).cpu()
            )
            if len(observations) < 32:
                for index in range(min(batch["obs"]["agent_pos"].shape[0], 32 - len(observations))):
                    observations.append(
                        {key: value[index : index + 1].clone() for key, value in batch["obs"].items()}
                    )
            seen += pixel_error.shape[0]
            if seen >= args.max_samples:
                break

        pixels = torch.cat(pixel_errors)[: args.max_samples]
        xyz = torch.cat(xyz_errors)[: args.max_samples]
        print(f"samples={len(xyz)}")
        print(f"pixel_error_mean={pixels.mean().item():.2f}px median={pixels.median().item():.2f}px")
        print(
            f"pck@5px={(pixels <= 5).float().mean().item():.3f} "
            f"pck@10px={(pixels <= 10).float().mean().item():.3f}"
        )
        print(f"xyz_error_mean={xyz.mean().item()*1000:.2f}mm median={xyz.median().item()*1000:.2f}mm")

        if len(observations) >= 2:
            fixed = observations[0]
            predicted_goals = []
            for source in observations:
                counterfactual = {
                    key: (source[key] if key in policy.rgb_keys else fixed[key])
                    for key in fixed
                }
                predicted_goals.append(policy.predict_action(counterfactual)["spatial_goal_xyz"].cpu())
            goals = torch.cat(predicted_goals)
            spread = (goals - goals.mean(dim=0)).norm(dim=-1).mean()
            print(f"fixed_state_image_swap_goal_spread={spread.item()*1000:.2f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
