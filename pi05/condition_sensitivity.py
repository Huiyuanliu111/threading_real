#!/usr/bin/env python3
"""Measure pi0.5 sensitivity to images and state while holding flow noise fixed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)


def _predict_normalized(
    policy: PI05Policy,
    batch: dict[str, torch.Tensor],
    noise: torch.Tensor,
) -> torch.Tensor:
    images, image_masks = policy._preprocess_images(batch)
    states, state_masks = policy._prepare_memory_states(batch)
    actions = policy.model.sample_actions(
        images,
        image_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        states=states,
        state_masks=state_masks,
        noise=noise,
    )
    action_dim = policy.config.output_features["action"].shape[0]
    return actions[:, :, :action_dim]


def _difference(reference: torch.Tensor, changed: torch.Tensor) -> dict[str, object]:
    delta = changed.float() - reference.float()
    per_dim = delta.square().mean(dim=(0, 1)).sqrt()
    return {
        "normalized_rmse": float(delta.square().mean().sqrt().item()),
        "normalized_per_dim_rmse": per_dim.cpu().tolist(),
    }


def _physical_difference(reference: torch.Tensor, changed: torch.Tensor) -> dict[str, float]:
    delta = changed.float() - reference.float()
    return {
        "physical_rmse": float(delta.square().mean().sqrt().item()),
        "translation_endpoint_m": float(delta[0, :, :3].sum(dim=0).norm().item()),
        "rotation_sum_rad": float(delta[0, :, 3:6].sum(dim=0).norm().item()),
        "gripper_sum_m": float(delta[0, :, 6].sum().abs().item()),
    }


def _mean(records: list[dict[str, object]], condition: str, key: str) -> float:
    return float(np.mean([record[condition][key] for record in records]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--repo-id", default="threading_real/threading_combined_pi05_15hz_sg5_nozero"
    )
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("--pairs must be at least 2")

    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    dataset = LeRobotDataset(
        args.repo_id, root=dataset_root, video_backend="pyav"
    )
    policy = PI05Policy.from_pretrained(checkpoint).to(args.device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    # Interior, uniformly spaced frames avoid padded action chunks at episode ends.
    episode_ranges = [
        (int(ep["dataset_from_index"]), int(ep["dataset_to_index"]))
        for ep in dataset.meta.episodes
        if int(ep["dataset_to_index"]) - int(ep["dataset_from_index"])
        > policy.config.chunk_size * 2
    ]
    chosen_episodes = np.linspace(0, len(episode_ranges) - 1, args.pairs, dtype=int)
    base_indices = []
    for order, episode_index in enumerate(chosen_episodes):
        start, stop = episode_ranges[int(episode_index)]
        fraction = 0.2 + 0.6 * order / max(args.pairs - 1, 1)
        base_indices.append(int(start + fraction * (stop - start - policy.config.chunk_size)))
    donor_indices = list(reversed(base_indices))
    if any(base == donor for base, donor in zip(base_indices, donor_indices)):
        donor_indices = donor_indices[1:] + donor_indices[:1]

    camera_keys = tuple(policy.config.image_features)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    records: list[dict[str, object]] = []

    with torch.inference_mode():
        for pair_number, (base_index, donor_index) in enumerate(
            zip(base_indices, donor_indices, strict=True)
        ):
            base = dict(dataset[base_index])
            donor = dict(dataset[donor_index])
            image_swap = dict(base)
            state_swap = dict(base)
            both_swap = dict(base)
            for key in camera_keys:
                image_swap[key] = donor[key]
                both_swap[key] = donor[key]
            state_swap["observation.state"] = donor["observation.state"]
            both_swap["observation.state"] = donor["observation.state"]

            batches = {
                "base": preprocess(base),
                "image_swap": preprocess(image_swap),
                "state_swap": preprocess(state_swap),
                "both_swap": preprocess(both_swap),
            }
            noise_shape = (
                1,
                policy.config.chunk_size,
                policy.config.max_action_dim,
            )
            # PI0.5's native sampling starts from float32 noise even when the
            # transformer weights use bfloat16.
            fixed_noise = torch.randn(
                noise_shape,
                generator=generator,
                device=args.device,
                dtype=torch.float32,
            )
            other_noise = torch.randn(
                noise_shape,
                generator=generator,
                device=args.device,
                dtype=torch.float32,
            )
            predictions = {
                name: _predict_normalized(policy, batch, fixed_noise)
                for name, batch in batches.items()
            }
            predictions["noise_swap"] = _predict_normalized(
                policy, batches["base"], other_noise
            )
            # Determinism control: identical condition and noise should reproduce exactly.
            repeat = _predict_normalized(policy, batches["base"], fixed_noise)

            physical = {
                name: postprocess(value).detach().cpu()
                for name, value in predictions.items()
            }
            record: dict[str, object] = {
                "pair": pair_number,
                "base_index": base_index,
                "donor_index": donor_index,
                "base_episode": int(base["episode_index"].item()),
                "donor_episode": int(donor["episode_index"].item()),
                "repeat_max_abs_error": float(
                    (repeat.float() - predictions["base"].float()).abs().max().item()
                ),
            }
            for condition in ("image_swap", "state_swap", "both_swap", "noise_swap"):
                record[condition] = {
                    **_difference(predictions["base"], predictions[condition]),
                    **_physical_difference(physical["base"], physical[condition]),
                }
            records.append(record)
            print(f"completed condition pair {pair_number + 1}/{args.pairs}", flush=True)

    conditions = ("image_swap", "state_swap", "both_swap", "noise_swap")
    summary = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "pairs": args.pairs,
        "seed": args.seed,
        "method": "Each condition swap uses exactly the same initial flow noise as its base prediction.",
        "mean": {
            condition: {
                key: _mean(records, condition, key)
                for key in (
                    "normalized_rmse",
                    "physical_rmse",
                    "translation_endpoint_m",
                    "rotation_sum_rad",
                    "gripper_sum_m",
                )
            }
            for condition in conditions
        },
        "condition_to_noise_ratio": {
            condition: _mean(records, condition, "normalized_rmse")
            / _mean(records, "noise_swap", "normalized_rmse")
            for condition in ("image_swap", "state_swap", "both_swap")
        },
        "records": records,
    }
    serialized = json.dumps(summary, indent=2)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
