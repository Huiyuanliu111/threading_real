#!/usr/bin/env python3
"""Cache frozen pi0.5 visual tokens with TCP-motion soft chunk targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from torch.utils.data import DataLoader

from chunk_selector.chunk_dataset import ChunkFeatureWriter


def _pool_visual_tokens(tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Spatially pool square SigLIP patch tokens to a smaller square grid."""
    batch_size, token_count, feature_dim = tokens.shape
    source_grid = int(round(token_count**0.5))
    if source_grid * source_grid != token_count:
        raise ValueError(f"Expected square visual token grid, got {token_count} tokens")
    if not 1 <= grid_size <= source_grid:
        raise ValueError(f"pool grid must lie in [1, {source_grid}], got {grid_size}")
    maps = tokens.transpose(1, 2).reshape(batch_size, feature_dim, source_grid, source_grid)
    return F.adaptive_avg_pool2d(maps, (grid_size, grid_size)).flatten(2).transpose(1, 2)


def _load_labels(path: Path, candidates: tuple[int, ...]) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    required = {"index", "episode_index", "frame_index", "soft_chunk_size"}
    probability_columns = [f"chunk_probability_{chunk}" for chunk in candidates]
    missing = sorted(required.union(probability_columns) - set(table.column_names))
    if missing:
        raise ValueError(f"Soft-label table is missing columns: {missing}")
    result = {
        name: np.asarray(table[name].to_pylist())
        for name in required.union(probability_columns)
    }
    order = np.argsort(result["index"], kind="stable")
    result = {name: values[order] for name, values in result.items()}
    expected_indices = np.arange(len(table), dtype=np.int64)
    if not np.array_equal(result["index"].astype(np.int64), expected_indices):
        raise ValueError("Label indices must be unique and contiguous from zero")
    probabilities = np.column_stack([result[name] for name in probability_columns]).astype(np.float32)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("Label probabilities must be finite, non-negative, and sum to one")
    result["target_probabilities"] = probabilities
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d"),
    )
    parser.add_argument(
        "--repo-id",
        default="threading_real/threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            "data/threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d_tcp_chunk_soft_labels_4_10_smoothed/labels.parquet"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-chunks", type=int, nargs="+", default=[4, 10])
    parser.add_argument("--pool-grid", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.num_workers < 0 or args.pool_grid <= 0:
        parser.error("batch-size and pool-grid must be positive; num-workers cannot be negative")
    candidates = tuple(sorted(set(int(value) for value in args.candidate_chunks)))
    if len(candidates) < 2 or candidates[0] <= 0:
        parser.error("candidate-chunks must contain at least two positive values")

    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    output = args.output.expanduser().resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    for required in (checkpoint, dataset_root, labels_path):
        if not required.exists():
            parser.error(f"path not found: {required}")
    for existing in (output, partial):
        if existing.exists():
            parser.error(f"refusing to overwrite existing output: {existing}")

    labels = _load_labels(labels_path, candidates)
    dataset = LeRobotDataset(args.repo_id, root=dataset_root, video_backend="pyav")
    if len(dataset) != len(labels["index"]):
        raise ValueError(f"Dataset has {len(dataset)} frames but labels have {len(labels['index'])}")

    policy = PI05Policy.from_pretrained(checkpoint).to(args.device).eval()
    policy.requires_grad_(False)
    image_keys = list(policy.config.image_features)
    if not image_keys:
        raise ValueError("pi0.5 checkpoint has no configured image features")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )

    writer: ChunkFeatureWriter | None = None
    offset = 0
    try:
        for raw_batch in loader:
            batch_size = int(raw_batch[image_keys[0]].shape[0])
            image_batch = {key: raw_batch[key] for key in image_keys if key in raw_batch}
            images, image_masks = policy._preprocess_images(image_batch)
            pooled_views = []
            valid_views = []
            with torch.inference_mode():
                for image, mask in zip(images, image_masks, strict=True):
                    tokens = policy.model.paligemma_with_expert.embed_image(image)
                    pooled_views.append(_pool_visual_tokens(tokens, args.pool_grid).float())
                    valid_views.append(mask)
            features = torch.cat(pooled_views, dim=1).detach().cpu()
            tokens_per_camera = args.pool_grid * args.pool_grid
            camera_ids = torch.arange(len(pooled_views), dtype=torch.long).repeat_interleave(
                tokens_per_camera
            )
            spatial_ids = torch.arange(tokens_per_camera, dtype=torch.long).repeat(
                len(pooled_views)
            )
            padding_mask = torch.cat(
                [~mask[:, None].expand(-1, tokens_per_camera) for mask in valid_views], dim=1
            ).detach().cpu()
            if padding_mask.any():
                raise ValueError("This extractor requires every configured camera to be present")

            stop = offset + batch_size
            target_probabilities = labels["target_probabilities"][offset:stop]
            hard_labels = target_probabilities.argmax(axis=1).astype(np.int64)
            if writer is None:
                writer = ChunkFeatureWriter(
                    partial,
                    feature_shape=(int(features.shape[1]), int(features.shape[2])),
                    candidate_chunks=candidates,
                    metadata={
                        "base_policy": "pi05",
                        "checkpoint": str(checkpoint),
                        "dataset_root": str(dataset_root),
                        "labels": str(labels_path),
                        "pool_grid": args.pool_grid,
                        "image_keys": image_keys,
                        "target_type": "tcp_motion_soft_probabilities",
                    },
                )
            writer.append(
                features,
                labels=hard_labels,
                target_probabilities=target_probabilities,
                episode_ids=[str(value) for value in labels["episode_index"][offset:stop]],
                decision_steps=labels["frame_index"][offset:stop].astype(np.int64),
                camera_ids=camera_ids,
                spatial_ids=spatial_ids,
            )
            offset = stop
            print(f"cached {offset}/{len(dataset)} frames", flush=True)
    except BaseException:
        if writer is not None:
            writer.close()
        raise
    if writer is None:
        raise RuntimeError("Dataset produced no batches")
    writer.close()
    partial.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "samples": offset,
                "feature_shape": list(features.shape[1:]),
                "candidate_chunks": list(candidates),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
