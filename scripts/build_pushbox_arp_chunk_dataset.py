#!/usr/bin/env python3
"""Build a spatially labelled PushBox ARP chunk-selector dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from chunk_selector.chunk_dataset import ChunkFeatureWriter
from pushbox.lerobot_io import load_lerobot_episodes
from scripts.eval_policy import load_policy


SYMMETRIC_LABEL_RULE_VERSION = "pushbox_symmetric_crossing_v1"
ASYMMETRIC_LABEL_RULE_VERSION = "pushbox_asymmetric_crossing_v2"
REGIONS = ("free_before_crossing", "crossing_precision", "free_after_crossing")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spatial_labels(
    box_y: np.ndarray,
    *,
    crossing_threshold: float | None = None,
    crossing_region: tuple[float, float] | None = None,
    small_chunk: int,
    full_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    box_y = np.asarray(box_y, dtype=np.float64).reshape(-1)
    if crossing_region is not None:
        crossing_start, crossing_end = crossing_region
        if crossing_start >= crossing_end:
            raise ValueError("crossing_region requires START < END")
    else:
        if crossing_threshold is None or crossing_threshold <= 0:
            raise ValueError("crossing_threshold must be positive")
        crossing_start, crossing_end = -crossing_threshold, crossing_threshold
    regions = np.full(len(box_y), "crossing_precision", dtype=object)
    regions[box_y < crossing_start] = "free_before_crossing"
    regions[box_y > crossing_end] = "free_after_crossing"
    chunks = np.where(
        regions == "crossing_precision", int(small_chunk), int(full_chunk)
    ).astype(np.int64)
    return regions, chunks


def _context_rows(rows: np.ndarray, n_obs_steps: int) -> np.ndarray:
    offsets = np.arange(n_obs_steps - 1, -1, -1, dtype=np.int64)
    return np.maximum(rows[:, None] - offsets[None], 0)


def _extract_features(
    policy: torch.nn.Module,
    episode: dict[str, np.ndarray],
    rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    context = _context_rows(rows, int(policy.n_obs_steps))
    batch_size = len(rows)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    flat_images: dict[str, torch.Tensor] = {}
    for key in ("top45", "sideview"):
        images = torch.from_numpy(episode[key][context]).to(device=device)
        images = (images - mean) / std
        flat_images[key] = images.reshape(-1, *images.shape[2:])
    with torch.inference_mode():
        features = policy.obs_encoder(flat_images)
        features = policy.obs_feat_linear(features)
        return features.reshape(
            batch_size, int(policy.n_obs_steps), int(policy.policy.cfg.n_embd)
        ).detach().cpu()


def _create_metric_datasets(writer: ChunkFeatureWriter) -> None:
    for name in ("box_x", "box_y"):
        writer.h5.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype="f4",
            fillvalue=np.nan,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/datagen"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/huiyuan/实验结果/pushbox/23-08-49/checkpoints/"
            "epoch=0140-val_loss=-30.500.ckpt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/chunk_selector/"
            "pushbox_arp_spatial_v1_threshold_0p15_chunk_5_19.hdf5"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-frames-per-episode", type=int, default=500)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--crossing-threshold", type=float, default=0.15)
    parser.add_argument(
        "--crossing-region",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        help=(
            "asymmetric inclusive SMALL-chunk interval; overrides "
            "--crossing-threshold"
        ),
    )
    parser.add_argument("--small-chunk", type=int, default=5)
    parser.add_argument("--full-chunk", type=int, default=19)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.sample_every <= 0:
        parser.error("--batch-size and --sample-every must be positive")
    if args.max_frames_per_episode <= 0:
        parser.error("--max-frames-per-episode must be positive")
    if args.limit_episodes is not None and args.limit_episodes <= 0:
        parser.error("--limit-episodes must be positive")
    if args.crossing_threshold <= 0:
        parser.error("--crossing-threshold must be positive")
    if args.crossing_region is not None and not (
        args.crossing_region[0] < args.crossing_region[1]
    ):
        parser.error("--crossing-region requires START < END")
    if not 0 < args.small_chunk < args.full_chunk:
        parser.error("chunk sizes must satisfy 0 < SMALL < FULL")

    dataset_path = args.dataset.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    summary_path = output_path.with_suffix(".summary.json")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if not (dataset_path / "meta" / "info.json").is_file():
        parser.error(f"LeRobot dataset not found: {dataset_path}")
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")
    existing = [path for path in (output_path, summary_path, partial_path) if path.exists()]
    if existing and not args.overwrite:
        parser.error(f"output already exists: {existing[0]}")
    if args.overwrite:
        for path in existing:
            path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    policy = load_policy(
        str(checkpoint_path),
        device=str(device),
        weights=args.weights,
        use_checkpoint_config=True,
    )
    maximum_chunk_label = int(
        getattr(policy, "max_selector_chunk_label", policy.max_selector_chunk)
    )
    if args.full_chunk > maximum_chunk_label:
        parser.error(
            f"full chunk {args.full_chunk} exceeds maximum "
            f"label {maximum_chunk_label}"
        )

    episodes = load_lerobot_episodes(
        dataset_path, max_frames_per_ep=args.max_frames_per_episode
    )
    if args.limit_episodes is not None:
        episodes = episodes[: args.limit_episodes]
    if not episodes:
        raise RuntimeError("source dataset has no episodes")

    candidates = (int(args.small_chunk), int(args.full_chunk))
    crossing_region = (
        (-float(args.crossing_threshold), float(args.crossing_threshold))
        if args.crossing_region is None
        else (float(args.crossing_region[0]), float(args.crossing_region[1]))
    )
    label_rule_version = (
        SYMMETRIC_LABEL_RULE_VERSION
        if args.crossing_region is None
        else ASYMMETRIC_LABEL_RULE_VERSION
    )
    metadata: dict[str, Any] = {
        "label_source": label_rule_version,
        "policy_type": "pushbox_arp",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "weights": args.weights,
        "source_dataset": str(dataset_path),
        "candidate_chunks": list(candidates),
        "region_chunks": {
            "free_before_crossing": args.full_chunk,
            "crossing_precision": args.small_chunk,
            "free_after_crossing": args.full_chunk,
        },
        "spatial_thresholds_m": {
            "crossing_before": crossing_region[0],
            "crossing_after": crossing_region[1],
            "small_region_inclusive": True,
        },
        "max_frames_per_episode": args.max_frames_per_episode,
        "sample_every": args.sample_every,
        "n_obs_steps": int(policy.n_obs_steps),
        "rgb_keys": ["top45", "sideview"],
        "visual_token_shape": None,
        "seed": args.seed,
    }

    writer: ChunkFeatureWriter | None = None
    chunk_counts: Counter[int] = Counter()
    region_counts: Counter[str] = Counter()
    episode_summaries: list[dict[str, Any]] = []
    sample_count = 0
    try:
        for episode_index, episode in enumerate(episodes):
            length = len(episode["action"])
            regions, chunks = spatial_labels(
                episode["box_pos"][:, 1],
                crossing_threshold=args.crossing_threshold,
                crossing_region=(
                    None if args.crossing_region is None else crossing_region
                ),
                small_chunk=args.small_chunk,
                full_chunk=args.full_chunk,
            )
            sampled_rows = np.arange(0, length, args.sample_every, dtype=np.int64)
            episode_regions = Counter(str(regions[row]) for row in sampled_rows)
            episode_summaries.append(
                {
                    "episode": episode_index,
                    "length": length,
                    "sample_count": len(sampled_rows),
                    "box_y_min": float(np.min(episode["box_pos"][:, 1])),
                    "box_y_max": float(np.max(episode["box_pos"][:, 1])),
                    "region_counts": dict(sorted(episode_regions.items())),
                }
            )
            episode_id = f"episode_{episode_index:06d}"
            for start in range(0, len(sampled_rows), args.batch_size):
                rows = sampled_rows[start : start + args.batch_size]
                features = _extract_features(policy, episode, rows, device)
                if writer is None:
                    feature_shape = (int(features.shape[1]), int(features.shape[2]))
                    metadata["visual_token_shape"] = list(feature_shape)
                    writer = ChunkFeatureWriter(
                        partial_path,
                        feature_shape=feature_shape,
                        candidate_chunks=candidates,
                        metadata=metadata,
                    )
                    _create_metric_datasets(writer)
                batch_chunks = chunks[rows]
                batch_regions = [str(region) for region in regions[rows]]
                old_size, new_size = writer.append(
                    features,
                    labels=[candidates.index(int(chunk)) for chunk in batch_chunks],
                    episode_ids=[episode_id] * len(rows),
                    decision_steps=rows.tolist(),
                    subtasks=batch_regions,
                )
                writer.h5["box_x"][old_size:new_size] = episode["box_pos"][rows, 0]
                writer.h5["box_y"][old_size:new_size] = episode["box_pos"][rows, 1]
                chunk_counts.update(int(value) for value in batch_chunks)
                region_counts.update(batch_regions)
                sample_count += len(rows)
            print(
                f"[{episode_index + 1:03d}/{len(episodes):03d}] {episode_id}: "
                f"length={length} regions={dict(sorted(episode_regions.items()))}",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("no selector samples were generated")
    os.replace(partial_path, output_path)
    summary = {
        "metadata": metadata,
        "num_episodes": len(episode_summaries),
        "num_samples": sample_count,
        "label_chunk_counts": {
            str(chunk): chunk_counts.get(chunk, 0) for chunk in candidates
        },
        "region_counts": dict(sorted(region_counts.items())),
        "episodes": episode_summaries,
        "feature_dataset": str(output_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"dataset: {output_path}")
    print(f"summary: {summary_path}")
    print(f"samples: {sample_count}")
    print(f"label chunks: {dict(sorted(chunk_counts.items()))}")
    print(f"regions: {dict(sorted(region_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
