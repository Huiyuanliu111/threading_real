#!/usr/bin/env python3
"""Build a sidecar chunk-selector dataset from a Threading ARP checkpoint."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch

from pushbox.chunk_dataset import ChunkFeatureWriter
from scripts.eval_policy import load_policy
from threading_task.chunk_labels import (
    LABEL_RULE_VERSION,
    NEEDLE_CENTER_OFFSET,
    NEEDLE_HANDLE_OFFSET,
    RING_CENTER_OFFSET,
    distance_to_next_event,
    spatial_labels,
    spatial_metrics,
)
from threading_task.dataset import _as_chw_float, _read_rows, sorted_demo_keys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _create_metric_datasets(writer: ChunkFeatureWriter) -> None:
    for name, dtype, fillvalue in (
        ("eef_handle_distance", "f4", np.nan),
        ("lift_height", "f4", np.nan),
        ("needle_ring_distance", "f4", np.nan),
        ("distance_to_next_event", "i4", -1),
    ):
        writer.h5.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=dtype,
            fillvalue=fillvalue,
        )


def _context_rows(rows: np.ndarray, n_obs_steps: int) -> np.ndarray:
    offsets = np.arange(n_obs_steps - 1, -1, -1, dtype=np.int64)
    return np.maximum(rows[:, None] - offsets[None], 0)


def _state_batch(obs: h5py.Group, rows: np.ndarray, state_dim: int) -> np.ndarray:
    if state_dim == 8:
        state = np.concatenate(
            (
                _read_rows(obs["robot0_eef_pos"], rows),
                _read_rows(obs["robot0_eef_quat"], rows),
                _read_rows(obs["robot0_gripper_qpos"], rows)[..., :1],
            ),
            axis=-1,
        )
    elif state_dim == 9:
        state = np.concatenate(
            (
                _read_rows(obs["robot0_joint_pos"], rows),
                _read_rows(obs["robot0_gripper_qpos"], rows),
            ),
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported ARP agent state dimension: {state_dim}")
    if state.shape[-1] != state_dim:
        raise ValueError(f"Expected {state_dim}D state, got {state.shape}")
    return state.astype(np.float32, copy=False)


def _extract_features(
    policy: torch.nn.Module,
    obs: h5py.Group,
    rows: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    context = _context_rows(rows, policy.n_obs_steps)
    batch_size = len(rows)
    images: dict[str, torch.Tensor] = {}
    for output_key, dataset_key in zip(policy.rgb_keys, policy.camera_obs_keys):
        size = int(policy.image_shapes[output_key][-1])
        raw = _read_rows(obs[dataset_key], context)
        flat = raw.reshape(-1, *raw.shape[2:])
        chw = _as_chw_float(flat, dataset_key, size).reshape(
            batch_size, policy.n_obs_steps, 3, size, size
        )
        images[output_key] = torch.from_numpy(chw).to(device)

    state = torch.from_numpy(
        _state_batch(obs, context, int(policy.agent_state_dim))
    ).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    normalized_images = {key: (value - mean) / std for key, value in images.items()}
    normalized_state = policy.normalizer["agent_pos"].normalize(
        state.flatten(0, 1)
    ).reshape(batch_size, policy.n_obs_steps, policy.agent_state_dim)
    with torch.inference_mode():
        return policy._visual_tokens(normalized_images, normalized_state).detach().cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/threading/threading_d0.hdf5"))
    parser.add_argument("--checkpoint", type=Path, default=Path("epoch20_arp.ckpt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chunk_selector/threading_arp_spatial_v1_chunk_4_10.hdf5"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--limit-demos", type=int)
    parser.add_argument("--candidate-chunks", type=int, nargs="+", default=[4, 10])
    parser.add_argument("--full-chunk", type=int, default=10)
    parser.add_argument("--pick-chunk", type=int, default=4)
    parser.add_argument("--insert-chunk", type=int, default=4)
    parser.add_argument("--grasp-distance", type=float, default=0.10)
    parser.add_argument("--lift-threshold", type=float, default=0.05)
    parser.add_argument("--insert-approach-distance", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.sample_every <= 0:
        parser.error("--batch-size and --sample-every must be positive")
    if args.limit_demos is not None and args.limit_demos <= 0:
        parser.error("--limit-demos must be positive")
    candidates = tuple(sorted(set(int(value) for value in args.candidate_chunks)))
    region_chunks = {
        "free_approach": int(args.full_chunk),
        "pick_precision": int(args.pick_chunk),
        "free_transport": int(args.full_chunk),
        "insert_precision": int(args.insert_chunk),
    }
    missing = sorted(set(region_chunks.values()) - set(candidates))
    if missing:
        parser.error(f"region chunks are absent from --candidate-chunks: {missing}")

    dataset_path = args.dataset.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    summary_path = output_path.with_suffix(".summary.json")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    for required in (dataset_path, checkpoint_path):
        if not required.is_file():
            parser.error(f"file not found: {required}")
    existing = [path for path in (output_path, summary_path, partial_path) if path.exists()]
    if existing and not args.overwrite:
        parser.error(f"output already exists: {existing[0]}")
    if args.overwrite:
        for path in existing:
            path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    policy = load_policy(
        str(checkpoint_path),
        device=str(device),
        weights=args.weights,
        use_checkpoint_config=True,
    )
    if not hasattr(policy, "_visual_tokens"):
        parser.error("checkpoint policy does not expose Threading ARP visual tokens")
    if policy.camera_obs_keys is None:
        parser.error("checkpoint does not record camera observation keys")
    if max(candidates) > int(policy.horizon):
        parser.error(
            f"candidate chunks {candidates} exceed ARP horizon {policy.horizon}"
        )

    metadata: dict[str, Any] = {
        "label_source": LABEL_RULE_VERSION,
        "policy_type": "threading_arp",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "weights": args.weights,
        "source_dataset": str(dataset_path),
        "candidate_chunks": list(candidates),
        "region_chunks": region_chunks,
        "spatial_thresholds_m": {
            "grasp_distance": args.grasp_distance,
            "lift_threshold": args.lift_threshold,
            "insert_approach_distance": args.insert_approach_distance,
        },
        "geometry_offsets_local_m": {
            "needle_handle": NEEDLE_HANDLE_OFFSET.tolist(),
            "needle_center": NEEDLE_CENTER_OFFSET.tolist(),
            "ring_center": RING_CENTER_OFFSET.tolist(),
        },
        "sample_every": args.sample_every,
        "visual_token_shape": None,
        "rgb_keys": list(policy.rgb_keys),
        "camera_obs_keys": list(policy.camera_obs_keys),
        "n_obs_steps": int(policy.n_obs_steps),
    }

    writer: ChunkFeatureWriter | None = None
    chunk_counts: Counter[int] = Counter()
    region_counts: Counter[str] = Counter()
    episode_summaries: list[dict[str, Any]] = []
    sample_count = 0
    try:
        with h5py.File(dataset_path, "r") as h5:
            demos = sorted_demo_keys(h5["data"])
            if args.limit_demos is not None:
                demos = demos[: args.limit_demos]
            for demo_number, demo_name in enumerate(demos, start=1):
                demo = h5["data"][demo_name]
                obs = demo["obs"]
                length = len(demo["actions"])
                handle_distance, lift_height, insert_distance = spatial_metrics(
                    np.asarray(obs["object"]),
                    np.asarray(obs["robot0_eef_pos"]),
                )
                regions, chunks, events = spatial_labels(
                    handle_distance,
                    lift_height,
                    insert_distance,
                    grasp_distance=args.grasp_distance,
                    lift_threshold=args.lift_threshold,
                    insert_approach_distance=args.insert_approach_distance,
                    region_chunks=region_chunks,
                    demo=demo_name,
                )
                sampled_rows = np.arange(0, length, args.sample_every, dtype=np.int64)
                episode_counts = Counter(str(regions[row]) for row in sampled_rows)
                episode_summaries.append(
                    {
                        "demo": demo_name,
                        "length": length,
                        **events,
                        "sample_count": len(sampled_rows),
                        "region_counts": dict(sorted(episode_counts.items())),
                    }
                )

                for start in range(0, len(sampled_rows), args.batch_size):
                    rows = sampled_rows[start : start + args.batch_size]
                    features = _extract_features(policy, obs, rows, device)
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
                    class_ids = [candidates.index(int(chunk)) for chunk in batch_chunks]
                    batch_regions = [str(region) for region in regions[rows]]
                    old_size, new_size = writer.append(
                        features,
                        labels=class_ids,
                        episode_ids=[demo_name] * len(rows),
                        decision_steps=rows.tolist(),
                        subtasks=batch_regions,
                    )
                    writer.h5["eef_handle_distance"][old_size:new_size] = handle_distance[rows]
                    writer.h5["lift_height"][old_size:new_size] = lift_height[rows]
                    writer.h5["needle_ring_distance"][old_size:new_size] = insert_distance[rows]
                    writer.h5["distance_to_next_event"][old_size:new_size] = (
                        distance_to_next_event(rows, events)
                    )
                    chunk_counts.update(int(value) for value in batch_chunks)
                    region_counts.update(batch_regions)
                    sample_count += len(rows)
                print(
                    f"[{demo_number:04d}/{len(demos):04d}] {demo_name}: "
                    f"events={events} samples={len(sampled_rows)}",
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
        "num_demos": len(episode_summaries),
        "num_samples": sample_count,
        "label_chunk_counts": {
            str(chunk): chunk_counts.get(chunk, 0) for chunk in candidates
        },
        "region_counts": dict(sorted(region_counts.items())),
        "episodes": episode_summaries,
        "feature_dataset": str(output_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"dataset: {output_path}")
    print(f"summary: {summary_path}")
    print(f"samples: {sample_count}")
    print(f"label chunks: {dict(sorted(chunk_counts.items()))}")
    print(f"regions: {dict(sorted(region_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
