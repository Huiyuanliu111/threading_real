#!/usr/bin/env python3
"""Cache frozen ARP MVT point-cloud features with spatial probability labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from chunk_selector.chunk_dataset import ChunkFeatureWriter
from chunk_selector.mvt_features import selector_tokens


class LabeledPointcloudDataset(Dataset):
    """All labeled observations, including final frames (no action horizon needed)."""
    def __init__(self, path, labels):
        self.path = str(path)
        self.rows = labels.reset_index(drop=True)
        self._file = None
        with h5py.File(self.path, "r") as source:
            for key, rows in self.rows.groupby("episode_key"):
                if key not in source or len(rows) != len(source[key]["points"]):
                    raise ValueError(f"labels do not cover all source frames for {key}")
                if not np.array_equal(np.sort(rows.frame_index), np.arange(len(rows))):
                    raise ValueError(f"duplicate or missing frame indices for {key}")
            if set(source) != set(self.rows.episode_key):
                raise ValueError("label/source episode sets differ")

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __getitem__(self, index):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        row = self.rows.iloc[index]
        group = self._file[row.episode_key]
        frame = int(row.frame_index)
        return dict(points=torch.from_numpy(group["points"][frame].astype(np.float32)[None]),
                    colors=torch.from_numpy(group["colors"][frame].astype(np.float32)[None] / 255),
                    valid_points=torch.tensor([int(group["valid_points"][frame])]), index=index)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def load_mvt_policy(checkpoint, task, device, weights):
    if task == "threading":
        from scripts.arp.policy_loader import load_policy
    else:
        sys.path.insert(0, str(ROOT.parent / "maze_real"))
        from maze_policy.checkpoint import load_policy
    policy = load_policy(checkpoint, device=device, weights=weights)
    if not hasattr(policy, "vit") or not hasattr(policy, "_visual"):
        raise ValueError("selector feature extraction requires an MVT ARP checkpoint")
    policy.requires_grad_(False)
    return policy


def extract(checkpoint, labels_dir, output, *, device="cpu", weights="model",
            pool_grid=0, batch_size=2, num_workers=0, dataset_path=None):
    labels_dir = Path(labels_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = json.loads((labels_dir / "summary.json").read_text())
    if summary.get("label_source") not in {"spatial_rule", "progress_rule"}:
        raise ValueError("expected spatial_rule or progress_rule label metadata")
    candidates = tuple(summary["candidate_chunks"])
    labels = pq.read_table(labels_dir / "labels.parquet").to_pandas()
    probabilities = labels[[f"chunk_probability_{n}" for n in candidates]].to_numpy(np.float32)
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(1), 1)):
        raise ValueError("invalid spatial soft labels")
    policy = load_mvt_policy(checkpoint, summary["rule"]["task"], device, weights)
    if max(candidates) > policy.horizon:
        raise ValueError("maximum chunk exceeds policy horizon")
    source_dataset = Path(dataset_path or summary["source_dataset"]).expanduser().resolve()
    dataset = LabeledPointcloudDataset(source_dataset, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    writer = None
    try:
        for batch in tqdm(loader, desc="MVT selector features"):
            indices = batch.pop("index").numpy()
            obs = {key: value.to(device) for key, value in batch.items()}
            with torch.inference_mode():
                _, visual_tokens, _ = policy._visual(obs)
                tokens, ids = selector_tokens(visual_tokens,
                    side=policy.image_size // policy.patch_size, pool_grid=pool_grid)
            if writer is None:
                writer = ChunkFeatureWriter(output, feature_shape=tuple(tokens.shape[1:]),
                    candidate_chunks=candidates, metadata={
                        "label_source": summary["label_source"], summary["label_source"]: summary["rule"],
                        "source_dataset": str(source_dataset),
                        "source_checkpoint": str(Path(checkpoint).resolve()), "weights": weights,
                        "source_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                        "source_labels": str(labels_dir), "mvt_pool_grid": pool_grid,
                        "fit_episode_ids": summary["fit_episode_ids"],
                        "validation_episode_ids": summary["validation_episode_ids"],
                    })
            rows = labels.iloc[indices]
            writer.append(tokens, labels=probabilities[indices].argmax(1),
                          target_probabilities=probabilities[indices],
                          episode_ids=rows.episode_key.tolist(),
                          decision_steps=rows.frame_index.tolist(), **ids)
    finally:
        if writer is not None:
            writer.close()
        dataset.close()
    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path,
                        help="override source HDF5 path after moving the dataset")
    parser.add_argument("--pool-grid", type=int, default=0, help="0: all MVT output tokens; e.g. 4: pool each virtual view to 4x4")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    print(f"cached {extract(**vars(parser.parse_args()))} frames")


if __name__ == "__main__":
    main()
