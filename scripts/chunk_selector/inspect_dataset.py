#!/usr/bin/env python3
"""Summarize a cached chunk-selector HDF5 dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import h5py


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    path = args.dataset.expanduser().resolve()
    with h5py.File(path, "r") as h5:
        candidates = tuple(json.loads(h5.attrs["candidate_chunks"]))
        feature_shape = tuple(json.loads(h5.attrs["feature_shape"]))
        metadata = json.loads(h5.attrs.get("metadata", "{}"))
        labels = [int(value) for value in h5["labels"][:]]
        subtasks = h5["subtasks"].asstr()[:].tolist()
        episode_ids = h5["episode_ids"].asstr()[:].tolist()

    invalid_labels = sorted(set(labels) - set(range(len(candidates))))
    if invalid_labels:
        raise ValueError(f"Invalid class ids in dataset: {invalid_labels}")

    chunk_counts = Counter(candidates[label] for label in labels)
    print(f"path: {path}")
    print(f"samples: {len(labels)}")
    print(f"episodes/demos: {len(set(episode_ids))}")
    print(f"feature_shape: {feature_shape}")
    print(f"candidate_chunks: {list(candidates)}")
    print(f"subtask_counts: {dict(sorted(Counter(subtasks).items()))}")
    print(f"label_chunk_counts: {dict(sorted(chunk_counts.items()))}")
    print(f"metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
