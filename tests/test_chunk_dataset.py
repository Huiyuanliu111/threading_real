from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from chunk_selector.chunk_dataset import (
    ChunkFeatureDataset,
    ChunkFeatureWriter,
    CoarseChunkFeatureCollector,
    episode_split_indices,
    parse_subtask_chunks,
)


def test_chunk_feature_dataset_roundtrip_and_episode_split(tmp_path):
    path = tmp_path / "features.hdf5"
    with ChunkFeatureWriter(
        path,
        feature_shape=(3, 4),
        candidate_chunks=(1, 2, 4),
        metadata={"policy": "unit-test"},
    ) as writer:
        writer.append(
            torch.arange(4 * 3 * 4).reshape(4, 3, 4),
            labels=[0, 1, 2, 1],
            utilities=np.asarray(
                [
                    [1.0, 0.0, -1.0],
                    [0.0, 1.0, -1.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 1.0, -1.0],
                ],
                dtype=np.float32,
            ),
            sample_weights=[1.0, 0.5, 0.25, 1.0],
            episode_ids=["ep0", "ep0", "ep1", "ep1"],
            decision_steps=[0, 1, 0, 1],
            subtasks=["approach", "approach", "insert", "insert"],
        )

    dataset = ChunkFeatureDataset(path)
    assert len(dataset) == 4
    assert dataset.candidate_chunks == (1, 2, 4)
    assert dataset.metadata["policy"] == "unit-test"
    assert dataset[2]["features"].shape == (3, 4)
    assert dataset[2]["label"].item() == 2
    assert dataset[2]["has_utilities"].item()
    assert dataset[2]["sample_weight"].item() == pytest.approx(0.25)
    assert dataset[2]["episode_id"] == "ep1"

    train_indices, val_indices = episode_split_indices(path, val_ratio=0.5, seed=3)
    train_episodes = {dataset[index]["episode_id"] for index in train_indices}
    val_episodes = {dataset[index]["episode_id"] for index in val_indices}
    assert train_episodes
    assert val_episodes
    assert train_episodes.isdisjoint(val_episodes)


def test_coarse_collector_maps_subtask_chunk_to_class(tmp_path):
    path = tmp_path / "coarse.hdf5"
    with CoarseChunkFeatureCollector(
        path,
        candidate_chunks=(1, 4),
        subtask_chunks={"approach": 1, "insert": 4},
    ) as collector:
        collector.record(
            torch.randn(1, 5, 6),
            episode_id="ep0",
            decision_step=10,
            subtask="insert",
        )
    dataset = ChunkFeatureDataset(path)
    assert dataset[0]["label"].item() == 1
    assert dataset[0]["subtask"] == "insert"
    assert dataset.metadata["subtask_chunks"]["insert"] == 4


def test_legacy_dataset_without_sample_weights_defaults_to_one(tmp_path):
    path = tmp_path / "legacy.hdf5"
    with ChunkFeatureWriter(
        path,
        feature_shape=(2, 3),
        candidate_chunks=(1, 2),
    ) as writer:
        writer.append(
            torch.zeros(1, 2, 3),
            labels=[0],
            episode_ids=["ep0"],
            decision_steps=[0],
        )
    with h5py.File(path, "r+") as h5:
        del h5["sample_weights"]
    assert ChunkFeatureDataset(path)[0]["sample_weight"].item() == pytest.approx(1.0)


def test_coarse_collector_rejects_classes_without_positive_labels(tmp_path):
    with pytest.raises(ValueError, match="without positive labels"):
        CoarseChunkFeatureCollector(
            tmp_path / "unused-class.hdf5",
            candidate_chunks=(1, 2, 4),
            subtask_chunks={"approach": 1, "insert": 4},
        )


def test_parse_subtask_chunks():
    assert parse_subtask_chunks(["approach=1", "pick=4", "insert=8"]) == {
        "approach": 1,
        "pick": 4,
        "insert": 8,
    }


def test_parse_subtask_chunks_rejects_invalid_values():
    for assignments in (
        ["approach"],
        ["approach=x"],
        ["approach=0"],
        ["approach=1", "approach=2"],
    ):
        try:
            parse_subtask_chunks(assignments)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {assignments}")
