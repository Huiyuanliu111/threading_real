"""HDF5 storage for cached visual features and chunk-selection labels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA_VERSION = 1


def parse_subtask_chunks(assignments: Sequence[str]) -> dict[str, int]:
    """Parse CLI values such as `approach=1 pick=4 insert=8`."""
    result: dict[str, int] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(
                f"Expected SUBTASK=CHUNK assignment, got {assignment!r}"
            )
        name, raw_chunk = assignment.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate subtask name in {assignment!r}")
        try:
            chunk = int(raw_chunk)
        except ValueError as exc:
            raise ValueError(f"Invalid chunk in {assignment!r}") from exc
        if chunk <= 0:
            raise ValueError(f"Chunk must be positive in {assignment!r}")
        result[name] = chunk
    return result


class CoarseChunkFeatureCollector:
    """Lazy writer for the initial `subtask -> best chunk` supervision."""

    def __init__(
        self,
        path: str | Path,
        *,
        candidate_chunks: Sequence[int],
        subtask_chunks: dict[str, int],
        overwrite: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        self.path = Path(path).expanduser().resolve()
        self.candidate_chunks = tuple(int(value) for value in candidate_chunks)
        self.subtask_chunks = {
            str(name): int(chunk) for name, chunk in subtask_chunks.items()
        }
        missing = sorted(set(self.subtask_chunks.values()) - set(self.candidate_chunks))
        if missing:
            raise ValueError(
                f"Subtask labels {missing} are absent from candidate_chunks"
            )
        unused = sorted(set(self.candidate_chunks) - set(self.subtask_chunks.values()))
        if unused:
            raise ValueError(
                "Coarse supervision cannot train candidate chunks without positive "
                f"labels: {unused}"
            )
        self.overwrite = overwrite
        self.metadata = dict(metadata or {})
        self.metadata["label_source"] = "subtask_best_chunk"
        self.metadata["subtask_chunks"] = self.subtask_chunks
        self._writer: ChunkFeatureWriter | None = None

    def __enter__(self) -> "CoarseChunkFeatureCollector":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def record(
        self,
        features: torch.Tensor,
        *,
        episode_id: str,
        decision_step: int,
        subtask: str,
        camera_ids: torch.Tensor | None = None,
        time_ids: torch.Tensor | None = None,
        spatial_ids: torch.Tensor | None = None,
    ) -> tuple[int, int]:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError(
                "Coarse online collection expects one [1, L, D] feature batch"
            )
        subtask = str(subtask)
        if subtask not in self.subtask_chunks:
            raise KeyError(f"No coarse chunk label configured for subtask {subtask!r}")
        if self._writer is None:
            self._writer = ChunkFeatureWriter(
                self.path,
                feature_shape=(int(features.shape[1]), int(features.shape[2])),
                candidate_chunks=self.candidate_chunks,
                overwrite=self.overwrite,
                metadata=self.metadata,
            )
        label_chunk = self.subtask_chunks[subtask]
        label = self.candidate_chunks.index(label_chunk)
        return self._writer.append(
            features,
            labels=[label],
            episode_ids=[episode_id],
            decision_steps=[decision_step],
            subtasks=[subtask],
            camera_ids=camera_ids,
            time_ids=time_ids,
            spatial_ids=spatial_ids,
        )


class ChunkFeatureWriter:
    """Append fixed-shape selector samples to a portable HDF5 file."""

    def __init__(
        self,
        path: str | Path,
        *,
        feature_shape: tuple[int, int],
        candidate_chunks: Sequence[int],
        overwrite: bool = False,
        metadata: dict[str, Any] | None = None,
        feature_dtype: str = "float16",
    ):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "x"
        self.h5 = h5py.File(self.path, mode)
        token_count, feature_dim = (int(value) for value in feature_shape)
        candidates = tuple(int(value) for value in candidate_chunks)
        if token_count <= 0 or feature_dim <= 0:
            raise ValueError("feature_shape must contain positive token and feature dimensions")
        if not candidates or tuple(sorted(set(candidates))) != candidates:
            raise ValueError("candidate_chunks must be unique and strictly increasing")
        if feature_dtype not in ("float16", "float32"):
            raise ValueError("feature_dtype must be float16 or float32")

        self.feature_shape = (token_count, feature_dim)
        self.candidate_chunks = candidates
        self.h5.attrs["schema_version"] = SCHEMA_VERSION
        self.h5.attrs["candidate_chunks"] = json.dumps(list(candidates))
        self.h5.attrs["feature_shape"] = json.dumps(list(self.feature_shape))
        self.h5.attrs["metadata"] = json.dumps(metadata or {}, sort_keys=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self.h5.create_dataset(
            "features",
            shape=(0, token_count, feature_dim),
            maxshape=(None, token_count, feature_dim),
            chunks=(1, token_count, feature_dim),
            dtype=feature_dtype,
        )
        for name in ("camera_ids", "time_ids", "spatial_ids"):
            self.h5.create_dataset(
                name,
                shape=(0, token_count),
                maxshape=(None, token_count),
                chunks=(64, token_count),
                dtype="i4",
                fillvalue=-1,
            )
        self.h5.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype="i8",
        )
        self.h5.create_dataset(
            "utilities",
            shape=(0, len(candidates)),
            maxshape=(None, len(candidates)),
            chunks=(64, len(candidates)),
            dtype="f4",
            fillvalue=np.nan,
        )
        self.h5.create_dataset(
            "target_probabilities",
            shape=(0, len(candidates)),
            maxshape=(None, len(candidates)),
            chunks=(64, len(candidates)),
            dtype="f4",
            fillvalue=np.nan,
        )
        self.h5.create_dataset(
            "sample_weights",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype="f4",
            fillvalue=1.0,
        )
        self.h5.create_dataset(
            "episode_ids",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=string_dtype,
        )
        self.h5.create_dataset(
            "subtasks",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype=string_dtype,
        )
        self.h5.create_dataset(
            "decision_steps",
            shape=(0,),
            maxshape=(None,),
            chunks=(1024,),
            dtype="i8",
        )

    def __enter__(self) -> "ChunkFeatureWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self.h5:
            self.h5.flush()
            self.h5.close()

    def _ids_array(
        self,
        ids: torch.Tensor | np.ndarray | None,
        *,
        batch_size: int,
    ) -> np.ndarray:
        token_count = self.feature_shape[0]
        if ids is None:
            return np.full((batch_size, token_count), -1, dtype=np.int32)
        result = np.asarray(
            ids.detach().cpu().numpy() if isinstance(ids, torch.Tensor) else ids,
            dtype=np.int32,
        )
        if result.ndim == 1:
            result = np.broadcast_to(result[None], (batch_size, token_count))
        if result.shape != (batch_size, token_count):
            raise ValueError(
                f"Token ids must have shape {(batch_size, token_count)}, got {result.shape}"
            )
        return result

    def append(
        self,
        features: torch.Tensor | np.ndarray,
        labels: Sequence[int] | torch.Tensor | np.ndarray,
        *,
        episode_ids: Sequence[str],
        decision_steps: Sequence[int],
        subtasks: Sequence[str] | None = None,
        utilities: torch.Tensor | np.ndarray | None = None,
        target_probabilities: torch.Tensor | np.ndarray | None = None,
        sample_weights: Sequence[float] | torch.Tensor | np.ndarray | None = None,
        camera_ids: torch.Tensor | np.ndarray | None = None,
        time_ids: torch.Tensor | np.ndarray | None = None,
        spatial_ids: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[int, int]:
        feature_array = np.asarray(
            features.detach().cpu().numpy()
            if isinstance(features, torch.Tensor)
            else features
        )
        if feature_array.ndim == 2:
            feature_array = feature_array[None]
        expected_tail = self.feature_shape
        if tuple(feature_array.shape[1:]) != expected_tail:
            raise ValueError(
                f"Expected features [B, {expected_tail[0]}, {expected_tail[1]}], "
                f"got {feature_array.shape}"
            )
        batch_size = feature_array.shape[0]
        label_array = np.asarray(
            labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels,
            dtype=np.int64,
        ).reshape(-1)
        if label_array.shape != (batch_size,):
            raise ValueError(f"Expected {batch_size} labels, got {label_array.shape}")
        if label_array.size and (
            label_array.min() < 0 or label_array.max() >= len(self.candidate_chunks)
        ):
            raise ValueError("labels contain a class outside candidate_chunks")
        if len(episode_ids) != batch_size or len(decision_steps) != batch_size:
            raise ValueError("episode_ids and decision_steps must match feature batch size")
        subtasks = list(subtasks or [""] * batch_size)
        if len(subtasks) != batch_size:
            raise ValueError("subtasks must match feature batch size")
        if sample_weights is None:
            sample_weight_array = np.ones(batch_size, dtype=np.float32)
        else:
            sample_weight_array = np.asarray(
                sample_weights.detach().cpu().numpy()
                if isinstance(sample_weights, torch.Tensor)
                else sample_weights,
                dtype=np.float32,
            ).reshape(-1)
            if sample_weight_array.shape != (batch_size,):
                raise ValueError(
                    f"Expected {batch_size} sample weights, got "
                    f"{sample_weight_array.shape}"
                )
            if (
                not np.isfinite(sample_weight_array).all()
                or np.any(sample_weight_array <= 0.0)
            ):
                raise ValueError("sample_weights must be finite and strictly positive")

        if utilities is None:
            utility_array = np.full(
                (batch_size, len(self.candidate_chunks)),
                np.nan,
                dtype=np.float32,
            )
        else:
            utility_array = np.asarray(
                utilities.detach().cpu().numpy()
                if isinstance(utilities, torch.Tensor)
                else utilities,
                dtype=np.float32,
            )
            if utility_array.ndim == 1:
                utility_array = utility_array[None]
            if utility_array.shape != (batch_size, len(self.candidate_chunks)):
                raise ValueError(
                    "utilities must have shape "
                    f"{(batch_size, len(self.candidate_chunks))}, got {utility_array.shape}"
                )

        if target_probabilities is None:
            target_probability_array = np.full(
                (batch_size, len(self.candidate_chunks)),
                np.nan,
                dtype=np.float32,
            )
        else:
            target_probability_array = np.asarray(
                target_probabilities.detach().cpu().numpy()
                if isinstance(target_probabilities, torch.Tensor)
                else target_probabilities,
                dtype=np.float32,
            )
            if target_probability_array.ndim == 1:
                target_probability_array = target_probability_array[None]
            expected_shape = (batch_size, len(self.candidate_chunks))
            if target_probability_array.shape != expected_shape:
                raise ValueError(
                    f"target_probabilities must have shape {expected_shape}, "
                    f"got {target_probability_array.shape}"
                )
            if (
                not np.isfinite(target_probability_array).all()
                or np.any(target_probability_array < 0.0)
                or not np.allclose(target_probability_array.sum(axis=1), 1.0, atol=1e-5)
            ):
                raise ValueError(
                    "target_probabilities must be finite, non-negative, and sum to one"
                )

        old_size = len(self.h5["labels"])
        new_size = old_size + batch_size
        for dataset in self.h5.values():
            dataset.resize(new_size, axis=0)
        self.h5["features"][old_size:new_size] = feature_array
        self.h5["labels"][old_size:new_size] = label_array
        self.h5["utilities"][old_size:new_size] = utility_array
        self.h5["target_probabilities"][old_size:new_size] = target_probability_array
        self.h5["sample_weights"][old_size:new_size] = sample_weight_array
        self.h5["camera_ids"][old_size:new_size] = self._ids_array(
            camera_ids, batch_size=batch_size
        )
        self.h5["time_ids"][old_size:new_size] = self._ids_array(
            time_ids, batch_size=batch_size
        )
        self.h5["spatial_ids"][old_size:new_size] = self._ids_array(
            spatial_ids, batch_size=batch_size
        )
        self.h5["episode_ids"][old_size:new_size] = list(episode_ids)
        self.h5["decision_steps"][old_size:new_size] = np.asarray(
            decision_steps, dtype=np.int64
        )
        self.h5["subtasks"][old_size:new_size] = subtasks
        self.h5.flush()
        return old_size, new_size


class ChunkFeatureDataset(Dataset):
    """Lazy HDF5 dataset safe to reopen inside DataLoader workers."""

    def __init__(self, path: str | Path, indices: Sequence[int] | None = None):
        self.path = Path(path).expanduser().resolve()
        with h5py.File(self.path, "r") as h5:
            schema_version = int(h5.attrs.get("schema_version", -1))
            if schema_version != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported chunk dataset schema {schema_version}; expected {SCHEMA_VERSION}"
                )
            self.candidate_chunks = tuple(
                int(value) for value in json.loads(h5.attrs["candidate_chunks"])
            )
            self.feature_shape = tuple(
                int(value) for value in json.loads(h5.attrs["feature_shape"])
            )
            self.metadata = json.loads(h5.attrs.get("metadata", "{}"))
            total = len(h5["labels"])
        self.indices = (
            np.arange(total, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if self.indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if self.indices.size and (
            self.indices.min() < 0 or self.indices.max() >= total
        ):
            raise IndexError("dataset indices are outside the HDF5 sample range")
        self._h5: h5py.File | None = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        utilities = torch.from_numpy(np.asarray(self.h5["utilities"][index])).float()
        if "target_probabilities" in self.h5:
            target_probabilities = torch.from_numpy(
                np.asarray(self.h5["target_probabilities"][index])
            ).float()
        else:
            target_probabilities = torch.full(
                (len(self.candidate_chunks),), float("nan"), dtype=torch.float32
            )
        sample_weight = (
            float(self.h5["sample_weights"][index])
            if "sample_weights" in self.h5
            else 1.0
        )
        return {
            "features": torch.from_numpy(np.asarray(self.h5["features"][index])).float(),
            "label": torch.as_tensor(int(self.h5["labels"][index]), dtype=torch.long),
            "utilities": utilities,
            "has_utilities": torch.isfinite(utilities).all(),
            "target_probabilities": target_probabilities,
            "has_target_probabilities": torch.isfinite(target_probabilities).all(),
            "sample_weight": torch.as_tensor(sample_weight, dtype=torch.float32),
            "camera_ids": torch.from_numpy(
                np.asarray(self.h5["camera_ids"][index], dtype=np.int64)
            ),
            "time_ids": torch.from_numpy(
                np.asarray(self.h5["time_ids"][index], dtype=np.int64)
            ),
            "spatial_ids": torch.from_numpy(
                np.asarray(self.h5["spatial_ids"][index], dtype=np.int64)
            ),
            "episode_id": self.h5["episode_ids"].asstr()[index],
            "decision_step": int(self.h5["decision_steps"][index]),
            "subtask": self.h5["subtasks"].asstr()[index],
            "index": index,
        }


def episode_split_indices(
    path: str | Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole episodes, never adjacent frames, across train and validation."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must lie strictly between 0 and 1")
    with h5py.File(Path(path).expanduser().resolve(), "r") as h5:
        episode_ids = np.asarray(h5["episode_ids"].asstr()[:])
    unique_episodes = np.unique(episode_ids)
    if len(unique_episodes) < 2:
        raise ValueError("At least two distinct episodes are required for a split")
    shuffled = np.random.default_rng(seed).permutation(unique_episodes)
    val_count = min(max(1, round(len(shuffled) * val_ratio)), len(shuffled) - 1)
    val_episodes = set(shuffled[:val_count].tolist())
    val_mask = np.asarray([episode in val_episodes for episode in episode_ids])
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
