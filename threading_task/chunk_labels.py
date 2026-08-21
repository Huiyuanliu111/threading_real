"""Model-independent spatial chunk labels for Threading demonstrations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


NEEDLE_HANDLE_OFFSET = np.array([0.0, 0.06, 0.0], dtype=np.float64)
NEEDLE_CENTER_OFFSET = np.array([0.0, -0.02, 0.0], dtype=np.float64)
RING_CENTER_OFFSET = np.array([0.0, 0.0, 0.088], dtype=np.float64)
LABEL_RULE_VERSION = "threading_spatial_events_v2"
REGIONS = (
    "free_approach",
    "pick_precision",
    "free_transport",
    "insert_precision",
)


@dataclass(frozen=True)
class SpatialChunkDecision:
    """One online decision made by the deterministic spatial chunk rule."""

    region: str
    chunk_size: int
    handle_distance: float
    lift_height: float
    insert_distance: float


class OnlineSpatialChunkRule:
    """Monotonic online counterpart of :func:`spatial_labels`."""

    def __init__(
        self,
        *,
        small_chunk: int,
        full_chunk: int,
        grasp_distance: float = 0.10,
        lift_threshold: float = 0.05,
        insert_approach_distance: float = 0.20,
    ) -> None:
        if not 0 < small_chunk < full_chunk:
            raise ValueError("Expected 0 < small_chunk < full_chunk")
        self.small_chunk = int(small_chunk)
        self.full_chunk = int(full_chunk)
        self.grasp_distance = float(grasp_distance)
        self.lift_threshold = float(lift_threshold)
        self.insert_approach_distance = float(insert_approach_distance)
        self._initial_needle_z: float | None = None
        self._region_index = 0

    def reset(self, obs: dict[str, Any]) -> None:
        self._initial_needle_z = float(np.asarray(obs["needle_pos"])[2])
        self._region_index = 0

    def config(self) -> dict[str, Any]:
        return {
            "label_rule_version": LABEL_RULE_VERSION,
            "small_chunk": self.small_chunk,
            "full_chunk": self.full_chunk,
            "grasp_distance": self.grasp_distance,
            "lift_threshold": self.lift_threshold,
            "insert_approach_distance": self.insert_approach_distance,
        }

    def select(self, obs: dict[str, Any]) -> SpatialChunkDecision:
        if self._initial_needle_z is None:
            raise RuntimeError("OnlineSpatialChunkRule.reset() must be called first")

        needle_position = np.asarray(obs["needle_pos"], dtype=np.float64)
        needle_quaternion = np.asarray(obs["needle_quat"], dtype=np.float64)
        tripod_position = np.asarray(obs["tripod_pos"], dtype=np.float64)
        tripod_quaternion = np.asarray(obs["tripod_quat"], dtype=np.float64)
        eef_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)

        handle_position = needle_position + _quat_rotate_xyzw(
            needle_quaternion, NEEDLE_HANDLE_OFFSET
        )
        needle_center = needle_position + _quat_rotate_xyzw(
            needle_quaternion, NEEDLE_CENTER_OFFSET
        )
        ring_center = tripod_position + _quat_rotate_xyzw(
            tripod_quaternion, RING_CENTER_OFFSET
        )
        handle_distance = float(np.linalg.norm(eef_position - handle_position))
        lift_height = float(needle_position[2] - self._initial_needle_z)
        insert_distance = float(np.linalg.norm(needle_center - ring_center))

        # Catch up across every already-crossed boundary, while never moving backward.
        if self._region_index == 0 and handle_distance <= self.grasp_distance:
            self._region_index = 1
        if self._region_index == 1 and lift_height >= self.lift_threshold:
            self._region_index = 2
        if (
            self._region_index == 2
            and insert_distance <= self.insert_approach_distance
        ):
            self._region_index = 3

        region = REGIONS[self._region_index]
        chunk_size = (
            self.full_chunk if self._region_index in (0, 2) else self.small_chunk
        )
        return SpatialChunkDecision(
            region=region,
            chunk_size=chunk_size,
            handle_distance=handle_distance,
            lift_height=lift_height,
            insert_distance=insert_distance,
        )


def _quat_rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    xyz = quaternion[..., :3]
    w = quaternion[..., 3:4]
    vectors = np.broadcast_to(np.asarray(vector, dtype=np.float64), xyz.shape)
    return vectors + 2.0 * np.cross(
        xyz,
        np.cross(xyz, vectors) + w * vectors,
    )


def spatial_metrics(
    object_obs: np.ndarray,
    eef_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return EEF-handle distance, needle lift height, and needle-ring distance."""
    object_obs = np.asarray(object_obs, dtype=np.float64)
    eef_positions = np.asarray(eef_positions, dtype=np.float64)
    if object_obs.ndim != 2 or object_obs.shape[1] < 21:
        raise ValueError(f"Expected object observations [T, >=21], got {object_obs.shape}")
    if eef_positions.shape != (len(object_obs), 3):
        raise ValueError(
            f"Expected EEF positions {(len(object_obs), 3)}, got {eef_positions.shape}"
        )

    needle_position = object_obs[:, :3]
    needle_quaternion = object_obs[:, 3:7]
    tripod_position = object_obs[:, 14:17]
    tripod_quaternion = object_obs[:, 17:21]
    handle_position = needle_position + _quat_rotate_xyzw(
        needle_quaternion, NEEDLE_HANDLE_OFFSET
    )
    needle_center = needle_position + _quat_rotate_xyzw(
        needle_quaternion, NEEDLE_CENTER_OFFSET
    )
    ring_center = tripod_position + _quat_rotate_xyzw(
        tripod_quaternion, RING_CENTER_OFFSET
    )
    return (
        np.linalg.norm(eef_positions - handle_position, axis=1),
        needle_position[:, 2] - needle_position[0, 2],
        np.linalg.norm(needle_center - ring_center, axis=1),
    )


def _first_true(mask: np.ndarray, start: int, name: str, demo: str) -> int:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool)[start:])
    if not len(indices):
        raise ValueError(f"{demo}: spatial event {name!r} was not detected")
    return start + int(indices[0])


def spatial_labels(
    handle_distance: np.ndarray,
    lift_height: np.ndarray,
    insert_distance: np.ndarray,
    *,
    grasp_distance: float,
    lift_threshold: float,
    insert_approach_distance: float,
    region_chunks: dict[str, int],
    demo: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Split a trajectory at the first ordered crossing of three spatial events."""
    missing = sorted(set(REGIONS) - set(region_chunks))
    if missing:
        raise ValueError(f"Missing region chunk labels: {missing}")
    grasp_entry = _first_true(handle_distance <= grasp_distance, 0, "grasp_entry", demo)
    lift_entry = _first_true(
        lift_height >= lift_threshold, grasp_entry + 1, "lift_entry", demo
    )
    insert_entry = _first_true(
        insert_distance <= insert_approach_distance,
        lift_entry + 1,
        "insert_entry",
        demo,
    )
    if not grasp_entry < lift_entry < insert_entry:
        raise ValueError(
            f"{demo}: invalid event order {grasp_entry}, {lift_entry}, {insert_entry}"
        )

    regions = np.empty(len(handle_distance), dtype=object)
    regions[:grasp_entry] = "free_approach"
    regions[grasp_entry:lift_entry] = "pick_precision"
    regions[lift_entry:insert_entry] = "free_transport"
    regions[insert_entry:] = "insert_precision"
    chunks = np.asarray([region_chunks[str(region)] for region in regions], dtype=np.int64)
    return regions, chunks, {
        "grasp_entry": grasp_entry,
        "lift_entry": lift_entry,
        "insert_entry": insert_entry,
    }


def distance_to_next_event(rows: np.ndarray, events: dict[str, int]) -> np.ndarray:
    event_steps = np.asarray(
        [events["grasp_entry"], events["lift_entry"], events["insert_entry"]],
        dtype=np.int64,
    )
    result = np.full(len(rows), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        future = event_steps[event_steps > row]
        if len(future):
            result[index] = int(future[0] - row)
    return result
