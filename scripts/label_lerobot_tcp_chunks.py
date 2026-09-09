#!/usr/bin/env python3
"""Create auditable TCP-motion chunk-size pseudo-labels from Cartesian LeRobot v3.

The output is a separate Parquet label table: it does not alter the LeRobot
dataset. Use it as supervision when extracting visual features for a chunk
selector. Labels map slow/complex TCP motion to short action chunks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convert_lerobot_v3_to_cartesian import UrdfForwardKinematics
from threading_task.tcp_chunk_labels import (
    LABEL_RULE_VERSION,
    label_tcp_motion,
    smooth_chunk_labels,
    smooth_chunk_probabilities,
    soft_chunk_targets,
    tcp_motion_metrics,
)


def _read_vectors(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table.column(name).to_pylist(), dtype=np.float32)


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def create_labels(
    dataset: Path,
    output: Path,
    *,
    urdf: Path,
    candidate_chunks: tuple[int, ...],
    smoothing_window: int,
    label_smoothing_window: int,
    direction_speed_floor_mps: float,
) -> dict:
    dataset = dataset.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    info = json.loads((dataset / "meta" / "info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError("dataset must be LeRobot v3")
    if info["features"]["observation.state"]["shape"] != [8]:
        raise ValueError("observation.state must contain q1..q7 and gripper width")
    if info["features"]["action"]["shape"] != [7]:
        raise ValueError("dataset action must be 7D Cartesian delta")
    fps = float(info["fps"])
    if smoothing_window <= 0 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be positive and odd")
    if label_smoothing_window <= 0 or label_smoothing_window % 2 == 0:
        raise ValueError("label_smoothing_window must be positive and odd")

    tables = [pq.read_table(path) for path in sorted((dataset / "data").rglob("*.parquet"))]
    table = pa.concat_tables(tables)
    states = _read_vectors(table, "observation.state")
    actions = _read_vectors(table, "action")
    episode_indices = np.asarray(table.column("episode_index"), dtype=np.int64)
    frame_indices = np.asarray(table.column("frame_index"), dtype=np.int64)
    global_indices = np.asarray(table.column("index"), dtype=np.int64)
    fk = UrdfForwardKinematics(urdf.resolve())

    tcp_positions = np.empty((len(states), 3), dtype=np.float64)
    metric_columns = {
        name: np.empty(len(states), dtype=np.float32)
        for name in (
            "tcp_linear_speed_mps",
            "tcp_angular_speed_radps",
            "tcp_curvature_radps",
            "tcp_acceleration_mps2",
            "gripper_speed_mps",
        )
    }
    for episode in np.unique(episode_indices):
        rows = np.flatnonzero(episode_indices == episode)
        rows = rows[np.argsort(frame_indices[rows], kind="stable")]
        positions, _ = fk.poses(states[rows, :7])
        tcp_positions[rows] = positions
        episode_metrics = tcp_motion_metrics(
            positions,
            actions[rows],
            fps=fps,
            smoothing_window=smoothing_window,
            direction_speed_floor_mps=direction_speed_floor_mps,
        )
        for name, values in episode_metrics.items():
            metric_columns[name][rows] = values

    raw_chunks, precision_score = label_tcp_motion(
        metric_columns, candidate_chunks=candidate_chunks
    )
    target_probabilities, _ = soft_chunk_targets(
        precision_score,
        candidate_chunks=candidate_chunks,
    )
    chunks = raw_chunks.copy()
    raw_transitions = 0
    smoothed_transitions = 0
    for episode in np.unique(episode_indices):
        rows = np.flatnonzero(episode_indices == episode)
        rows = rows[np.argsort(frame_indices[rows], kind="stable")]
        chunks[rows] = smooth_chunk_labels(
            raw_chunks[rows], candidate_chunks=candidate_chunks,
            window=label_smoothing_window,
        )
        target_probabilities[rows] = smooth_chunk_probabilities(
            target_probabilities[rows],
            window=label_smoothing_window,
        )
        raw_transitions += int(np.count_nonzero(np.diff(raw_chunks[rows])))
        smoothed_transitions += int(np.count_nonzero(np.diff(chunks[rows])))
    soft_chunk_size = target_probabilities @ np.asarray(candidate_chunks, dtype=np.float32)
    output.mkdir(parents=True)
    labels = pa.table(
        {
            "episode_index": pa.array(episode_indices),
            "frame_index": pa.array(frame_indices),
            "index": pa.array(global_indices),
            "chunk_size": pa.array(chunks),
            "precision_score": pa.array(precision_score),
            "soft_chunk_size": pa.array(soft_chunk_size),
            **{
                f"chunk_probability_{chunk}": pa.array(target_probabilities[:, class_id])
                for class_id, chunk in enumerate(candidate_chunks)
            },
            "tcp_x_m": pa.array(tcp_positions[:, 0]),
            "tcp_y_m": pa.array(tcp_positions[:, 1]),
            "tcp_z_m": pa.array(tcp_positions[:, 2]),
            **{name: pa.array(values) for name, values in metric_columns.items()},
        }
    )
    pq.write_table(labels, output / "labels.parquet", compression="zstd")

    chunk_counts = Counter(int(value) for value in chunks)
    episode_summary = []
    for episode in np.unique(episode_indices):
        rows = np.flatnonzero(episode_indices == episode)
        episode_summary.append(
            {
                "episode_index": int(episode),
                "frames": int(len(rows)),
                "chunk_counts": {str(chunk): int((chunks[rows] == chunk).sum()) for chunk in candidate_chunks},
                "linear_speed_mps": {
                    "p50": float(np.quantile(metric_columns["tcp_linear_speed_mps"][rows], 0.5)),
                    "p99": float(np.quantile(metric_columns["tcp_linear_speed_mps"][rows], 0.99)),
                },
            }
        )
    summary = {
        "label_rule_version": LABEL_RULE_VERSION,
        "source_dataset": str(dataset),
        "urdf": str(urdf.resolve()),
        "tcp_frame": "panda_hand_tcp",
        "fps": fps,
        "candidate_chunks": list(candidate_chunks),
        "smoothing_window_frames": smoothing_window,
        "label_smoothing": {
            "method": "per_episode_median_filter_on_ordered_chunk_ranks",
            "window_frames": label_smoothing_window,
            "window_seconds": label_smoothing_window / fps,
            "raw_chunk_counts": {str(chunk): int((raw_chunks == chunk).sum()) for chunk in candidate_chunks},
            "changed_frames": int((chunks != raw_chunks).sum()),
            "raw_transitions": raw_transitions,
            "smoothed_transitions": smoothed_transitions,
        },
        "direction_speed_floor_mps": direction_speed_floor_mps,
        "score": "0.70*slow_linear + 0.10*slow_angular + 0.20*max(curvature, acceleration, gripper_speed) percentiles",
        "label_mapping": "higher precision-score percentile maps to a shorter chunk",
        "soft_label_mapping": (
            "linear interpolation between neighboring candidate chunks by global "
            "precision-score rank"
        ),
        "soft_chunk_size": {
            "min": float(soft_chunk_size.min()),
            "p25": float(np.quantile(soft_chunk_size, 0.25)),
            "median": float(np.median(soft_chunk_size)),
            "p75": float(np.quantile(soft_chunk_size, 0.75)),
            "max": float(soft_chunk_size.max()),
        },
        "total_frames": int(len(labels)),
        "chunk_counts": {str(chunk): chunk_counts.get(chunk, 0) for chunk in candidate_chunks},
        "episodes": episode_summary,
    }
    (output / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/threading_lerobot_v3_cartesian"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/threading_lerobot_v3_cartesian_tcp_chunk_labels"),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("remote_controller/src/remote_controller/assets/panda/panda_arm.urdf"),
    )
    parser.add_argument("--candidate-chunks", type=int, nargs="+", default=[1, 2, 4, 8, 20])
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--label-smoothing-window", type=int, default=1,
                        help="odd per-episode median-filter width after label assignment; 1 disables it")
    parser.add_argument(
        "--direction-speed-floor-mps",
        type=float,
        default=0.005,
        help="ignore curvature below this TCP linear speed",
    )
    args = parser.parse_args()
    candidates = tuple(sorted(set(args.candidate_chunks)))
    if len(candidates) < 2 or candidates[0] <= 0:
        parser.error("--candidate-chunks must have at least two positive values")
    summary = create_labels(
        args.dataset,
        args.output,
        urdf=args.urdf,
        candidate_chunks=candidates,
        smoothing_window=args.smoothing_window,
        label_smoothing_window=args.label_smoothing_window,
        direction_speed_floor_mps=args.direction_speed_floor_mps,
    )
    print(json.dumps({"output": str(args.output.resolve()), "chunk_counts": summary["chunk_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
