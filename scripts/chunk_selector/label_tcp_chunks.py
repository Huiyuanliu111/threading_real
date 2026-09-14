#!/usr/bin/env python3
"""Create auditable TCP-motion or endpoint-distance chunk-size pseudo-labels from Cartesian LeRobot v3.

The output is a separate Parquet label table: it does not alter the LeRobot
dataset. Use it as supervision when extracting visual features for a chunk
selector. Labels map slow/complex TCP motion or proximity to a fixed endpoint
to short action chunks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from convert_lerobot_v3_to_cartesian import UrdfForwardKinematics
from scripts.deployment.endpoint_schedule import EndpointExecutionSchedule
from threading_task.tcp_chunk_labels import (
    LABEL_RULE_VERSION,
    DISTANCE_LABEL_RULE_VERSION,
    distance_chunk_targets,
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
    label_method: str = "speed",
    endpoint_schedule: Path | None = None,
    coarse_radius_m: float | None = None,
) -> dict:
    if label_method not in {"speed", "distance"}:
        raise ValueError("label_method must be speed or distance")
    schedule = None
    if label_method == "distance":
        if endpoint_schedule is None:
            raise ValueError("distance labels require endpoint_schedule")
        schedule = EndpointExecutionSchedule.from_file(endpoint_schedule)
        if coarse_radius_m is None:
            coarse_radius_m = 2 * schedule.fine_radius_m
        # Validate configuration before reading the dataset or running FK.
        distance_chunk_targets(
            np.zeros((1, 3)), endpoint_xyz_m=schedule.endpoint_xyz_m,
            fine_radius_m=schedule.fine_radius_m, coarse_radius_m=coarse_radius_m,
            candidate_chunks=candidate_chunks,
        )
    elif endpoint_schedule is not None or coarse_radius_m is not None:
        raise ValueError("endpoint_schedule and coarse_radius_m require distance labels")
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

    distance_columns = {}
    if schedule is None:
        raw_chunks, precision_score = label_tcp_motion(
            metric_columns, candidate_chunks=candidate_chunks
        )
        target_probabilities, _ = soft_chunk_targets(
            precision_score, candidate_chunks=candidate_chunks,
        )
    else:
        target_probabilities, expected, distances = distance_chunk_targets(
            tcp_positions, endpoint_xyz_m=schedule.endpoint_xyz_m,
            fine_radius_m=schedule.fine_radius_m, coarse_radius_m=coarse_radius_m,
            candidate_chunks=candidate_chunks,
        )
        raw_chunks = np.asarray(candidate_chunks)[target_probabilities.argmax(axis=1)]
        precision_score = 1 - (expected - candidate_chunks[0]) / (
            candidate_chunks[-1] - candidate_chunks[0]
        )
        distance_columns["endpoint_distance_m"] = distances
    chunks = raw_chunks.copy()
    raw_transitions = 0
    smoothed_transitions = 0
    for episode in np.unique(episode_indices):
        rows = np.flatnonzero(episode_indices == episode)
        rows = rows[np.argsort(frame_indices[rows], kind="stable")]
        if schedule is None:
            chunks[rows] = smooth_chunk_labels(
                raw_chunks[rows], candidate_chunks=candidate_chunks,
                window=label_smoothing_window,
            )
        target_probabilities[rows] = smooth_chunk_probabilities(
            target_probabilities[rows],
            window=label_smoothing_window,
        )
        if schedule is not None:
            chunks[rows] = np.asarray(candidate_chunks)[target_probabilities[rows].argmax(axis=1)]
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
            **{name: pa.array(values) for name, values in distance_columns.items()},
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
    summary["label_method"] = label_method
    if schedule is not None:
        summary.update({
            "label_rule_version": DISTANCE_LABEL_RULE_VERSION,
            "frame": "panda_link0",
            "endpoint_schedule": str(endpoint_schedule.expanduser().resolve()),
            "endpoint_xyz_m": schedule.endpoint_xyz_m.tolist(),
            "fine_radius_m": schedule.fine_radius_m,
            "coarse_radius_m": coarse_radius_m,
            "score": "1 - clip((distance - fine_radius) / (coarse_radius - fine_radius), 0, 1)",
            "label_mapping": "argmax of smoothed distance probabilities; ties prefer shorter chunk",
            "soft_label_mapping": "linear distance-to-chunk mapping, interpolated between neighboring chunk sizes",
            "history_latch": False,
        })
        summary["label_smoothing"]["method"] = "per_episode_probability_median_filter_then_argmax"
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
    parser.add_argument("--label-method", choices=("speed", "distance"), default="speed")
    parser.add_argument("--endpoint-schedule", type=Path,
                        help="execution schedule YAML with endpoint and fine radius; required for distance")
    parser.add_argument("--coarse-radius-m", type=float,
                        help="distance where maximum chunk is reached; defaults to twice fine radius")
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
        label_method=args.label_method,
        endpoint_schedule=args.endpoint_schedule,
        coarse_radius_m=args.coarse_radius_m,
    )
    print(json.dumps({"output": str(args.output.resolve()), "chunk_counts": summary["chunk_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
