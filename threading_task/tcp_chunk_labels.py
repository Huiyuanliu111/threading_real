"""TCP-motion pseudo-labels for real-robot adaptive chunk selection.

The labels intentionally use only signals that can also be observed online:
current robot state and a short state history.  Future trajectory data is used
offline solely to create supervision labels.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import rankdata


LABEL_RULE_VERSION = "tcp_motion_precision_v1"


def smooth_chunk_labels(
    chunks: np.ndarray,
    *,
    candidate_chunks: tuple[int, ...],
    window: int,
) -> np.ndarray:
    """Median-filter one episode's ordered labels without creating new values."""
    values = np.asarray(chunks, dtype=np.int64)
    candidates = np.asarray(sorted(set(int(v) for v in candidate_chunks)), dtype=np.int64)
    if window <= 0 or window % 2 == 0:
        raise ValueError("label smoothing window must be positive and odd")
    ranks = np.searchsorted(candidates, values)
    if not np.array_equal(candidates[ranks], values):
        raise ValueError("chunks must all be members of candidate_chunks")
    if window == 1 or len(values) < 2:
        return values.copy()
    return candidates[median_filter(ranks, size=window, mode="nearest")]


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected a vector, got {values.shape}")
    if window <= 0 or window % 2 == 0:
        raise ValueError("smoothing window must be positive and odd")
    if len(values) == 0 or window == 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.full(window, 1.0 / window), mode="valid")


def tcp_motion_metrics(
    tcp_positions: np.ndarray,
    cartesian_actions: np.ndarray,
    *,
    fps: float,
    smoothing_window: int = 5,
    direction_speed_floor_mps: float = 0.005,
) -> dict[str, np.ndarray]:
    """Return per-frame TCP motion quantities from Cartesian delta actions.

    The action convention is ``[dx, dy, dz, drotvec_x, drotvec_y,
    drotvec_z, dgripper]`` in the base/world frame.  Curvature is the change
    in direction of successive translational increments per second.
    """
    tcp_positions = np.asarray(tcp_positions, dtype=np.float64)
    actions = np.asarray(cartesian_actions, dtype=np.float64)
    if tcp_positions.ndim != 2 or tcp_positions.shape[1] != 3:
        raise ValueError(f"Expected TCP positions [T,3], got {tcp_positions.shape}")
    if actions.shape != (len(tcp_positions), 7):
        raise ValueError(f"Expected actions {(len(tcp_positions), 7)}, got {actions.shape}")
    if fps <= 0:
        raise ValueError("fps must be positive")

    translation_step = np.linalg.norm(actions[:, :3], axis=1)
    linear_speed = translation_step * fps
    angular_speed = np.linalg.norm(actions[:, 3:6], axis=1) * fps
    gripper_speed = np.abs(actions[:, 6]) * fps

    if direction_speed_floor_mps < 0:
        raise ValueError("direction_speed_floor_mps must be non-negative")
    directions = np.zeros_like(actions[:, :3])
    # Direction is meaningless for near-stationary sub-millimetre motions;
    # excluding them prevents encoder / recorder noise from looking like a
    # high-curvature precision maneuver.
    moving = linear_speed > direction_speed_floor_mps
    directions[moving] = actions[moving, :3] / translation_step[moving, None]
    curvature = np.zeros(len(actions), dtype=np.float64)
    if len(actions) > 1:
        consecutive = moving[1:] & moving[:-1]
        cosine = np.einsum("ij,ij->i", directions[1:], directions[:-1])
        angles = np.arccos(np.clip(cosine, -1.0, 1.0))
        curvature[1:] = np.where(consecutive, angles * fps, 0.0)
    acceleration = np.abs(np.diff(linear_speed, prepend=linear_speed[0])) * fps

    return {
        "tcp_linear_speed_mps": _moving_average(linear_speed, smoothing_window),
        "tcp_angular_speed_radps": _moving_average(angular_speed, smoothing_window),
        "tcp_curvature_radps": _moving_average(curvature, smoothing_window),
        "tcp_acceleration_mps2": _moving_average(acceleration, smoothing_window),
        "gripper_speed_mps": _moving_average(gripper_speed, smoothing_window),
    }


def empirical_percentile(values: np.ndarray) -> np.ndarray:
    """Tie-aware empirical percentile in [0,1], safe for constant vectors."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Expected a non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError("Metric values must be finite")
    return rankdata(values, method="average") / len(values)


def label_tcp_motion(
    metrics: dict[str, np.ndarray],
    *,
    candidate_chunks: tuple[int, ...],
    slow_linear_weight: float = 0.70,
    slow_angular_weight: float = 0.10,
    complexity_weight: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Map TCP-motion metrics to chunks, where high precision means short.

    Slow translational/rotational motion increases precision. Curvature,
    acceleration, and gripper motion add a complexity term. The final score is
    converted by global empirical quantiles, giving each permitted chunk a
    useful amount of supervision instead of collapsing to a single class.
    """
    candidates = tuple(sorted(set(int(value) for value in candidate_chunks)))
    if len(candidates) < 2 or candidates[0] <= 0:
        raise ValueError("candidate_chunks must contain at least two positive values")
    weights = slow_linear_weight + slow_angular_weight + complexity_weight
    if not np.isclose(weights, 1.0) or min(
        slow_linear_weight, slow_angular_weight, complexity_weight
    ) < 0:
        raise ValueError("label weights must be non-negative and sum to one")

    required = {
        "tcp_linear_speed_mps",
        "tcp_angular_speed_radps",
        "tcp_curvature_radps",
        "tcp_acceleration_mps2",
        "gripper_speed_mps",
    }
    if set(metrics) != required:
        raise KeyError(f"Expected exactly metric keys {sorted(required)}, got {sorted(metrics)}")
    size = len(metrics["tcp_linear_speed_mps"])
    if any(np.asarray(value).shape != (size,) for value in metrics.values()):
        raise ValueError("All metric vectors must have the same length")

    slow_linear = 1.0 - empirical_percentile(metrics["tcp_linear_speed_mps"])
    slow_angular = 1.0 - empirical_percentile(metrics["tcp_angular_speed_radps"])
    complexity = np.maximum.reduce(
        (
            empirical_percentile(metrics["tcp_curvature_radps"]),
            empirical_percentile(metrics["tcp_acceleration_mps2"]),
            empirical_percentile(metrics["gripper_speed_mps"]),
        )
    )
    score = (
        slow_linear_weight * slow_linear
        + slow_angular_weight * slow_angular
        + complexity_weight * complexity
    )
    precision_percentile = empirical_percentile(score)
    # High precision score -> the smallest public chunk label.
    bins = np.minimum((precision_percentile * len(candidates)).astype(np.int64), len(candidates) - 1)
    chunks = np.asarray([candidates[len(candidates) - 1 - item] for item in bins], dtype=np.int64)
    return chunks, score.astype(np.float32)
