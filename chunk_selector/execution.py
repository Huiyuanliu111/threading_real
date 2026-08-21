"""Shared execution-length rules for adaptive action chunks."""
from __future__ import annotations

from typing import Literal


PredictionMode = Literal["full_then_truncate", "required_only"]
PREDICTION_MODES: tuple[PredictionMode, ...] = (
    "full_then_truncate",
    "required_only",
)


def _validated_execution_steps(
    *,
    prediction_horizon: int,
    default_execution_steps: int,
    requested_execution_steps: int | None,
    max_execution_steps: int,
) -> tuple[int, int]:
    prediction_horizon = int(prediction_horizon)
    default_execution_steps = int(default_execution_steps)
    max_execution_steps = int(max_execution_steps)
    if prediction_horizon <= 0 or default_execution_steps <= 0:
        raise ValueError("prediction and default execution lengths must be positive")
    if max_execution_steps <= 0 or max_execution_steps > prediction_horizon:
        raise ValueError(
            "max_execution_steps must lie in [1, prediction_horizon]"
        )
    execution_steps = (
        default_execution_steps
        if requested_execution_steps is None
        else int(requested_execution_steps)
    )
    if not 1 <= execution_steps <= max_execution_steps:
        raise ValueError(
            f"execution chunk must lie in [1, {max_execution_steps}], "
            f"got {execution_steps}"
        )
    return prediction_horizon, execution_steps


def full_plan_execution_lengths(
    *,
    prediction_horizon: int,
    default_execution_steps: int,
    requested_execution_steps: int | None,
    max_execution_steps: int,
) -> tuple[int, int]:
    """Keep prediction length fixed while selecting an executable prefix."""
    return _validated_execution_steps(
        prediction_horizon=prediction_horizon,
        default_execution_steps=default_execution_steps,
        requested_execution_steps=requested_execution_steps,
        max_execution_steps=max_execution_steps,
    )


def required_only_execution_lengths(
    *,
    prediction_horizon: int,
    default_execution_steps: int,
    requested_execution_steps: int | None,
    max_execution_steps: int,
) -> tuple[int, int]:
    """Generate exactly the actions that will be executed before replanning."""
    _, execution_steps = _validated_execution_steps(
        prediction_horizon=prediction_horizon,
        default_execution_steps=default_execution_steps,
        requested_execution_steps=requested_execution_steps,
        max_execution_steps=max_execution_steps,
    )
    return execution_steps, execution_steps


def prediction_execution_lengths(
    *,
    mode: PredictionMode,
    prediction_horizon: int,
    default_execution_steps: int,
    requested_execution_steps: int | None,
    max_execution_steps: int,
    dropped_prediction_steps: int = 0,
) -> tuple[int, int]:
    """Resolve prediction and execution lengths for an explicit inference mode.

    ``dropped_prediction_steps`` accounts for leading predictions that a policy
    generates but intentionally discards before execution.  In required-only
    mode those slots must still be generated in addition to the executable
    actions.
    """
    dropped_prediction_steps = int(dropped_prediction_steps)
    if dropped_prediction_steps < 0:
        raise ValueError("dropped_prediction_steps must be non-negative")
    if mode == "full_then_truncate":
        resolver = full_plan_execution_lengths
    elif mode == "required_only":
        resolver = required_only_execution_lengths
    else:
        raise ValueError(
            f"unknown prediction mode {mode!r}; expected one of {PREDICTION_MODES}"
        )
    prediction_steps, execution_steps = resolver(
        prediction_horizon=prediction_horizon,
        default_execution_steps=default_execution_steps,
        requested_execution_steps=requested_execution_steps,
        max_execution_steps=max_execution_steps,
    )
    if mode == "required_only":
        prediction_steps += dropped_prediction_steps
        if prediction_steps > int(prediction_horizon):
            raise ValueError(
                "required prediction length including dropped steps exceeds "
                f"prediction horizon: {prediction_steps} > {prediction_horizon}"
            )
    return prediction_steps, execution_steps


def execution_steps_from_chunk_label(
    chunk_label: int,
    *,
    max_execution_steps: int,
    full_chunk_label: int,
) -> int:
    """Map a public chunk label to the executable action-prefix length.

    PushBox predicts ``full_chunk_label`` actions but drops its stale alignment
    prediction at index zero.  The public full-chunk label therefore remains 20
    for reports and plots while the corresponding executable prefix is 19.
    Other labels retain their literal execution length.
    """
    chunk_label = int(chunk_label)
    max_execution_steps = int(max_execution_steps)
    full_chunk_label = int(full_chunk_label)
    if max_execution_steps <= 0:
        raise ValueError("max_execution_steps must be positive")
    if full_chunk_label < max_execution_steps:
        raise ValueError("full_chunk_label must be >= max_execution_steps")
    if not 1 <= chunk_label <= full_chunk_label:
        raise ValueError(
            f"chunk label must lie in [1, {full_chunk_label}], got {chunk_label}"
        )
    if chunk_label == full_chunk_label:
        return max_execution_steps
    if chunk_label > max_execution_steps:
        raise ValueError(
            f"only the full chunk label {full_chunk_label} may alias execution "
            f"beyond {max_execution_steps}; got {chunk_label}"
        )
    return chunk_label
