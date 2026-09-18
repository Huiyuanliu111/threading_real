"""Batched pi0.5 AAC deployment using the existing policy and processors."""
from __future__ import annotations

import torch

from .inference import threading_delta_to_aac


@torch.inference_mode()
def predict_pi05_aac(deployment, frame, *, closed_width_threshold: float):
    """Sample full chunks once and retain the existing fixed-gripper behavior."""
    batch = deployment.preprocess(frame)
    normalized = None

    def sample_chunks(n):
        nonlocal normalized
        repeated = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0 or value.shape[0] != 1:
                    raise ValueError(f"pi0.5 AAC expects a single observation: {key}")
                repeated[key] = value.repeat_interleave(n, dim=0)
            elif isinstance(value, list) and len(value) == 1:
                repeated[key] = value * n
            else:
                repeated[key] = value
        normalized = deployment.model.predict_action_chunk(repeated)
        if normalized.shape[:2] != (n, deployment.horizon):
            raise ValueError("pi0.5 AAC requires full-horizon candidate predictions")
        actions = deployment.postprocess(normalized.clone()).clone()
        actions[..., 6] = 0.0  # Match PI05DeploymentPolicy execution semantics.
        return actions

    prediction = deployment.aac.predict(
        sample_chunks,
        to_aac=lambda actions: threading_delta_to_aac(
            actions,
            current_width=float(frame["observation.state"][-1].item()),
            closed_width_threshold=closed_width_threshold,
        ),
        to_entropy=lambda _: normalized,
    )
    deployment.last_prediction_diagnostics = {
        **prediction.metrics(),
        "prediction_mode": "full_then_truncate",
        "execution_chunk": prediction.decision.h_star,
        "continuous_chunk": float(prediction.decision.h_star),
        "predicted_chunk": prediction.predicted_horizon,
        "total_seconds": prediction.predict_action_elapsed_sec,
    }
    return {"action": prediction.actions.unsqueeze(0)}
