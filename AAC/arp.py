"""Shared PlanARP AAC adapter for Maze XY and Threading Cartesian deltas."""
from __future__ import annotations

import torch

from .inference import AACInference, threading_delta_to_aac


@torch.inference_mode()
def predict_arp_aac(
    policy, obs, aac: AACInference, *, planar: bool = False,
    current_width: float = 0.0, closed_width_threshold: float = 0.035,
    sample_batch_size: int | None = None,
):
    """Encode once; batch stochastic full plans; return the designated plan.

    Continuous entropy is in physical delta units. Spatial ARP outputs pixels,
    not normalized Cartesian actions, so no fictitious action normalizer is used.
    Maze XY deltas are embedded in XYZ with fixed zero Z/rotation/gripper.
    The constant entropy terms do not affect prefix-entropy differences.
    """
    if sample_batch_size is not None and (
        isinstance(sample_batch_size, bool) or not isinstance(sample_batch_size, int)
        or sample_batch_size < 1
    ):
        raise ValueError("sample_batch_size must be a positive integer")
    policy.eval()
    full_actions = None

    def repeat(value, n):
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != 1:
            raise ValueError("ARP AAC requires tensors with observation batch size one")
        return value.repeat_interleave(n, dim=0)

    def sample_chunks(n):
        nonlocal full_actions
        visual = policy._visual(obs)
        batch_size = min(sample_batch_size or n, n)
        chunks = []
        previous = [(module, module.low_var_eval) for module in policy.modules()
                    if hasattr(module, "low_var_eval")]
        try:
            for module, _ in previous:
                module.low_var_eval = False
            for start in range(0, n, batch_size):
                count = min(batch_size, n - start)
                repeated_visual = tuple(repeat(value, count) for value in visual)
                repeated_obs = {key: repeat(value, count) for key, value in obs.items()}
                prediction = policy.predict_action(
                    repeated_obs, visual_features=repeated_visual,
                    prediction_mode="full_then_truncate", requested_steps=policy.horizon,
                    sample=True,
                )
                chunks.append(prediction["action_pred"].clone())
                del prediction, repeated_obs, repeated_visual
        finally:
            for module, value in previous:
                module.low_var_eval = value
        full_actions = torch.cat(chunks, dim=0)
        expected_dim = 2 if planar else 7
        if full_actions.shape != (n, policy.horizon, expected_dim):
            raise ValueError("ARP AAC requires full-horizon candidate action predictions")
        if not planar:
            return full_actions
        padded = full_actions.new_zeros(n, policy.horizon, 7)
        padded[..., :2] = full_actions
        return padded

    result = aac.predict(
        sample_chunks,
        to_aac=None if planar else lambda actions: threading_delta_to_aac(
            actions, current_width=current_width,
            closed_width_threshold=closed_width_threshold,
        ),
    )
    index = aac.config.execution_candidate_index
    selected = full_actions[index:index + 1].clone()
    metrics = {
        **result.metrics(),
        "alpha": aac.config.alpha,
        "num_samples": aac.config.num_samples,
        "sample_batch_size": min(sample_batch_size or aac.config.num_samples, aac.config.num_samples),
        "entropy_action_units": "physical_cartesian_delta",
        "prediction_mode": "full_then_truncate",
        "requested_steps": policy.horizon,
        "generated_steps": policy.horizon,
        "execution_chunk": result.decision.h_star,
        "continuous_chunk": float(result.decision.h_star),
        "predicted_chunk": policy.horizon,
        "total_seconds": result.predict_action_elapsed_sec,
    }
    return {"action_pred": selected, "action": selected[:, :result.decision.h_star],
            "prediction_diagnostics": metrics}, result
