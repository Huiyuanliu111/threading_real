"""Policy-independent sampling boundary and Threading Cartesian adaptation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import torch
from torch import Tensor

from .aac import AAC_IMPLEMENTATION_VERSION, AACConfig, AACResult, _finite_float, select_chunk_size


def threading_delta_to_aac(
    actions: Tensor, *, current_width: float, closed_width_threshold: float,
) -> Tensor:
    """Convert physical [dxyz,drotvec,dwidth] to AAC's binary gripper format.

    Widths and threshold must use the same units (meters in Threading).
    The caller must apply any execution-time action overrides before conversion.
    """
    actions = _finite_float(actions, "actions")
    if actions.ndim not in (2, 3) or actions.shape[-1] != 7 or actions.shape[-2] < 1:
        raise ValueError("actions must have shape [H,7] or [N,H,7]")
    if not math.isfinite(current_width) or not math.isfinite(closed_width_threshold):
        raise ValueError("width and threshold must be finite")
    converted = actions.clone()
    widths = current_width + actions[..., 6].cumsum(dim=-1)
    converted[..., 6] = (widths < closed_width_threshold).to(actions.dtype)
    return converted


@dataclass(frozen=True)
class AACPrediction:
    actions: Tensor
    decision: AACResult
    predicted_horizon: int
    execution_candidate_index: int
    generated_action_tokens: int
    candidate_mean_variance: float
    candidate_max_variance: float
    predict_action_elapsed_sec: float

    def metrics(self) -> dict:
        return {
            **self.decision.metrics(),
            "implementation_version": AAC_IMPLEMENTATION_VERSION,
            "execution_candidate_index": self.execution_candidate_index,
            "generated_action_tokens": self.generated_action_tokens,
            "candidate_mean_variance": self.candidate_mean_variance,
            "candidate_max_variance": self.candidate_max_variance,
            "predict_action_elapsed_sec": self.predict_action_elapsed_sec,
        }


class AACInference:
    """One batched stochastic call, select and execute the designated candidate.

    sample_chunks(N) must reuse the SAME observation and return [N,H,7]
    executable actions with independent generation noise and the full horizon.
    A closure can reuse already-computed visual features; AAC adds no encoder.
    to_aac converts executable actions into physical deltas + binary gripper.
    It is identity for paper-format actions. No resampling after selection.
    """

    def __init__(self, config: AACConfig | None = None) -> None:
        self.config = config or AACConfig()

    @torch.inference_mode()
    def predict(
        self,
        sample_chunks: Callable[[int], Tensor],
        *,
        to_aac: Callable[[Tensor], Tensor] | None = None,
        to_entropy: Callable[[Tensor], Tensor] | None = None,
    ) -> AACPrediction:
        started = time.perf_counter()
        actions = sample_chunks(self.config.num_samples)
        _finite_float(actions, "sampled actions")
        if actions.ndim != 3 or actions.shape[0] != self.config.num_samples or actions.shape[1] < 1 or actions.shape[2] != 7:
            raise ValueError("sampler must return full chunks shaped [num_samples,H>=1,7]")
        # Deterministic policy repeats are not independent uncertainty samples.
        if torch.equal(actions, actions[:1].expand_as(actions)):
            raise ValueError("identical candidate chunks: enable stochastic policy sampling")
        candidates = actions if to_aac is None else to_aac(actions.clone())
        if candidates.shape != actions.shape:
            raise ValueError("to_aac must preserve the candidate shape")
        entropy_candidates = None if to_entropy is None else to_entropy(actions.clone())
        decision = select_chunk_size(candidates, entropy_candidates=entropy_candidates, config=self.config)
        source = candidates if entropy_candidates is None else entropy_candidates
        variance = source.float().var(dim=0, unbiased=True)
        mean_variance = float(variance.mean().cpu())
        max_variance = float(variance.max().cpu())
        selected = actions[self.config.execution_candidate_index, :decision.h_star].clone()
        # Include completion of the executable prefix in wall-clock timing.
        if actions.is_cuda:
            torch.cuda.synchronize(actions.device)
        return AACPrediction(
            actions=selected,
            decision=decision,
            predicted_horizon=actions.shape[1],
            execution_candidate_index=self.config.execution_candidate_index,
            generated_action_tokens=actions.shape[0] * actions.shape[1],
            candidate_mean_variance=mean_variance,
            candidate_max_variance=max_variance,
            predict_action_elapsed_sec=time.perf_counter() - started,
        )
