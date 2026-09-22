"""Temporary attention hooks for the real Threading vector-action ARP policy."""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor

from .core import AutoHorizonConfig, AutoHorizonResult, select_horizon


@contextmanager
def capture_action_attention(policy):
    """Capture post-softmax, pre-dropout self-attention without changing weights.

    Each layer retains its latest generation pass. Calls must not overlap on
    the same policy instance. Multi-chunk action generation is rejected below:
    earlier rows in its final pass would describe re-encoded sampled actions.
    """
    maps = {}
    handles = []
    try:
        for index, block in enumerate(policy.policy.blocks):
            def capture(module, args, index=index):
                maps[index] = args[0].detach()
            handles.append(block.attn.attn_dropout.register_forward_pre_hook(capture))
        yield maps
    finally:
        for handle in handles:
            handle.remove()


def _prediction_offset(policy) -> int:
    # Lazy imports: the standalone tensor algorithm needs only torch.
    from threading_task.policy import ThreadingARPolicy

    if isinstance(policy, ThreadingARPolicy):
        return 0
    raise TypeError("Unknown policy: supply prediction_offset explicitly")


@torch.no_grad()
def predict_action_with_autohorizon(
    policy: torch.nn.Module,
    observation: dict[str, Tensor],
    config: AutoHorizonConfig | None = None,
    *,
    prediction_offset: int | None = None,
) -> AutoHorizonResult:
    """Generate once, estimate on all predicted tokens, return [executed_steps,D].

    Sampling and low_var_eval remain as configured in the checkpoint. The
    attached learned selector is temporarily disabled. State and hooks are
    restored on success or failure. Supports one environment per call.
    """
    if getattr(policy, "uses_mvt", False):
        if prediction_offset not in (None, 0):
            raise ValueError("MVT actions start at prediction_offset=0")
        from .mvt import predict_mvt_with_autohorizon
        return predict_mvt_with_autohorizon(policy, observation, config)[1]
    if getattr(policy, "action_tokens", 1) != 1:
        raise ValueError("AutoHorizon requires one token per action for non-MVT policies")
    if policy.training or any(m.training for m in policy.policy.blocks):
        raise ValueError("AutoHorizon requires policy.eval()")
    if not observation or any(v.shape[0] != 1 for v in observation.values() if isinstance(v, Tensor)):
        raise ValueError("AutoHorizon supports observation batch size 1")
    offset = _prediction_offset(policy) if prediction_offset is None else prediction_offset
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("prediction_offset must be a non-negative integer")
    # Preserve the reference single-generation-group restriction.
    if policy.action_chunk_size < policy.horizon:
        raise ValueError("AutoHorizon currently requires action_chunk_size >= horizon")
    config = config or AutoHorizonConfig()
    old_chunk = policy.inference_chunk_size
    old_mode = policy.prediction_mode
    old_selector = policy.chunk_selector
    try:
        policy.set_prediction_mode("full_then_truncate")
        policy.inference_chunk_size = policy.horizon
        # Avoid set_chunk_selector validation/side effects; local policies store
        # this optional component outside their registered module tree.
        object.__setattr__(policy, "chunk_selector", None)
        with capture_action_attention(policy) as maps:
            output = policy.predict_action(observation)
        predicted = output["action_pred"]
        actions = output["action"]
        if predicted.ndim != 3 or actions.ndim != 3 or predicted.shape[0] != 1 or actions.shape[0] != 1:
            raise ValueError("policy must return [1,H,D] action_pred and action tensors")
        horizon, usable = predicted.shape[1], actions.shape[1]
        if horizon != policy.horizon or usable < 1 or offset + usable > horizon:
            raise ValueError("policy did not return a valid full-horizon executable slice")
        if not torch.equal(actions, predicted[:, offset:offset + usable]):
            raise ValueError("prediction_offset does not match the returned executable actions")
        if len(maps) != len(policy.policy.blocks) or not maps:
            raise RuntimeError("missing self-attention for one or more ARP layers")
        expected_tokens = policy.n_obs_steps + policy.plan_steps + horizon
        slices = []
        for a in maps.values():
            if a.ndim != 4 or a.shape[0] != 1 or a.shape[-2:] != (expected_tokens, expected_tokens):
                raise RuntimeError("captured attention does not match the full ARP token sequence")
            start = expected_tokens - horizon
            slices.append(a[0, :, start:, start:])
        result = select_horizon(torch.stack(slices), config)
        # Select on the full map before translating to the executable prefix.
        # Real Threading starts at position zero; explicit adapters may differ.
        result.h_star = max(1, min(result.raw_horizon - offset, usable))
        result.actions = actions[0, :result.h_star]
        result.generated_action_tokens = horizon
        result.prediction_offset = offset
        return result
    finally:
        policy.inference_chunk_size = old_chunk
        object.__setattr__(policy, "chunk_selector", old_selector)
        policy.set_prediction_mode(old_mode)
