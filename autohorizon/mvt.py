"""Action-step attention adapter for single-camera point-cloud MVT/PlanARP."""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch

from .core import AutoHorizonConfig, select_horizon

MVT_IMPLEMENTATION_VERSION = "official_c7504f1_mvt_step_attention_v1"


def validate_mvt_policy(policy, config: AutoHorizonConfig) -> None:
    if not getattr(policy, "uses_mvt", False):
        raise ValueError("AutoHorizon MVT adapter requires a Threading MVT/PlanARP policy")
    if policy.training or any(m.training for m in policy.policy.blocks):
        raise ValueError("AutoHorizon requires policy.eval()")
    if policy.horizon < 2:
        raise ValueError("official AutoHorizon requires horizon >= 2")
    if config.method == "bidirectional" and config.run_len >= policy.horizon:
        raise ValueError("upstream bidirectional implementation requires run_len < horizon")
    if not 1 <= policy.action_chunk_size <= policy.horizon:
        raise ValueError("MVT action_chunk_size must lie in [1, horizon]")
    if policy.action_tokens != 6 + int(policy.predict_gripper):
        raise ValueError("unexpected MVT action token layout")


def action_step_attention(attention: torch.Tensor, horizon: int, tokens_per_step: int):
    """Average query tokens and sum key tokens within each physical action step.

    Input: [..., H*K, H*K], output: [..., H, H]. This preserves attention mass
    to action keys; no normalization occurs before the official algorithm.
    """
    if horizon < 2 or tokens_per_step < 1 or attention.ndim < 2:
        raise ValueError("require horizon >= 2 and tokens_per_step >= 1")
    if attention.shape[-2:] != (horizon * tokens_per_step,) * 2:
        raise ValueError("attention shape does not match the action token layout")
    grouped = attention.reshape(*attention.shape[:-2], horizon, tokens_per_step,
                                horizon, tokens_per_step)
    return grouped.sum(dim=-1).mean(dim=-2)


@contextmanager
def capture_mvt_generation_attention(policy):
    """Keep each query row only when its action group is being generated.

    Earlier action rows are re-encoded on later passes and must not overwrite
    their generation-time attention. Future groups are causally unavailable.
    """
    start = 6 + 6 * policy.plan_steps
    total = start + policy.horizon * policy.action_tokens
    group_tokens = policy.action_chunk_size * policy.action_tokens
    maps, ends, handles = {}, {}, []
    try:
        for index, block in enumerate(policy.policy.blocks):
            def capture(module, args, index=index):
                a = args[0].detach()
                if a.ndim != 4 or a.shape[0] != 1 or a.shape[-2] != a.shape[-1]:
                    raise RuntimeError("expected square batch-one MVT self-attention")
                end = a.shape[-1]
                if end <= start:
                    return  # Coarse-plan generation, not action queries.
                previous = ends.get(index, start)
                if end != min(previous + group_tokens, total) or previous >= total:
                    raise RuntimeError("unexpected MVT action generation sequence")
                rows = a[0, :, previous:end, start:end]
                # Preserve causal zeros for keys that did not exist on this pass.
                rows = torch.nn.functional.pad(rows, (0, total - end))
                maps.setdefault(index, []).append(rows)
                ends[index] = end
            handles.append(block.attn.attn_dropout.register_forward_pre_hook(capture))
        yield maps
        if len(ends) != len(policy.policy.blocks) or not ends or any(v != total for v in ends.values()):
            raise RuntimeError("missing full MVT generation attention")
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def predict_mvt_with_autohorizon(policy, observation, config=None, *, sample=False):
    """Generate once, return (full prediction with selected prefix, decision).

    The single physical camera still renders TWO virtual views, with three
    control points per view (six spatial tokens per step, plus optional grip).
    Each action group contributes its generation-time query rows; plan tokens
    are context. Future action keys retain their causal zero attention.
    ``sample`` has the same meaning/default as MVT.predict_action. No weights,
    decoder settings, camera selection or action values are modified.
    """
    config = config or AutoHorizonConfig()
    validate_mvt_policy(policy, config)
    if not observation or any(
        not isinstance(v, torch.Tensor) or v.ndim == 0 or v.shape[0] != 1
        for v in observation.values()
    ):
        raise ValueError("AutoHorizon supports observation batch size 1")
    started = time.monotonic()
    with capture_mvt_generation_attention(policy) as maps:
        prediction = policy.predict_action(
            observation, prediction_mode="full_then_truncate",
            requested_steps=policy.horizon, sample=sample,
        )
    horizon, count = policy.horizon, policy.action_tokens
    actions = prediction["action_pred"]
    if actions.shape != (1, horizon, 7) or not torch.isfinite(actions).all():
        raise ValueError("MVT must return finite full-horizon [1,H,7] actions")
    if len(maps) != len(policy.policy.blocks) or not maps:
        raise RuntimeError("missing self-attention for one or more ARP layers")
    layers = [action_step_attention(torch.cat(rows, dim=-2), horizon, count)
              for rows in maps.values()]
    result = select_horizon(torch.stack(layers), config)
    result.actions = actions[0, :result.h_star]
    result.generated_action_tokens = horizon * count
    result.diagnostics.update(
        implementation_version=MVT_IMPLEMENTATION_VERSION,
        attention_aggregation="query_mean_key_sum_then_layer_head_mean",
        tokens_per_action=count, generated_action_steps=horizon,
        plan_token_count=6 * policy.plan_steps,
        attention_capture="per_group_generation_queries_causal_future_zero",
        action_chunk_size=policy.action_chunk_size,
        pointcloud_views=list(policy.pointcloud_views),
    )
    prediction = dict(prediction)
    prediction["action"] = actions[:, :result.h_star]
    prediction["prediction_diagnostics"] = {
        **prediction.get("prediction_diagnostics", {}), **result.metrics(),
        "prediction_mode": "full_then_truncate", "requested_steps": horizon,
        "generated_steps": horizon, "execution_chunk": result.h_star,
        "continuous_chunk": float(result.h_star), "predicted_chunk": horizon,
        "total_seconds": time.monotonic() - started,
    }
    return prediction, result
