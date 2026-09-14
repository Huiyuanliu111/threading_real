"""Shared MVT token preparation for frozen-feature training and live inference."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def selector_tokens(visual_tokens, *, side: int, pool_grid: int = 0):
    if visual_tokens.ndim != 3 or visual_tokens.shape[1] != 2 * side * side:
        raise ValueError("expected two MVT virtual views in [B, 2*side*side, D] order")
    if not 0 <= pool_grid <= side:
        raise ValueError("pool_grid must be zero (all tokens) or at most the MVT grid size")
    tokens = visual_tokens
    if pool_grid:
        batch, _, channels = tokens.shape
        maps = tokens.reshape(batch * 2, side, side, channels).permute(0, 3, 1, 2)
        tokens = F.adaptive_avg_pool2d(maps, pool_grid).permute(0, 2, 3, 1)
        tokens = tokens.reshape(batch, 2 * pool_grid * pool_grid, channels)
        side = pool_grid
    spatial_count = side * side
    return tokens, dict(
        camera_ids=torch.arange(2, device=tokens.device).repeat_interleave(spatial_count),
        spatial_ids=torch.arange(spatial_count, device=tokens.device).repeat(2),
    )


def select_from_visual(policy, selector, visual):
    tokens, ids = selector_tokens(
        visual[1], side=policy.image_size // policy.patch_size,
        pool_grid=int(selector.config.metadata.get("mvt_pool_grid", 0)),
    )
    if selector.config.num_cameras == 0:
        ids.pop("camera_ids")
    if selector.config.max_spatial_positions == 0:
        ids.pop("spatial_ids")
    return selector.select(tokens, **ids)


@torch.inference_mode()
def predict_with_selector(policy, selector, obs, *, prediction_mode="full_then_truncate"):
    """Run one MVT encoding for both action prediction and chunk selection.

    Return all generated actions for the runner's guards. For batched calls,
    use the largest requested prefix so the returned tensors remain rectangular.
    Live deployment uses batch size one.
    """
    policy.eval()
    selector.eval()
    visual = policy._visual(obs)
    selection = select_from_visual(policy, selector, visual)
    requested_steps = int(selection.chunk_sizes.max().item())
    prediction = policy.predict_action(obs, visual_features=visual,
                                      prediction_mode=prediction_mode,
                                      requested_steps=requested_steps)
    return prediction, selection


@torch.inference_mode()
def predict_for_execution(policy, obs, *, prediction_mode, execute_steps,
                          selector=None, schedule=None):
    """Resolve the required prefix before decoding, without updating schedule state."""
    if selector is not None and schedule is not None:
        raise ValueError("choose either selector or endpoint schedule")
    if selector is not None:
        return predict_with_selector(policy, selector, obs, prediction_mode=prediction_mode)
    requested = execute_steps
    if schedule is not None:
        requested = schedule.fine_steps if schedule.fine_mode else schedule.coarse_steps
    return policy.predict_action(obs, prediction_mode=prediction_mode,
                                 requested_steps=requested), None
