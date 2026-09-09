#!/usr/bin/env python3
"""Run LeRobot training with training-only dropout of pi0.5 state prompt tokens."""

from __future__ import annotations

import os
import runpy

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch


# PaliGemma tokenizer IDs for the exact separators emitted by
# Pi05PrepareStateTokenizerProcessorStep: `, State: ...;\nAction: `.
STATE_PREFIX_IDS = (235269, 3040, 235292)
STATE_SUFFIX_ID = 235289


def replace_gripper_with_noop(actions: torch.Tensor, target: float) -> torch.Tensor:
    """Return actions with the recorded gripper channel replaced by a constant no-op."""
    if actions.shape[-1] <= 6:
        raise ValueError(f"expected an action tensor with at least 7 dimensions, got {actions.shape}")
    actions = actions.clone()
    actions[..., 6] = target
    return actions


def configure_visual_expert_finetuning(policy: PI05Policy, vision_lr_scale: float) -> None:
    """Freeze language weights and train vision, projection, and action modules."""
    core = policy.model
    for parameter in core.parameters():
        parameter.requires_grad = False

    visual = core.paligemma_with_expert.paligemma.model.vision_tower
    projector = core.paligemma_with_expert.paligemma.model.multi_modal_projector
    action_expert = core.paligemma_with_expert.gemma_expert.model
    full_lr_modules = (
        projector,
        action_expert,
        core.action_in_proj,
        core.action_out_proj,
        core.time_mlp_in,
        core.time_mlp_out,
    )
    for parameter in visual.parameters():
        parameter.requires_grad = True
    for module in full_lr_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    visual_ids = {id(parameter) for parameter in visual.parameters()}
    visual_parameters = [parameter for parameter in core.parameters() if id(parameter) in visual_ids]
    other_parameters = [
        parameter
        for parameter in core.parameters()
        if parameter.requires_grad and id(parameter) not in visual_ids
    ]
    policy._pi05_optimizer_groups = [
        {"params": other_parameters, "name": "action_and_projector"},
        {
            "params": visual_parameters,
            "lr": policy.config.optimizer_lr * vision_lr_scale,
            "name": "vision",
        },
    ]


def selective_optim_params(policy: PI05Policy):
    groups = getattr(policy, "_pi05_optimizer_groups", None)
    return groups if groups is not None else policy.parameters()


def drop_state_prompt_tokens(
    tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the `, State: ...` span independently from selected batch rows."""
    selected = torch.rand(tokens.shape[0], device=tokens.device) < probability
    if not bool(selected.any()):
        return tokens, attention_mask

    output_tokens = tokens.clone()
    output_mask = attention_mask.clone()
    prefix_length = len(STATE_PREFIX_IDS)
    for row in selected.nonzero(as_tuple=False).flatten().tolist():
        active_length = int(attention_mask[row].sum().item())
        active = tokens[row, :active_length]
        start = None
        for index in range(active_length - prefix_length + 1):
            if tuple(active[index : index + prefix_length].tolist()) == STATE_PREFIX_IDS:
                start = index
                break
        if start is None:
            raise RuntimeError("selected pi0.5 prompt has no ', State:' token span")
        suffix_matches = (active[start + prefix_length :] == STATE_SUFFIX_ID).nonzero(
            as_tuple=False
        )
        if suffix_matches.numel() == 0:
            raise RuntimeError("selected pi0.5 prompt has no state-closing semicolon")
        suffix = start + prefix_length + int(suffix_matches[0].item())
        kept = torch.cat((active[:start], active[suffix:]))
        pad_id = int(tokens[row, active_length].item()) if active_length < tokens.shape[1] else 0
        output_tokens[row].fill_(pad_id)
        output_tokens[row, : kept.numel()] = kept
        output_mask[row].fill_(False)
        output_mask[row, : kept.numel()] = True
    return output_tokens, output_mask


def main() -> None:
    probability = float(os.environ.get("PI05_PROPRIO_DROPOUT", "0"))
    if not 0.0 <= probability < 1.0:
        raise ValueError("PI05_PROPRIO_DROPOUT must be in [0, 1)")
    ignore_gripper = os.environ.get("PI05_IGNORE_GRIPPER_ACTION", "false").lower() == "true"
    gripper_target = float(os.environ.get("PI05_GRIPPER_TARGET_NORMALIZED", "0"))
    finetune_mode = os.environ.get("PI05_FINETUNE_MODE", "default")
    vision_lr_scale = float(os.environ.get("PI05_VISION_LR_SCALE", "0.1"))
    if finetune_mode not in {"default", "visual_expert"}:
        raise ValueError(f"unknown PI05_FINETUNE_MODE: {finetune_mode}")
    if not 0.0 < vision_lr_scale <= 1.0:
        raise ValueError("PI05_VISION_LR_SCALE must be in (0, 1]")

    original_forward = PI05Pytorch.forward
    original_policy_init = PI05Policy.__init__

    def policy_init_with_finetuning(self, *args, **kwargs):
        original_policy_init(self, *args, **kwargs)
        if finetune_mode == "visual_expert":
            configure_visual_expert_finetuning(self, vision_lr_scale)
            trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
            total = sum(parameter.numel() for parameter in self.parameters())
            print(
                "Selective finetuning: vision tower at "
                f"{vision_lr_scale:g}x LR; multimodal projector and action expert at full LR; "
                f"PaliGemma language backbone frozen; trainable={trainable}/{total}",
                flush=True,
            )

    def forward_with_state_dropout(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise,
        time,
        prefix_mask=None,
        states=None,
        state_masks=None,
    ):
        if probability > 0 and self.training:
            tokens, masks = drop_state_prompt_tokens(tokens, masks, probability)
        if ignore_gripper:
            # The threading task starts with the object already grasped. Recorded
            # gripper deltas are therefore irrelevant artifacts. Replace them
            # before x_t and the flow target are built, while retaining a
            # supervised constant no-op channel for stable diffusion sampling.
            actions = replace_gripper_with_noop(actions, gripper_target)
        return original_forward(
            self,
            images,
            img_masks,
            tokens,
            masks,
            actions,
            noise,
            time,
            prefix_mask=prefix_mask,
            states=states,
            state_masks=state_masks,
        )

    PI05Policy.__init__ = policy_init_with_finetuning
    PI05Policy.get_optim_params = selective_optim_params
    PI05Pytorch.forward = forward_with_state_dropout
    if probability > 0:
        print(f"Enabled training-only pi0.5 state-prompt dropout: p={probability}", flush=True)
    else:
        print("State-prompt dropout disabled; using full state prompts", flush=True)
    if ignore_gripper:
        print(
            "Ignoring recorded gripper actions; "
            f"normalized no-op target={gripper_target:.9g}",
            flush=True,
        )
    runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")


if __name__ == "__main__":
    main()
