#!/usr/bin/env python3
"""Run LeRobot training with training-only dropout of pi0.5 state prompt tokens."""

from __future__ import annotations

import os
import runpy
import json
from pathlib import Path

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch


# PaliGemma tokenizer IDs for the exact separators emitted by
# Pi05PrepareStateTokenizerProcessorStep: `, State: ...;\nAction: `.
STATE_PREFIX_IDS = (235269, 3040, 235292)
STATE_SUFFIX_ID = 235289
STATE_DIMENSIONS = {"joint": 8, "tcp_pose": 8, "tcp_pose_6d": 10}
STATE_METADATA_FILE = "state_representation.json"


def replace_gripper_with_noop(actions: torch.Tensor, target: float) -> torch.Tensor:
    """Return actions with the recorded gripper channel replaced by a constant no-op."""
    if actions.shape[-1] <= 6:
        raise ValueError(f"expected an action tensor with at least 7 dimensions, got {actions.shape}")
    actions = actions.clone()
    actions[..., 6] = target
    return actions


def configure_visual_expert_finetuning(
    policy: PI05Policy,
    *,
    train_expert_mlp: bool,
    vision_lr: float,
    projector_lr: float,
    expert_attention_lr: float,
    expert_mlp_lr: float,
    action_lr: float,
) -> None:
    """Train the visual path and the selected action-expert submodules."""
    core = policy.model
    for parameter in core.parameters():
        parameter.requires_grad = False

    visual = core.paligemma_with_expert.paligemma.model.vision_tower
    projector = core.paligemma_with_expert.paligemma.model.multi_modal_projector
    action_expert = core.paligemma_with_expert.gemma_expert.model
    action_modules = (
        core.action_in_proj,
        core.action_out_proj,
        core.time_mlp_in,
        core.time_mlp_out,
    )
    for parameter in visual.parameters():
        parameter.requires_grad = True
    for parameter in projector.parameters():
        parameter.requires_grad = True
    for name, parameter in action_expert.named_parameters():
        # The attention projections are the route by which action tokens read
        # the visual/language prefix. The full-expert mode also adapts MLPs and
        # expert-specific normalization parameters.
        if ".self_attn." in name:
            parameter.requires_grad = True
        if train_expert_mlp:
            parameter.requires_grad = True
    for module in action_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    def parameters_of(module: torch.nn.Module) -> list[torch.nn.Parameter]:
        return [parameter for parameter in module.parameters() if parameter.requires_grad]

    attention_parameters = [
        parameter
        for name, parameter in action_expert.named_parameters()
        if parameter.requires_grad and ".self_attn." in name
    ]
    mlp_parameters = [
        parameter
        for name, parameter in action_expert.named_parameters()
        if parameter.requires_grad and ".mlp." in name
    ]
    other_expert_parameters = [
        parameter
        for name, parameter in action_expert.named_parameters()
        if parameter.requires_grad and ".self_attn." not in name and ".mlp." not in name
    ]
    action_parameters = [
        parameter
        for module in action_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    policy._pi05_optimizer_groups = [
        {"params": action_parameters, "lr": action_lr, "name": "action_projections"},
        {"params": parameters_of(projector), "lr": projector_lr, "name": "multimodal_projector"},
        {"params": attention_parameters, "lr": expert_attention_lr, "name": "expert_attention"},
    ]
    if train_expert_mlp:
        if not mlp_parameters:
            raise RuntimeError("visual_full_expert selected, but no expert MLP parameters were found")
        policy._pi05_optimizer_groups.append(
            {"params": mlp_parameters, "lr": expert_mlp_lr, "name": "expert_mlp"}
        )
        if other_expert_parameters:
            policy._pi05_optimizer_groups.append(
                {"params": other_expert_parameters, "lr": expert_mlp_lr, "name": "expert_norms"}
            )
    policy._pi05_optimizer_groups.append(
        {"params": parameters_of(visual), "lr": vision_lr, "name": "vision"}
    )

    grouped_ids = {
        id(parameter)
        for group in policy._pi05_optimizer_groups
        for parameter in group["params"]
    }
    trainable_ids = {id(parameter) for parameter in core.parameters() if parameter.requires_grad}
    if grouped_ids != trainable_ids:
        raise RuntimeError("optimizer groups do not exactly cover the trainable PI05 parameters")


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
    vision_lr = float(os.environ.get("PI05_VISION_LR", "2.5e-6"))
    projector_lr = float(os.environ.get("PI05_PROJECTOR_LR", "1e-5"))
    expert_attention_lr = float(os.environ.get("PI05_EXPERT_ATTENTION_LR", "5e-6"))
    expert_mlp_lr = float(os.environ.get("PI05_EXPERT_MLP_LR", "2.5e-6"))
    action_lr = float(os.environ.get("PI05_ACTION_LR", "1e-5"))
    high_noise_fraction = float(os.environ.get("PI05_HIGH_NOISE_FRACTION", "0"))
    high_noise_min_time = float(os.environ.get("PI05_HIGH_NOISE_MIN_TIME", "0.8"))
    fixed_eval_seed = int(os.environ.get("PI05_FIXED_EVAL_SEED", "0"))
    state_representation = os.environ.get("PI05_STATE_REPRESENTATION", "tcp_pose_6d")
    if state_representation not in STATE_DIMENSIONS:
        raise ValueError(
            f"PI05_STATE_REPRESENTATION must be one of {sorted(STATE_DIMENSIONS)}, "
            f"got {state_representation!r}"
        )
    if finetune_mode not in {"default", "visual_expert", "visual_full_expert"}:
        raise ValueError(f"unknown PI05_FINETUNE_MODE: {finetune_mode}")
    for name, value in {
        "PI05_VISION_LR": vision_lr,
        "PI05_PROJECTOR_LR": projector_lr,
        "PI05_EXPERT_ATTENTION_LR": expert_attention_lr,
        "PI05_EXPERT_MLP_LR": expert_mlp_lr,
        "PI05_ACTION_LR": action_lr,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= high_noise_fraction <= 1.0:
        raise ValueError("PI05_HIGH_NOISE_FRACTION must be in [0, 1]")
    if not 0.0 <= high_noise_min_time < 1.0:
        raise ValueError("PI05_HIGH_NOISE_MIN_TIME must be in [0, 1)")

    original_forward = PI05Pytorch.forward
    original_sample_time = PI05Pytorch.sample_time
    original_policy_forward = PI05Policy.forward
    original_policy_init = PI05Policy.__init__
    original_save_pretrained = PI05Policy.save_pretrained

    def policy_init_with_finetuning(self, *args, **kwargs):
        original_policy_init(self, *args, **kwargs)
        state_feature = self.config.input_features.get("observation.state")
        if state_feature is None:
            raise RuntimeError("pi0.5 config has no observation.state input feature")
        actual_state_dim = int(state_feature.shape[0])
        expected_state_dim = STATE_DIMENSIONS[state_representation]
        if actual_state_dim != expected_state_dim:
            raise RuntimeError(
                f"{state_representation} requires {expected_state_dim} state dimensions, "
                f"but the training config contains {actual_state_dim}"
            )
        self.state_representation = state_representation
        if finetune_mode in {"visual_expert", "visual_full_expert"}:
            configure_visual_expert_finetuning(
                self,
                train_expert_mlp=finetune_mode == "visual_full_expert",
                vision_lr=vision_lr,
                projector_lr=projector_lr,
                expert_attention_lr=expert_attention_lr,
                expert_mlp_lr=expert_mlp_lr,
                action_lr=action_lr,
            )
            trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
            total = sum(parameter.numel() for parameter in self.parameters())
            group_report = ", ".join(
                f"{group['name']}={sum(parameter.numel() for parameter in group['params'])}@{group['lr']:.2g}"
                for group in self._pi05_optimizer_groups
            )
            print(
                "Selective finetuning: PaliGemma language frozen; "
                f"{group_report}; trainable={trainable}/{total}",
                flush=True,
            )

    def save_pretrained_with_state_representation(self, save_directory, *args, **kwargs):
        result = original_save_pretrained(self, save_directory, *args, **kwargs)
        metadata = {
            "format_version": 1,
            "state_representation": state_representation,
            "state_dim": STATE_DIMENSIONS[state_representation],
        }
        path = Path(save_directory) / STATE_METADATA_FILE
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return result

    def sample_time_with_high_noise_mixture(self, bsize, device):
        time = original_sample_time(self, bsize, device)
        if high_noise_fraction <= 0:
            return time
        selected = torch.rand(bsize, device=device) < high_noise_fraction
        high_time = high_noise_min_time + (1.0 - high_noise_min_time) * torch.rand(
            bsize, device=device
        )
        return torch.where(selected, high_time, time)

    def policy_forward_with_fixed_eval(self, batch, reduction="mean"):
        if self.training or fixed_eval_seed <= 0:
            return original_policy_forward(self, batch, reduction=reduction)
        indices = batch.get("index")
        batch_seed = fixed_eval_seed
        if isinstance(indices, torch.Tensor):
            batch_seed += int(indices.detach().to(dtype=torch.int64).sum().cpu().item())
        action = batch.get("action")
        cuda_devices = []
        if isinstance(action, torch.Tensor) and action.is_cuda:
            cuda_devices = [action.device.index]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(batch_seed)
            return original_policy_forward(self, batch, reduction=reduction)

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
    PI05Policy.forward = policy_forward_with_fixed_eval
    PI05Policy.save_pretrained = save_pretrained_with_state_representation
    PI05Pytorch.forward = forward_with_state_dropout
    PI05Pytorch.sample_time = sample_time_with_high_noise_mixture
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
    if high_noise_fraction > 0:
        print(
            f"High-noise timestep mixture: fraction={high_noise_fraction:g}, "
            f"uniform_range=[{high_noise_min_time:g}, 1]",
            flush=True,
        )
    if fixed_eval_seed > 0:
        print(f"Deterministic evaluation RNG enabled: seed={fixed_eval_seed}", flush=True)
    print(
        f"State representation: {state_representation} "
        f"({STATE_DIMENSIONS[state_representation]} dimensions)",
        flush=True,
    )
    runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")


if __name__ == "__main__":
    main()
