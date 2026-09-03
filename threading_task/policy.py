"""Threading-specific ARP policy with EEF state and spatial visual tokens."""
from __future__ import annotations

import math
from types import MethodType
from typing import Any

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms.functional as tv_functional
from torchvision.transforms import InterpolationMode

from pushbox import arp
from chunk_selector.chunk_selector import ChunkSelection, ChunkSelector
from chunk_selector.execution import (
    PREDICTION_MODES,
    PredictionMode,
    prediction_execution_lengths,
)
from pushbox.diffusion_policy.common.pytorch_util import replace_submodules
from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from pushbox.diffusion_policy.policy.base_image_policy import BaseImagePolicy


def segmented_range_list(start: int, end: int, segment_size: int) -> list[int]:
    if segment_size <= 0:
        raise ValueError(f"segment_size must be positive, got {segment_size}")
    original_end = end
    if (end - start) % segment_size:
        end = start + math.ceil((end - start) / segment_size) * segment_size
    multiple = (end - start) // segment_size
    values = [start + index for index in range(multiple) for _ in range(segment_size)]
    return values[: original_end - start]


class ThreadingARPolicy(BaseImagePolicy):
    """Multi-view ARP for Threading only.

    Unlike the shared PushBox policy, this policy consumes an 8D EEF state and
    preserves a spatial ResNet feature grid from each camera as cross-attention
    context tokens.
    """

    def __init__(
        self,
        shape_meta: dict[str, Any],
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        pretrained: bool = True,
        backbone: str = "resnet18",
        camera_obs_keys: tuple[str, ...] | None = None,
        freeze_obs_encoder: bool = False,
        obs_encoder_group_norm: bool = True,
        use_view_fusion: bool = True,
        image_augmentation: dict[str, float] | None = None,
        action_mode: str = "absolute",
        arp_cfg: dict[str, Any] | None = None,
        **unused_kwargs,
    ):
        super().__init__()
        arp_cfg = arp_cfg or {}
        action_shape = tuple(shape_meta["action"]["shape"])
        state_shape = tuple(shape_meta["obs"]["agent_pos"]["shape"])
        if len(action_shape) != 1 or len(state_shape) != 1:
            raise ValueError(f"Expected vector state/action, got {state_shape=} {action_shape=}")
        self.action_dim = int(action_shape[0])
        self.agent_state_dim = int(state_shape[0])
        if self.agent_state_dim < self.action_dim:
            raise ValueError("Threading state width must be at least the action width")
        if action_mode not in {"absolute", "delta"}:
            raise ValueError("action_mode must be 'absolute' or 'delta'")
        self.action_mode = action_mode

        self.rgb_keys = tuple(
            key for key, value in shape_meta["obs"].items() if value.get("type") == "rgb"
        )
        if len(self.rgb_keys) < 2:
            raise ValueError(f"ThreadingARPolicy expects at least two RGB cameras, got {self.rgb_keys}")
        self.image_shapes = {
            key: tuple(shape_meta["obs"][key]["shape"]) for key in self.rgb_keys
        }
        if any(len(shape) != 3 or shape[0] != 3 for shape in self.image_shapes.values()):
            raise ValueError(f"Expected CHW RGB camera shapes, got {self.image_shapes}")
        self.image_shape = self.image_shapes[self.rgb_keys[0]]
        self.camera_obs_keys = tuple(camera_obs_keys) if camera_obs_keys is not None else None
        if self.camera_obs_keys is not None and len(self.camera_obs_keys) != len(self.rgb_keys):
            raise ValueError("camera_obs_keys must contain one environment key per RGB input")

        backbone_options = {
            "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.IMAGENET1K_V1),
            "resnet34": (tv_models.resnet34, tv_models.ResNet34_Weights.IMAGENET1K_V1),
            "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights.IMAGENET1K_V1),
        }
        if backbone not in backbone_options:
            raise ValueError(f"backbone must be one of {sorted(backbone_options)}, got {backbone!r}")
        backbone_builder, pretrained_weights = backbone_options[backbone]
        resnet = backbone_builder(weights=pretrained_weights if pretrained else None)
        self.backbone = backbone
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=resnet,
                predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                func=lambda module: nn.GroupNorm(
                    num_groups=module.num_features // 16,
                    num_channels=module.num_features,
                ),
            )
        self.obs_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.obs_feature_dim = int(resnet.fc.in_features)
        self.freeze_obs_encoder = bool(freeze_obs_encoder)
        if self.freeze_obs_encoder:
            self.obs_encoder.requires_grad_(False)

        self.horizon = int(horizon)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.plan_steps = int(arp_cfg.get("plan_steps", 4))
        self.plan_chunk_size = int(arp_cfg.get("plan_chunk_size", 1))
        self.action_chunk_size = int(arp_cfg.get("action_chunk_size", 4))
        self.num_latents = int(arp_cfg.get("num_latents", 1))
        self.low_var_eval = bool(arp_cfg.get("low_var_eval", True))
        self.use_sample: bool | str = arp_cfg.get("sample", True)
        self.use_view_fusion = bool(use_view_fusion)
        self.image_augmentation = {
            key: float(value)
            for key, value in dict(image_augmentation or {}).items()
        }
        unsupported_augmentations = set(self.image_augmentation) - {
            "brightness",
            "contrast",
            "saturation",
            "hue",
            "noise_std",
            "translate",
        }
        if unsupported_augmentations:
            raise ValueError(
                "Unsupported image augmentations: "
                f"{sorted(unsupported_augmentations)}"
            )
        if any(value < 0 for value in self.image_augmentation.values()):
            raise ValueError("Image augmentation strengths must be non-negative")
        n_embd = int(arp_cfg["n_embd"])
        action_predictor = str(arp_cfg.get("action_predictor", "gmm"))
        action_predictor_kwargs = dict(
            arp_cfg.get("action_predictor_kwargs", {})
        )
        if action_predictor == "gmm":
            action_predictor_kwargs.setdefault("num_latents", self.num_latents)
            action_predictor_kwargs.setdefault("low_var_eval", self.low_var_eval)

        tokens = [
            arp.TokenType.make(
                name="pos",
                dim=self.agent_state_dim,
                is_continuous=True,
                embedding="linear",
                is_control=True,
            ),
            arp.TokenType.make(
                name="coarse-plan",
                dim=self.agent_state_dim,
                is_continuous=True,
                embedding="linear",
                predictor="gmm",
                predictor_kwargs={
                    "num_latents": self.num_latents,
                    "low_var_eval": self.low_var_eval,
                },
            ),
            arp.TokenType.make(
                name="fine-action",
                dim=self.action_dim,
                is_continuous=True,
                embedding="linear",
                predictor=action_predictor,
                predictor_kwargs=action_predictor_kwargs,
            ),
        ]
        max_chunk_size = max(
            self.horizon,
            self.n_obs_steps,
            self.plan_chunk_size,
            self.action_chunk_size,
        )
        self.policy = arp.AutoRegressivePolicy(
            arp.ModelConfig(
                n_embd=n_embd,
                embd_pdrop=float(arp_cfg.get("embd_pdrop", 0.1)),
                layer_norm_every_block=bool(arp_cfg.get("layer_norm_every_block", True)),
                max_chunk_size=max_chunk_size,
                max_seq_len=(self.n_obs_steps + self.plan_steps + self.horizon) * 2,
                layers=[
                    arp.LayerType.make(
                        **arp_cfg["layer_cfg"],
                        condition_on="visual-token",
                    )
                ]
                * int(arp_cfg["num_layers"]),
                tokens=tokens,
            )
        )

        self.obs_projection = nn.Conv2d(self.obs_feature_dim, n_embd, kernel_size=1)
        self.view_fusion = nn.ModuleDict(
            {
                "state_projection": nn.Linear(self.agent_state_dim, n_embd),
                "score": nn.Sequential(
                    nn.LayerNorm(n_embd),
                    nn.Linear(n_embd, max(n_embd // 4, 32)),
                    nn.GELU(),
                    nn.Linear(max(n_embd // 4, 32), 1),
                ),
            }
        )
        self.visual_token_embeddings = nn.ModuleDict(
            {
                "camera": nn.Embedding(len(self.rgb_keys), n_embd),
                "time": nn.Embedding(self.n_obs_steps, n_embd),
                "row": nn.Embedding(8, n_embd),
                "column": nn.Embedding(8, n_embd),
            }
        )
        for embedding in self.visual_token_embeddings.values():
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        self.inference_chunk_size: int | None = None
        self.prediction_mode: PredictionMode = "full_then_truncate"
        # Kept out of the base action checkpoint and attached after strict load.
        object.__setattr__(self, "chunk_selector", None)
        self.last_chunk_features: torch.Tensor | None = None
        self.last_view_weights: dict[str, float] = {}

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer = normalizer

    def set_chunk_selector(self, selector: ChunkSelector | None) -> None:
        if selector is not None:
            selector.validate_for_policy(
                feature_dim=self.policy.cfg.n_embd,
                max_chunk=self.horizon,
            )
            selector.to(device=self.device)
            selector.eval()
            selector.requires_grad_(False)
        # Sidecar selector weights must not alter strict action checkpoint keys.
        self._modules.pop("chunk_selector", None)
        object.__setattr__(self, "chunk_selector", selector)

    def set_prediction_mode(self, mode: PredictionMode) -> None:
        if mode not in PREDICTION_MODES:
            raise ValueError(
                f"unknown prediction mode {mode!r}; expected one of {PREDICTION_MODES}"
            )
        self.prediction_mode = mode

    def get_optimizer(
        self,
        transformer_weight_decay: float,
        obs_encoder_weight_decay: float,
        lr: float,
        betas: tuple[float, float],
        obs_encoder_lr: float | None = None,
        obs_projection_lr: float | None = None,
    ) -> torch.optim.Optimizer:
        encoder_lr = lr if obs_encoder_lr is None else obs_encoder_lr
        projection_lr = lr if obs_projection_lr is None else obs_projection_lr
        groups: list[dict[str, Any]] = [
            {
                "params": self.policy.parameters(),
                "lr": lr,
                "weight_decay": transformer_weight_decay,
            },
            {
                "params": self.obs_projection.parameters(),
                "lr": projection_lr,
                "weight_decay": obs_encoder_weight_decay,
            },
            {
                "params": self.visual_token_embeddings.parameters(),
                "lr": projection_lr,
                "weight_decay": obs_encoder_weight_decay,
            },
            {
                "params": self.view_fusion.parameters(),
                "lr": projection_lr,
                "weight_decay": obs_encoder_weight_decay,
            },
        ]
        if not self.freeze_obs_encoder:
            groups.append(
                {
                    "params": self.obs_encoder.parameters(),
                    "lr": encoder_lr,
                    "weight_decay": obs_encoder_weight_decay,
                }
            )
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.predict_action(batch, training=True)

    def _augment_image_sequence(self, image: torch.Tensor) -> torch.Tensor:
        """Apply one color transform per sample, shared by all observation steps."""
        if not self.training or not self.image_augmentation:
            return image

        augmented = []
        for sample in image:
            output = sample
            for name in ("brightness", "contrast", "saturation"):
                strength = self.image_augmentation.get(name, 0.0)
                if strength <= 0:
                    continue
                factor = 1.0 + (
                    torch.rand((), device=image.device).item() * 2.0 - 1.0
                ) * strength
                adjust = getattr(tv_functional, f"adjust_{name}")
                output = adjust(output, factor)
            hue_strength = self.image_augmentation.get("hue", 0.0)
            if hue_strength > 0:
                hue_factor = (torch.rand((), device=image.device).item() * 2.0 - 1.0) * hue_strength
                output = tv_functional.adjust_hue(output, hue_factor)
            translate_strength = self.image_augmentation.get("translate", 0.0)
            if translate_strength > 0:
                height, width = output.shape[-2:]
                translate = [
                    int((torch.rand((), device=image.device).item() * 2.0 - 1.0) * translate_strength * width),
                    int((torch.rand((), device=image.device).item() * 2.0 - 1.0) * translate_strength * height),
                ]
                output = tv_functional.affine(
                    output,
                    angle=0.0,
                    translate=translate,
                    scale=1.0,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0.0,
                )
            noise_std = self.image_augmentation.get("noise_std", 0.0)
            if noise_std > 0:
                output = output + torch.randn_like(output) * noise_std
            augmented.append(output)
        return torch.stack(augmented).clamp_(0.0, 1.0)

    def _visual_tokens(
        self,
        images: dict[str, torch.Tensor],
        state_context: torch.Tensor,
    ) -> torch.Tensor:
        first = images[self.rgb_keys[0]]
        batch_size, obs_steps = first.shape[:2]
        num_cameras = len(self.rgb_keys)
        feature_grids: list[torch.Tensor] = []
        pooled_features: list[torch.Tensor] = []
        grid_shapes: list[tuple[int, int]] = []
        for key in self.rgb_keys:
            image = images[key]
            if image.shape[:2] != (batch_size, obs_steps):
                raise ValueError(f"Camera {key} has inconsistent batch/time shape {image.shape}")
            _, _, channels, height, width = image.shape
            feature = self.obs_projection(
                self.obs_encoder(image.reshape(-1, channels, height, width))
            )
            grid_height, grid_width = feature.shape[-2:]
            if (
                grid_height > self.visual_token_embeddings["row"].num_embeddings
                or grid_width
                > self.visual_token_embeddings["column"].num_embeddings
            ):
                raise ValueError(
                    f"Visual grid {(grid_height, grid_width)} exceeds the "
                    "supported 8x8 positional embedding"
                )
            num_spatial = grid_height * grid_width
            feature = feature.flatten(2).transpose(1, 2).reshape(
                batch_size,
                obs_steps,
                num_spatial,
                self.policy.cfg.n_embd,
            )
            feature_grids.append(feature)
            pooled_features.append(feature.mean(dim=2))
            grid_shapes.append((grid_height, grid_width))

        pooled = torch.stack(pooled_features, dim=2)
        if self.use_view_fusion:
            state_feature = self.view_fusion["state_projection"](state_context)[
                :, :, None, :
            ]
            view_logits = self.view_fusion["score"](
                pooled + state_feature
            ).squeeze(-1)
            view_weights = torch.softmax(view_logits, dim=2)
        else:
            view_weights = pooled.new_full(
                (batch_size, obs_steps, num_cameras),
                1.0 / num_cameras,
            )
        if not self.training:
            mean_weights = view_weights.detach().mean(dim=(0, 1)).cpu().tolist()
            self.last_view_weights = dict(zip(self.rgb_keys, mean_weights))

        device = first.device
        time_embedding = self.visual_token_embeddings["time"](
            torch.arange(obs_steps, device=device)
        )[None, :, None, None, :]
        tokens = []
        for camera_index, (feature, grid_shape) in enumerate(
            zip(feature_grids, grid_shapes)
        ):
            num_spatial = feature.shape[2]
            grid_height, grid_width = grid_shape
            camera_embedding = self.visual_token_embeddings["camera"](
                torch.as_tensor(camera_index, device=device)
            ).view(1, 1, 1, -1)
            row_embedding = self.visual_token_embeddings["row"](
                torch.arange(grid_height, device=device)
            )[:, None, :]
            column_embedding = self.visual_token_embeddings["column"](
                torch.arange(grid_width, device=device)
            )[None, :, :]
            spatial_embedding = (row_embedding + column_embedding).reshape(
                1,
                1,
                num_spatial,
                self.policy.cfg.n_embd,
            )
            scale = view_weights[:, :, camera_index, None, None] * num_cameras
            camera_tokens = (
                feature * scale
                + camera_embedding
                + time_embedding.squeeze(2)
                + spatial_embedding
            )
            tokens.append(camera_tokens.reshape(batch_size, -1, self.policy.cfg.n_embd))
        return torch.cat(tokens, dim=1)

    def _future_chunks(self, inference_horizon: int) -> tuple[list[str], list[int]]:
        token_types = ["coarse-plan"] * self.plan_steps + ["fine-action"] * inference_horizon
        chunk_ids: list[int] = []
        next_chunk_id = self.n_obs_steps
        if self.plan_steps:
            plan_ids = segmented_range_list(
                next_chunk_id,
                next_chunk_id + self.plan_steps,
                self.plan_chunk_size,
            )
            chunk_ids.extend(plan_ids)
            next_chunk_id = max(plan_ids) + 1
        action_ids = segmented_range_list(
            next_chunk_id,
            next_chunk_id + inference_horizon,
            self.action_chunk_size,
        )
        chunk_ids.extend(action_ids)
        return token_types, chunk_ids

    def predict_action(
        self,
        batch_or_obs_dict: dict[str, Any],
        training: bool = False,
    ) -> dict[str, torch.Tensor]:
        observations = (
            batch_or_obs_dict["obs"] if "obs" in batch_or_obs_dict else batch_or_obs_dict
        )
        missing = [key for key in (*self.rgb_keys, "agent_pos") if key not in observations]
        if missing:
            raise KeyError(f"Threading observation is missing {missing}")
        batch_size = observations["agent_pos"].shape[0]
        device = observations["agent_pos"].device

        imagenet_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=device
        ).view(1, 1, 3, 1, 1)
        imagenet_std = torch.tensor(
            [0.229, 0.224, 0.225], device=device
        ).view(1, 1, 3, 1, 1)
        images: dict[str, torch.Tensor] = {}
        for key in self.rgb_keys:
            image = self._augment_image_sequence(
                observations[key][:, : self.n_obs_steps]
            )
            if image.min() < 0 or image.max() > 1:
                raise ValueError(f"{key} must be in [0, 1]")
            images[key] = (image - imagenet_mean) / imagenet_std
        state = observations["agent_pos"]
        normalized_state = self.normalizer["agent_pos"].normalize(
            state.flatten(0, 1)
        ).reshape(batch_size, -1, self.agent_state_dim)
        state_context = normalized_state[:, : self.n_obs_steps]
        visual_tokens = self._visual_tokens(images, state_context)
        if not training:
            self.last_chunk_features = visual_tokens.detach()

        chunk_selection: ChunkSelection | None = None
        requested_chunk = self.inference_chunk_size
        if not training and self.chunk_selector is not None:
            chunk_selection = self.chunk_selector.select(visual_tokens.detach())
            selected_chunks = chunk_selection.chunk_sizes
            if not torch.equal(selected_chunks, selected_chunks[:1].expand_as(selected_chunks)):
                raise ValueError(
                    "Threading adaptive chunk inference currently requires one shared chunk per batch"
                )
            requested_chunk = int(selected_chunks[0].item())

        # Training always uses the checkpoint horizon. At inference the
        # explicit mode controls whether a prefix or the complete plan is
        # generated before selecting the executable actions.
        prediction_mode: PredictionMode = (
            "full_then_truncate" if training else self.prediction_mode
        )
        inference_horizon, action_steps = prediction_execution_lengths(
            mode=prediction_mode,
            prediction_horizon=self.horizon,
            default_execution_steps=self.n_action_steps,
            requested_execution_steps=(
                requested_chunk if not training else None
            ),
            max_execution_steps=self.horizon,
        )
        future_types, future_chunk_ids = self._future_chunks(inference_horizon)
        future_spec = [
            {
                "tk_id": self.policy.token_name_2_ids[token_type],
                "chk_id": chunk_id,
            }
            for token_type, chunk_id in zip(future_types, future_chunk_ids)
        ]

        if training:
            actions = batch_or_obs_dict["action"]
            all_normalized_actions = self.normalizer["action"].normalize(
                actions.flatten(0, 1)
            ).reshape(batch_size, -1, self.action_dim)
            # Observations contain frames [t-1, t], so the first action to
            # imitate online is action[t], at index n_obs_steps - 1. Build the
            # prediction target from that point instead of predicting a stale
            # action for the older observation and discarding it at inference.
            action_start = self.n_obs_steps - 1
            normalized_actions = all_normalized_actions[
                :, action_start : action_start + self.horizon
            ]
            if normalized_actions.shape[1] != self.horizon:
                raise ValueError(
                    f"Need {self.horizon} aligned action targets after index "
                    f"{action_start}, got {normalized_actions.shape[1]}"
                )
            if self.plan_steps > 0:
                future_state = normalized_state[:, self.n_obs_steps :]
                coarse_plan = F.interpolate(
                    future_state.permute(0, 2, 1),
                    size=self.plan_steps,
                    mode="linear",
                    align_corners=self.plan_steps >= 3,
                ).permute(0, 2, 1)
            else:
                # A zero-length coarse plan explicitly disables plan tokens.
                # F.interpolate does not accept an output length of zero, so
                # preserve the expected [B, 0, state_dim] sequence directly.
                coarse_plan = normalized_state.new_empty(
                    (batch_size, 0, self.agent_state_dim)
                )
            padded_actions = F.pad(
                normalized_actions,
                (0, self.agent_state_dim - self.action_dim),
            )
            token_values = torch.cat([state_context, coarse_plan, padded_actions], dim=1)
            token_names = ["pos"] * self.n_obs_steps + future_types
            token_ids = torch.as_tensor(
                [self.policy.token_name_2_ids[name] for name in token_names],
                device=device,
            ).view(1, -1, 1).repeat(batch_size, 1, 1)
            sequence = torch.cat([token_values, token_ids], dim=-1)
            valid_token_mask = None
            if "action_is_pad" in batch_or_obs_dict:
                action_is_pad = batch_or_obs_dict["action_is_pad"]
                valid_actions = ~action_is_pad[
                    :, action_start : action_start + self.horizon
                ].bool()
                prefix_is_valid = torch.ones(
                    (
                        batch_size,
                        self.n_obs_steps + self.plan_steps,
                    ),
                    dtype=torch.bool,
                    device=device,
                )
                valid_token_mask = torch.cat(
                    [prefix_is_valid, valid_actions],
                    dim=1,
                )
            return self.policy.compute_loss(
                sequence,
                chk_ids=torch.as_tensor(
                    list(range(self.n_obs_steps)) + future_chunk_ids,
                    device=device,
                ),
                contexts={"visual-token": visual_tokens},
                valid_tk_mask=valid_token_mask,
            )

        token_ids = torch.as_tensor(
            [self.policy.token_name_2_ids["pos"]] * self.n_obs_steps,
            device=device,
        ).view(1, -1, 1).repeat(batch_size, 1, 1)
        sequence = torch.cat([state_context, token_ids], dim=-1)
        generated = self.policy.generate(
            sequence,
            future_spec,
            contexts={"visual-token": visual_tokens},
            sample=self.use_sample,
        )
        normalized_prediction = generated[
            :, self.n_obs_steps + self.plan_steps :, : self.action_dim
        ]
        action_prediction = self.normalizer["action"].unnormalize(
            normalized_prediction.reshape(-1, self.action_dim)
        ).reshape(batch_size, -1, self.action_dim)
        end = min(action_steps, action_prediction.shape[1])
        result = {
            "action_pred": action_prediction,
            "action": action_prediction[:, :end],
        }
        if chunk_selection is not None:
            result.update(
                {
                    "chunk_logits": chunk_selection.logits,
                    "chunk_probabilities": chunk_selection.probabilities,
                    "chunk_size": chunk_selection.chunk_sizes,
                    "chunk_confidence": chunk_selection.confidences,
                }
            )
        return result


def enable_map_gmm_inference(policy: nn.Module) -> None:
    """Install deterministic MAP sampling only on this policy instance."""

    for predictor in policy.modules():
        if not isinstance(predictor, arp.GMMPredictor):
            continue
        original_sample = predictor.sample

        def sample_map(
            instance,
            distributions,
            do_sample,
            _original=original_sample,
            **extra_contexts,
        ):
            if do_sample != "map":
                return _original(distributions, do_sample, **extra_contexts)
            outputs = []
            for distribution in distributions:
                if isinstance(distribution, D.MixtureSameFamily):
                    means = distribution.component_distribution.base_dist.mean
                    component = distribution.mixture_distribution.probs.argmax(dim=-1)
                    index = component[..., None, None].expand(
                        *component.shape,
                        1,
                        means.shape[-1],
                    )
                    outputs.append(means.gather(-2, index).squeeze(-2))
                else:
                    outputs.append(distribution.mean)
            return outputs

        predictor.sample = MethodType(sample_map, predictor)
