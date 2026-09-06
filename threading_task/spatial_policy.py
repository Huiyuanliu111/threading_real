"""Vision-forced spatial ARP for the minimal real-robot approach task.

Unlike the legacy Cartesian regression policy, this model cannot produce
translation from proprioception alone. It predicts one target heatmap per
fixed camera, triangulates the heatmap peaks, and converts the resulting 3D
goal into a bounded closed-loop Cartesian step.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
from torchvision.transforms import functional as tv_functional

from pushbox.diffusion_policy.common.pytorch_util import replace_submodules
from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from pushbox.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from threading_task.spatial_geometry import (
    gaussian_heatmaps,
    load_projection_calibration,
    spatial_softargmax,
    triangulate_dlt,
)


def _sinusoidal_grid(height: int, width: int, dim: int) -> torch.Tensor:
    """Return a deterministic [H*W,D] 2D sinusoidal position embedding."""
    if dim % 4:
        raise ValueError("spatial embedding dimension must be divisible by four")
    quarter = dim // 4
    omega = torch.arange(quarter, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter - 1, 1)))
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)


class ThreadingSpatialARPolicy(BaseImagePolicy):
    """Multi-view heatmap ARP with calibrated 3D action decoding."""

    def __init__(
        self,
        shape_meta: dict[str, Any],
        calibration_path: str,
        spatial_camera_keys: tuple[str, ...] = ("sideview", "frontview"),
        horizon: int = 1,
        n_action_steps: int = 1,
        n_obs_steps: int = 2,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_obs_encoder: bool = False,
        obs_encoder_group_norm: bool = True,
        hidden_dim: int = 192,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.1,
        heatmap_size: int = 56,
        heatmap_sigma: float = 1.5,
        triangulation_loss_weight: float = 10.0,
        max_translation_step: float = 0.008,
        stop_distance: float = 0.003,
        max_goal_distance: float = 0.20,
        min_heatmap_confidence: float = 0.002,
        image_augmentation: dict[str, float] | None = None,
        action_mode: str = "cartesian_delta",
        **unused_kwargs: Any,
    ) -> None:
        super().__init__()
        action_shape = tuple(shape_meta["action"]["shape"])
        state_shape = tuple(shape_meta["obs"]["agent_pos"]["shape"])
        if action_shape != (7,) or len(state_shape) != 1:
            raise ValueError(
                "spatial policy expects 7D action and vector state, got "
                f"{action_shape}, {state_shape}"
            )
        if action_mode != "cartesian_delta":
            raise ValueError("ThreadingSpatialARPolicy only emits cartesian_delta actions")
        self.action_dim = 7
        self.agent_state_dim = int(state_shape[0])
        self.action_mode = action_mode
        self.horizon = int(horizon)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        if self.horizon <= 0 or not 0 < self.n_action_steps <= self.horizon:
            raise ValueError("require 0 < n_action_steps <= horizon")
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        self.requires_tcp_pos = True

        self.rgb_keys = tuple(
            key for key, value in shape_meta["obs"].items() if value.get("type") == "rgb"
        )
        self.spatial_camera_keys = tuple(spatial_camera_keys)
        if len(self.spatial_camera_keys) < 2:
            raise ValueError("at least two fixed spatial cameras are required")
        if not set(self.spatial_camera_keys).issubset(self.rgb_keys):
            raise ValueError("spatial_camera_keys must be RGB observation keys")
        self.spatial_camera_indices = tuple(self.rgb_keys.index(key) for key in self.spatial_camera_keys)
        shapes = {key: tuple(shape_meta["obs"][key]["shape"]) for key in self.rgb_keys}
        if any(len(shape) != 3 or shape[0] != 3 for shape in shapes.values()):
            raise ValueError(f"expected CHW RGB inputs, got {shapes}")
        if len({shape for shape in shapes.values()}) != 1:
            raise ValueError("all camera inputs must currently share one shape")
        self.image_shape = next(iter(shapes.values()))
        image_height, image_width = self.image_shape[-2:]
        self.heatmap_size = int(heatmap_size)
        self.heatmap_sigma = float(heatmap_sigma)
        if self.heatmap_size <= 1 or self.heatmap_sigma <= 0:
            raise ValueError("heatmap_size and sigma must be positive")

        builders = {
            "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights.IMAGENET1K_V1, 256),
            "resnet34": (tv_models.resnet34, tv_models.ResNet34_Weights.IMAGENET1K_V1, 256),
        }
        if backbone not in builders:
            raise ValueError(f"supported spatial backbones: {sorted(builders)}")
        builder, weights, feature_dim = builders[backbone]
        resnet = builder(weights=weights if pretrained else None)
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=resnet,
                predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                func=lambda module: nn.GroupNorm(
                    max(1, module.num_features // 16), module.num_features
                ),
            )
        # Stop at layer3: 224px -> 14x14 instead of the legacy 7x7 layer4 map.
        self.obs_encoder = nn.Sequential(*list(resnet.children())[:7])
        self.obs_projection = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.freeze_obs_encoder = bool(freeze_obs_encoder)
        if self.freeze_obs_encoder:
            self.obs_encoder.requires_grad_(False)

        with torch.no_grad():
            probe = torch.zeros(1, 3, image_height, image_width)
            grid = self.obs_projection(self.obs_encoder(probe))
        grid_height, grid_width = grid.shape[-2:]
        self.grid_shape = (int(grid_height), int(grid_width))
        self.register_buffer(
            "spatial_position_embedding",
            _sinusoidal_grid(*self.grid_shape, hidden_dim),
            persistent=True,
        )
        self.camera_embedding = nn.Embedding(len(self.rgb_keys), hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(max(1, hidden_dim // 16), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

        matrices = load_projection_calibration(
            calibration_path,
            self.spatial_camera_keys,
            self.heatmap_size,
            self.heatmap_size,
        )
        self.register_buffer("projection_matrices", torch.from_numpy(matrices), persistent=True)
        self.calibration_path = str(calibration_path)
        self.triangulation_loss_weight = float(triangulation_loss_weight)
        self.max_translation_step = float(max_translation_step)
        self.stop_distance = float(stop_distance)
        self.max_goal_distance = float(max_goal_distance)
        self.min_heatmap_confidence = float(min_heatmap_confidence)
        if not 0 < self.max_translation_step <= self.max_goal_distance:
            raise ValueError("translation step must be positive and no larger than max_goal_distance")

        self.image_augmentation = {
            key: float(value) for key, value in dict(image_augmentation or {}).items()
        }
        if self.image_augmentation.get("translate", 0.0):
            raise ValueError("image translation requires transforming spatial labels too; keep it zero")
        self.normalizer = LinearNormalizer()
        self.last_spatial_diagnostics: dict[str, torch.Tensor] = {}

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        lr: float,
        betas: tuple[float, float],
        transformer_weight_decay: float,
        obs_encoder_weight_decay: float,
        obs_encoder_lr: float | None = None,
        obs_projection_lr: float | None = None,
        **unused_kwargs: Any,
    ) -> torch.optim.Optimizer:
        encoder_lr = lr if obs_encoder_lr is None else obs_encoder_lr
        head_lr = lr if obs_projection_lr is None else obs_projection_lr
        encoder_ids = {id(parameter) for parameter in self.obs_encoder.parameters()}
        head_ids = {
            id(parameter)
            for module in (self.obs_projection, self.heatmap_head)
            for parameter in module.parameters()
        }
        groups = [
            {
                "params": [p for p in self.parameters() if id(p) not in encoder_ids | head_ids],
                "lr": lr,
                "weight_decay": transformer_weight_decay,
            },
            {
                "params": list(self.obs_projection.parameters()) + list(self.heatmap_head.parameters()),
                "lr": head_lr,
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

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.image_augmentation:
            return image
        output = image
        for name in ("brightness", "contrast", "saturation"):
            strength = self.image_augmentation.get(name, 0.0)
            if strength > 0:
                factor = 1.0 + (torch.rand((), device=image.device).item() * 2 - 1) * strength
                output = getattr(tv_functional, f"adjust_{name}")(output, factor)
        noise = self.image_augmentation.get("noise_std", 0.0)
        if noise > 0:
            output = output + torch.randn_like(output) * noise
        return output.clamp(0, 1)

    def _heatmap_logits(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [key for key in self.rgb_keys if key not in observations]
        if missing:
            raise KeyError(f"spatial observation is missing {missing}")
        first = observations[self.rgb_keys[0]]
        batch = first.shape[0]
        mean = first.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = first.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tokens = []
        for camera_index, key in enumerate(self.rgb_keys):
            image = self._augment(observations[key][:, self.n_obs_steps - 1])
            feature = self.obs_projection(self.obs_encoder((image - mean) / std))
            flat = feature.flatten(2).transpose(1, 2)
            flat = flat + self.spatial_position_embedding[None]
            flat = flat + self.camera_embedding.weight[camera_index][None, None]
            tokens.append(flat)
        # Deliberately exclude joints and TCP here. Otherwise a tiny dataset lets
        # the model identify a demonstration trajectory from proprioception and
        # satisfy the heatmap loss without observing the block.
        encoded = self.encoder(torch.cat(tokens, dim=1))
        spatial_count = self.grid_shape[0] * self.grid_shape[1]
        encoded_maps = encoded.reshape(batch, len(self.rgb_keys), spatial_count, -1)
        logits = []
        for index in self.spatial_camera_indices:
            fmap = encoded_maps[:, index].transpose(1, 2).reshape(
                batch, -1, *self.grid_shape
            )
            camera_logits = self.heatmap_head(fmap)
            camera_logits = F.interpolate(
                camera_logits,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )
            logits.append(camera_logits[:, 0])
        return torch.stack(logits, dim=1)

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        logits = self._heatmap_logits(batch["obs"])
        image_height, image_width = self.image_shape[-2:]
        target_pixels = batch["spatial_goal_pixels"].to(logits.dtype).clone()
        target_pixels[..., 0] *= self.heatmap_size / image_width
        target_pixels[..., 1] *= self.heatmap_size / image_height
        valid = batch["spatial_goal_valid"].bool()
        target = gaussian_heatmaps(
            target_pixels,
            self.heatmap_size,
            self.heatmap_size,
            self.heatmap_sigma,
        )
        per_camera = -(target.flatten(2) * logits.flatten(2).log_softmax(dim=-1)).sum(dim=-1)
        if not valid.any():
            raise ValueError("batch contains no valid spatial goal projections")
        heatmap_loss = per_camera[valid].mean()

        soft_pixels, _ = spatial_softargmax(logits)
        predicted_xyz = triangulate_dlt(soft_pixels, self.projection_matrices)
        xyz_loss = F.smooth_l1_loss(
            predicted_xyz,
            batch["spatial_goal_xyz"].to(predicted_xyz.dtype),
            beta=0.01,
        )
        return {
            "spatial_heatmap_ce": heatmap_loss,
            "spatial_xyz": xyz_loss * self.triangulation_loss_weight,
        }

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "tcp_pos" not in obs_dict:
            raise KeyError("spatial action decoding requires obs['tcp_pos']")
        logits = self._heatmap_logits(obs_dict)
        pixels, confidence = spatial_softargmax(logits)
        goal_xyz = triangulate_dlt(pixels, self.projection_matrices)
        tcp = obs_dict["tcp_pos"][:, self.n_obs_steps - 1]
        delta = goal_xyz - tcp
        distance = delta.norm(dim=-1, keepdim=True)
        bounded = delta * (self.max_translation_step / distance.clamp_min(1e-8)).clamp(max=1.0)
        trustworthy = (
            torch.isfinite(goal_xyz).all(dim=-1, keepdim=True)
            & (distance <= self.max_goal_distance)
            & (distance >= self.stop_distance)
            & (confidence.min(dim=-1, keepdim=True).values >= self.min_heatmap_confidence)
        )
        bounded = torch.where(trustworthy, bounded, torch.zeros_like(bounded))
        one_action = torch.cat((bounded, torch.zeros_like(bounded), bounded[:, :1] * 0), dim=-1)
        actions = one_action[:, None].repeat(1, self.horizon, 1)
        self.last_spatial_diagnostics = {
            "goal_xyz": goal_xyz.detach(),
            "goal_pixels": pixels.detach(),
            "confidence": confidence.detach(),
            "distance": distance.detach(),
            "trustworthy": trustworthy.detach(),
        }
        return {
            "action_pred": actions,
            "action": actions[:, : self.n_action_steps],
            "spatial_goal_xyz": goal_xyz,
            "spatial_goal_pixels": pixels,
            "spatial_confidence": confidence,
            "spatial_heatmap_logits": logits,
        }
