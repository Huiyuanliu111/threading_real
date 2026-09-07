"""Point-cloud Spatial ARP for visually conditioned closed-loop approach."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from pushbox.diffusion_policy.policy.base_image_policy import BaseImagePolicy


class ThreadingPointCloudSpatialARPolicy(BaseImagePolicy):
    """Autoregressively classify target X, then Y|X, then Z|X,Y.

    Proprioception is deliberately excluded from target prediction.  It is used
    only after visual decoding to turn the absolute 3-D target into a safe
    Cartesian delta.
    """

    def __init__(
        self,
        point_bounds: tuple[float, ...],
        goal_bounds: tuple[float, ...],
        horizon: int = 1,
        n_action_steps: int = 1,
        n_obs_steps: int = 1,
        bins: int = 96,
        hidden_dim: int = 192,
        dropout: float = 0.1,
        xyz_loss_weight: float = 20.0,
        point_localization_weight: float = 1.0,
        point_target_sigma: float = 0.025,
        point_jitter_std: float = 0.001,
        color_jitter_std: float = 0.02,
        point_dropout: float = 0.1,
        direct_point_decode: bool = False,
        translation_invariant_attention: bool = False,
        max_translation_step: float = 0.008,
        stop_distance: float = 0.003,
        max_goal_distance: float = 0.25,
        min_confidence: float = 0.08,
        action_mode: str = "cartesian_delta",
    ) -> None:
        super().__init__()
        if horizon != 1 or n_obs_steps != 1 or n_action_steps != 1:
            raise ValueError("point-cloud spatial policy currently requires all horizons to equal one")
        if action_mode != "cartesian_delta":
            raise ValueError("point-cloud spatial policy emits Cartesian deltas")
        self.horizon, self.n_action_steps, self.n_obs_steps = horizon, n_action_steps, n_obs_steps
        self.action_dim = 7
        self.agent_state_dim = 8
        self.action_mode = action_mode
        self.requires_tcp_pos = True
        self.uses_pointcloud = True
        self.rgb_keys: tuple[str, ...] = ()
        self.bins = int(bins)
        self.xyz_loss_weight = float(xyz_loss_weight)
        self.point_localization_weight = float(point_localization_weight)
        self.point_target_sigma = float(point_target_sigma)
        self.point_jitter_std = float(point_jitter_std)
        self.color_jitter_std = float(color_jitter_std)
        self.point_dropout = float(point_dropout)
        self.direct_point_decode = bool(direct_point_decode)
        self.translation_invariant_attention = bool(translation_invariant_attention)
        self.max_translation_step = float(max_translation_step)
        self.stop_distance = float(stop_distance)
        self.max_goal_distance = float(max_goal_distance)
        self.min_confidence = float(min_confidence)
        self.register_buffer("point_bounds", torch.tensor(point_bounds, dtype=torch.float32).reshape(2, 3))
        self.register_buffer("goal_bounds", torch.tensor(goal_bounds, dtype=torch.float32).reshape(2, 3))
        if torch.any(self.point_bounds[0] >= self.point_bounds[1]) or torch.any(self.goal_bounds[0] >= self.goal_bounds[1]):
            raise ValueError("invalid point/goal bounds")

        # Raw xyz, three Fourier frequencies (sin+cos), RGB, two camera IDs.
        input_dim = 3 + 3 * 3 * 2 + 3 + 2
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.global_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.x_head = nn.Linear(hidden_dim, self.bins)
        self.x_embedding = nn.Embedding(self.bins, hidden_dim)
        self.y_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self.bins))
        self.y_embedding = nn.Embedding(self.bins, hidden_dim)
        self.z_head = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self.bins))
        self.goal_offset = nn.Parameter(torch.zeros(3))
        self.normalizer = LinearNormalizer()
        self.last_spatial_diagnostics: dict[str, torch.Tensor] = {}

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self, lr: float, betas=(0.9, 0.999), transformer_weight_decay=1e-4, **kwargs):
        return torch.optim.AdamW(self.parameters(), lr=lr, betas=tuple(betas), weight_decay=transformer_weight_decay)

    def _features(self, obs: dict[str, torch.Tensor]):
        points = obs["points"][:, -1]
        colors = obs["colors"][:, -1]
        cameras = obs["camera_id"][:, -1].long().clamp(0, 1)
        lower, upper = self.point_bounds
        xyz = ((points - lower) / (upper - lower)).clamp(0, 1) * 2 - 1
        if self.training:
            xyz = xyz + torch.randn_like(xyz) * self.point_jitter_std
            colors = (colors + torch.randn_like(colors) * self.color_jitter_std).clamp(0, 1)
        frequencies = xyz[..., None] * xyz.new_tensor([1.0, 2.0, 4.0]) * torch.pi
        fourier = torch.cat((frequencies.sin(), frequencies.cos()), dim=-1)
        if self.translation_invariant_attention:
            # Preserve height and appearance, but remove absolute table X/Y.
            # The selected point coordinates are applied only after scoring.
            xyz = xyz.clone()
            fourier = fourier.clone()
            xyz[..., :2] = 0
            fourier[..., :2, :] = 0
        fourier = fourier.flatten(-2)
        camera_onehot = F.one_hot(cameras, num_classes=2).to(xyz.dtype)
        local = self.point_encoder(torch.cat((xyz, fourier, colors, camera_onehot), dim=-1))
        scores = self.attention(local).squeeze(-1)
        if self.training and self.point_dropout > 0:
            dropped = torch.rand_like(scores) < self.point_dropout
            scores = scores.masked_fill(dropped, -1e4)
        pooled_attention = (local * scores.softmax(dim=-1).unsqueeze(-1)).sum(dim=1)
        pooled_max = local.amax(dim=1)
        global_feature = self.global_projection(torch.cat((pooled_attention, pooled_max), dim=-1))
        return global_feature, scores, points

    def _target_bins(self, xyz: torch.Tensor) -> torch.Tensor:
        scaled = (xyz - self.goal_bounds[0]) / (self.goal_bounds[1] - self.goal_bounds[0])
        if torch.any((scaled < 0) | (scaled > 1)):
            raise ValueError("spatial goal lies outside configured goal_bounds")
        return (scaled * self.bins).long().clamp(0, self.bins - 1)

    def _logits(self, obs: dict[str, torch.Tensor], teacher: torch.Tensor | None = None):
        global_feature, _, _ = self._features(obs)
        return self._heads(global_feature, teacher)

    def _heads(self, global_feature: torch.Tensor, teacher: torch.Tensor | None = None):
        x_logits = self.x_head(global_feature)
        x_id = x_logits.argmax(-1) if teacher is None else teacher[:, 0]
        x_feature = self.x_embedding(x_id)
        y_logits = self.y_head(torch.cat((global_feature, x_feature), dim=-1))
        y_id = y_logits.argmax(-1) if teacher is None else teacher[:, 1]
        y_feature = self.y_embedding(y_id)
        z_logits = self.z_head(torch.cat((global_feature, x_feature, y_feature), dim=-1))
        return torch.stack((x_logits, y_logits, z_logits), dim=1)

    def _decode(self, logits: torch.Tensor, soft: bool) -> torch.Tensor:
        centers = (torch.arange(self.bins, device=logits.device, dtype=logits.dtype) + 0.5) / self.bins
        unit = (logits.softmax(-1) * centers).sum(-1) if soft else (logits.argmax(-1).to(logits.dtype) + 0.5) / self.bins
        return self.goal_bounds[0] + unit * (self.goal_bounds[1] - self.goal_bounds[0])

    def compute_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        goal = batch["spatial_goal_xyz"].float()
        target = self._target_bins(goal)
        global_feature, point_scores, points = self._features(batch["obs"])
        logits = self._heads(global_feature, target)
        ce = sum(F.cross_entropy(logits[:, axis], target[:, axis]) for axis in range(3)) / 3
        prediction = self._decode(logits, soft=True)
        xyz = F.smooth_l1_loss(prediction, goal, beta=0.005)
        # Dense geometric supervision prevents a small network from merely
        # memorizing one global descriptor per demonstration. It must place its
        # attention on observed points near the demonstrated target.
        distance2 = ((points - goal[:, None]) / self.point_target_sigma).square().sum(-1)
        target_weights = (-0.5 * distance2).softmax(dim=-1)
        localization = -(target_weights * point_scores.log_softmax(dim=-1)).sum(-1).mean()
        attention_xyz = (points * point_scores.softmax(dim=-1).unsqueeze(-1)).sum(dim=1) + self.goal_offset
        attention_error = F.smooth_l1_loss(attention_xyz, goal, beta=0.01)
        point_loss = (localization + attention_error * 20.0) * self.point_localization_weight
        if self.direct_point_decode:
            return {"point_localization": point_loss}
        return {"spatial_ar_ce": ce, "spatial_xyz": xyz * self.xyz_loss_weight, "point_localization": point_loss}

    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "tcp_pos" not in obs_dict:
            raise KeyError("point-cloud action decoding requires obs['tcp_pos']")
        logits = self._logits(obs_dict)
        if self.direct_point_decode:
            _, point_scores, points = self._features(obs_dict)
            point_probability = point_scores.softmax(-1)
            goal = (points * point_probability.unsqueeze(-1)).sum(1) + self.goal_offset
            concentration = point_probability.topk(min(32, point_probability.shape[-1]), dim=-1).values.sum(-1)
            confidence = concentration[:, None].expand(-1, 3)
        else:
            goal = self._decode(logits, soft=False)
            confidence = logits.softmax(-1).amax(-1)
        tcp = obs_dict["tcp_pos"][:, -1]
        delta = goal - tcp
        distance = delta.norm(dim=-1, keepdim=True)
        step = delta * (self.max_translation_step / distance.clamp_min(1e-8)).clamp(max=1.0)
        trustworthy = (distance <= self.max_goal_distance) & (distance >= self.stop_distance) & (confidence.amin(-1, keepdim=True) >= self.min_confidence)
        step = torch.where(trustworthy, step, torch.zeros_like(step))
        one = torch.cat((step, torch.zeros_like(step), step[:, :1] * 0), dim=-1)
        actions = one[:, None]
        self.last_spatial_diagnostics = {"goal_xyz": goal.detach(), "confidence": confidence.detach(), "distance": distance.detach(), "trustworthy": trustworthy.detach()}
        return {"action_pred": actions, "action": actions, "spatial_goal_xyz": goal, "spatial_confidence": confidence, "spatial_logits": logits}
