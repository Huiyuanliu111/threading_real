"""Translation-equivariant BEV Spatial ARP for fused RealSense point clouds."""
from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from pushbox.diffusion_policy.policy.base_image_policy import BaseImagePolicy


class ThreadingBEVSpatialARPolicy(BaseImagePolicy):
    def __init__(self, point_bounds, goal_bounds, bev_size=64, hidden_dim=96,
                 horizon=1, n_action_steps=1, n_obs_steps=1,
                 heatmap_loss_weight=1.0, xy_loss_weight=30.0, z_loss_weight=20.0,
                 max_bev_translation=0, bev_noise_std=0.0,
                 max_translation_step=.008, stop_distance=.003,
                 max_goal_distance=.25, min_confidence=.001,
                 close_distance=.012, closed_width=.012, gripper_target_width=.0,
                 post_grasp_lift_height=.05,
                 action_mode="cartesian_delta"):
        super().__init__()
        if horizon <= 0 or not 0 < n_action_steps <= horizon or n_obs_steps != 1:
            raise ValueError("BEV spatial policy requires horizon>=n_action_steps>0 and n_obs_steps=1")
        self.horizon, self.n_action_steps, self.n_obs_steps = int(horizon), int(n_action_steps), 1
        self.action_dim, self.agent_state_dim = 7, 8
        self.action_mode, self.requires_tcp_pos, self.uses_pointcloud = action_mode, True, True
        self.rgb_keys = ()
        self.bev_size = int(bev_size)
        self.heatmap_loss_weight, self.xy_loss_weight, self.z_loss_weight = heatmap_loss_weight, xy_loss_weight, z_loss_weight
        self.max_bev_translation, self.bev_noise_std = int(max_bev_translation), float(bev_noise_std)
        self.max_translation_step, self.stop_distance = float(max_translation_step), float(stop_distance)
        self.max_goal_distance, self.min_confidence = float(max_goal_distance), float(min_confidence)
        self.close_distance, self.closed_width = float(close_distance), float(closed_width)
        self.gripper_target_width = float(gripper_target_width)
        self.post_grasp_lift_height = float(post_grasp_lift_height)
        self.register_buffer("point_bounds", torch.tensor(point_bounds, dtype=torch.float32).reshape(2, 3))
        self.register_buffer("goal_bounds", torch.tensor(goal_bounds, dtype=torch.float32).reshape(2, 3))
        norm = lambda c: nn.GroupNorm(max(1, c // 16), c)
        self.encoder = nn.Sequential(
            nn.Conv2d(7, hidden_dim, 5, padding=2), norm(hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), norm(hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2), norm(hidden_dim), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4), norm(hidden_dim), nn.GELU(),
        )
        self.heatmap_head = nn.Conv2d(hidden_dim, 1, 1)
        self.z_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.normalizer = LinearNormalizer()
        self.last_spatial_diagnostics = {}

    def set_normalizer(self, normalizer): self.normalizer.load_state_dict(normalizer.state_dict())
    def get_optimizer(self, lr, betas=(.9,.999), transformer_weight_decay=1e-4, **kwargs):
        return torch.optim.AdamW(self.parameters(), lr=lr, betas=tuple(betas), weight_decay=transformer_weight_decay)

    def _forward(self, obs, teacher_xy=None):
        feature = self.encoder(obs["bev"][:, -1])
        logits = self.heatmap_head(feature)[:, 0]
        probability = logits.flatten(1).softmax(-1).reshape_as(logits)
        if teacher_xy is None:
            selection = probability
        else:
            x, y = teacher_xy[:, 0], teacher_xy[:, 1]
            yy, xx = torch.meshgrid(torch.arange(self.bev_size, device=logits.device), torch.arange(self.bev_size, device=logits.device), indexing="ij")
            selection = torch.exp(-((xx[None]-x[:,None,None]).square() + (yy[None]-y[:,None,None]).square()) / 4.0)
            selection = selection / selection.sum((1,2), keepdim=True)
        pooled = (feature * selection[:, None]).sum((2,3))
        z_unit = self.z_head(pooled).sigmoid()[:, 0]
        return logits, probability, z_unit

    def _target(self, goal):
        unit = (goal - self.goal_bounds[0]) / (self.goal_bounds[1] - self.goal_bounds[0])
        if torch.any((unit < 0) | (unit > 1)): raise ValueError("goal outside goal_bounds")
        xy = (unit[:, :2] * self.bev_size).long().clamp(0, self.bev_size-1)
        return xy, unit[:, 2]

    def compute_loss(self, batch: dict[str, Any]):
        goal = batch["spatial_goal_xyz"].float().clone()
        obs = batch["obs"]
        if self.training and self.max_bev_translation > 0:
            bev = obs["bev"].clone()
            unit = (goal[:, :2] - self.goal_bounds[0, :2]) / (self.goal_bounds[1, :2] - self.goal_bounds[0, :2])
            base_xy = (unit * self.bev_size).long().clamp(0, self.bev_size - 1)
            for index in range(len(bev)):
                dx = int(torch.randint(-self.max_bev_translation, self.max_bev_translation + 1, (), device=bev.device))
                dy = int(torch.randint(-self.max_bev_translation, self.max_bev_translation + 1, (), device=bev.device))
                dx = max(-int(base_xy[index, 0]), min(dx, self.bev_size - 1 - int(base_xy[index, 0])))
                dy = max(-int(base_xy[index, 1]), min(dy, self.bev_size - 1 - int(base_xy[index, 1])))
                bev[index] = torch.roll(bev[index], shifts=(dy, dx), dims=(-2, -1))
                if dx > 0: bev[index, :, :, :dx] = 0
                elif dx < 0: bev[index, :, :, dx:] = 0
                if dy > 0: bev[index, :, :dy, :] = 0
                elif dy < 0: bev[index, :, dy:, :] = 0
                goal[index, 0] += dx / self.bev_size * (self.goal_bounds[1, 0] - self.goal_bounds[0, 0])
                goal[index, 1] += dy / self.bev_size * (self.goal_bounds[1, 1] - self.goal_bounds[0, 1])
            if self.bev_noise_std > 0:
                bev[:, :, :3] = (bev[:, :, :3] + torch.randn_like(bev[:, :, :3]) * self.bev_noise_std).clamp(0, 1)
            obs = dict(obs); obs["bev"] = bev
        xy, z = self._target(goal)
        logits, probability, z_pred = self._forward(obs, xy)
        target_flat = xy[:, 1] * self.bev_size + xy[:, 0]
        heatmap = F.cross_entropy(logits.flatten(1), target_flat)
        gx = (probability.sum(1) * ((torch.arange(self.bev_size, device=goal.device)+.5)/self.bev_size)[None]).sum(1)
        gy = (probability.sum(2) * ((torch.arange(self.bev_size, device=goal.device)+.5)/self.bev_size)[None]).sum(1)
        pred_xy = self.goal_bounds[0,:2] + torch.stack((gx,gy),-1) * (self.goal_bounds[1,:2]-self.goal_bounds[0,:2])
        return {"bev_heatmap": heatmap*self.heatmap_loss_weight,
                "bev_xy": F.smooth_l1_loss(pred_xy, goal[:,:2], beta=.005)*self.xy_loss_weight,
                "bev_z": F.smooth_l1_loss(z_pred, z, beta=.03)*self.z_loss_weight}

    def predict_action(self, obs_dict):
        logits, probability, z_unit = self._forward(obs_dict)
        index = logits.flatten(1).argmax(-1); ix, iy = index % self.bev_size, index // self.bev_size
        unit_xy = torch.stack((ix,iy),-1).to(logits.dtype).add(.5).div(self.bev_size)
        goal = self.goal_bounds[0] + torch.cat((unit_xy,z_unit[:,None]),-1)*(self.goal_bounds[1]-self.goal_bounds[0])
        confidence = probability.flatten(1).amax(-1,keepdim=True).expand(-1,3)
        tcp=obs_dict["tcp_pos"][:,-1]
        width = obs_dict["agent_pos"][:, -1, -1:]
        closed = width <= self.closed_width
        motion_goal = torch.where(closed, goal + goal.new_tensor([0., 0., self.post_grasp_lift_height]), goal)
        delta=motion_goal-tcp; distance=delta.norm(dim=-1,keepdim=True)
        step=delta*(self.max_translation_step/distance.clamp_min(1e-8)).clamp(max=1)
        valid=(distance<=self.max_goal_distance)&(confidence[:,:1]>=self.min_confidence)
        moving=valid&(distance>=self.stop_distance)
        step=torch.where(moving,step,torch.zeros_like(step))
        # Emit a genuine action chunk: linearly interpolated, bounded TCP
        # increments, with gripper closure at the first near-goal waypoint.
        fractions = (torch.arange(1, self.horizon + 1, device=goal.device, dtype=goal.dtype) / self.horizon)[None, :, None]
        chunk_goal = tcp[:, None] + (motion_goal - tcp)[:, None] * fractions
        chunk_delta = chunk_goal[:, 1:] - chunk_goal[:, :-1]
        chunk_delta = torch.cat((chunk_goal[:, :1] - tcp[:, None], chunk_delta), dim=1)
        chunk_delta = chunk_delta * (self.max_translation_step / chunk_delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.0)
        chunk_delta = torch.where(valid[:, None], chunk_delta, torch.zeros_like(chunk_delta))
        close = valid & (~closed) & ((goal-tcp).norm(dim=-1,keepdim=True) <= self.close_distance)
        close_step = torch.zeros((len(goal), self.horizon, 1), device=goal.device, dtype=goal.dtype)
        close_step[:, -1] = torch.where(close, self.gripper_target_width - width, torch.zeros_like(width))
        actions = torch.cat((chunk_delta, torch.zeros_like(chunk_delta), close_step), -1)
        self.last_spatial_diagnostics={"goal_xyz":goal.detach(),"motion_goal_xyz":motion_goal.detach(),"confidence":confidence.detach(),"distance":distance.detach(),"trustworthy":valid.detach(),"close":close.detach(),"lift":closed.detach()}
        return {"action":actions[:, :self.n_action_steps],"action_pred":actions,"spatial_goal_xyz":goal,"spatial_confidence":confidence,"spatial_heatmap_logits":logits}
