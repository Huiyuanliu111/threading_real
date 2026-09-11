"""Original-style MVT + ViT + ARP policy for real Threading."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from pushbox import arp
from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from pushbox.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from threading_task.mvt_renderer import OrthographicMVTRenderer, pixel_location_channels
from threading_task.mvt_vit import MVTVisualTransformer
from threading_task.lamb import Lamb


def _rotation_from_control_points(points: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(points[..., 1, :] - points[..., 0, :], dim=-1)
    y0 = points[..., 2, :] - points[..., 0, :]
    y = torch.nn.functional.normalize(y0 - (y0 * x).sum(-1, keepdim=True) * x, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def _matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor:
    """Stable batched SO(3) logarithm for deployment action decoding."""
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6))
    vector = torch.stack((matrix[..., 2, 1] - matrix[..., 1, 2],
                          matrix[..., 0, 2] - matrix[..., 2, 0],
                          matrix[..., 1, 0] - matrix[..., 0, 1]), dim=-1)
    scale = angle / (2 * torch.sin(angle)).clamp_min(1e-6)
    result = vector * scale[..., None]
    small = angle < 1e-3
    return torch.where(small[..., None], vector * 0.5, result)


class ThreadingMVTARPPolicy(BaseImagePolicy):
    """Two virtual-view spatial ARP with a fully trainable ViT backbone."""

    def __init__(self, horizon: int = 10, n_action_steps: int = 5,
                 n_obs_steps: int = 1, image_size: int = 420, patch_size: int = 14,
                 hidden_dim: int = 128, vit_depth: int = 8, vit_heads: int = 8,
                 vit_mlp_dim: int = 256, arp_depth: int = 4, dropout: float = 0.1,
                 scene_bounds=(0.15, -0.40, -0.15, 0.75, 0.30, 0.50),
                 num_latents: int = 1, pointcloud_max_points: int = 131072,
                 plan_steps: int = 0, reverse_plan: bool = True,
                 action_chunk_size: int = 1, predict_gripper: bool = True) -> None:
        super().__init__()
        if n_obs_steps != 1 or image_size % patch_size:
            raise ValueError("MVT requires n_obs_steps=1 and image_size divisible by patch_size")
        self.horizon, self.n_action_steps, self.n_obs_steps = horizon, n_action_steps, n_obs_steps
        if not 1 <= n_action_steps <= horizon:
            raise ValueError("require 1 <= n_action_steps <= horizon")
        if not 0 <= plan_steps <= horizon or not 1 <= action_chunk_size <= horizon:
            raise ValueError("require 0 <= plan_steps <= horizon and 1 <= action_chunk_size <= horizon")
        self.plan_steps, self.reverse_plan = plan_steps, reverse_plan
        self.action_chunk_size = action_chunk_size
        self.predict_gripper = predict_gripper
        self.action_tokens = 6 + int(predict_gripper)
        self.action_dim, self.action_mode = 7, "cartesian_delta"
        self.uses_pointcloud, self.requires_tcp_pos = True, False
        self.uses_mvt = True
        self.pointcloud_max_points = int(pointcloud_max_points)
        if self.pointcloud_max_points <= 0:
            raise ValueError("pointcloud_max_points must be positive")
        self.axis_length = 0.04
        self.rgb_keys: tuple[str, ...] = ()
        self.image_size, self.patch_size, self.hidden_dim = image_size, patch_size, hidden_dim
        self.renderer = OrthographicMVTRenderer(image_size, scene_bounds)
        self.patchify = nn.Sequential(nn.Conv2d(10, hidden_dim, patch_size, patch_size),
                                      nn.BatchNorm2d(hidden_dim), nn.ReLU())
        side = image_size // patch_size
        self.vit = MVTVisualTransformer(2 * side * side, hidden_dim, vit_depth,
                                        vit_heads, vit_mlp_dim, dropout)
        self.register_buffer("pixel_loc", pixel_location_channels(2, image_size), persistent=False)
        tokens = [
            arp.TokenType.make(name="current-point", dim=2, is_continuous=True,
                               embedding="position_2d", is_control=True),
            arp.TokenType.make(name="target-point", dim=2, is_continuous=True,
                               dict_sizes=[image_size, image_size],
                               embedding="feat_grid_2d",
                               embedding_kwargs={"sampling_from": "visual-featmap", "stride": patch_size},
                               predictor="upsample_from_2d_attn",
                               predictor_kwargs={"attn_with": "action-predict-featmap",
                                                 "upscale_ratio": patch_size,
                                                 "label_name": "target-heatmap"}),
            arp.TokenType.make(name="target-gripper", dim=1, is_continuous=True,
                               embedding="linear", predictor="gmm",
                               predictor_kwargs={"num_latents": num_latents,
                                                 "low_var_eval": True}),
        ]
        if plan_steps:
            tokens.append(arp.TokenType.make(
                name="coarse-plan", dim=2, is_continuous=True,
                dict_sizes=[image_size, image_size], embedding="position_2d",
                predictor="upsample_from_2d_attn",
                predictor_kwargs={"attn_with": "plan-featmap",
                                  "upscale_ratio": patch_size,
                                  "label_name": "plan-heatmap"}))
        self.policy = arp.AutoRegressivePolicy(arp.ModelConfig(
            n_embd=hidden_dim, embd_pdrop=dropout, max_seq_len=6 + plan_steps * 6 + horizon * self.action_tokens,
            max_chunk_size=max(self.action_tokens * action_chunk_size, 6 * plan_steps),
            layer_norm_every_block=False, tokens=tokens,
            layers=[arp.LayerType.make(n_head=8, AdaLN=True,
                                       condition_on="visual-tokens") for _ in range(arp_depth)]))
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(self, lr: float, betas=(0.9, 0.999),
                      transformer_weight_decay=1e-4, **kwargs):
        # All groups include the patchifier and ViT: visual fine-tuning is intentional.
        return Lamb(self.parameters(), lr=lr, betas=tuple(betas),
                    weight_decay=transformer_weight_decay)

    def _visual(self, obs: dict[str, torch.Tensor]):
        points, colors = obs["points"][:, -1], obs["colors"][:, -1]
        valid = obs.get("valid_points")
        if valid is not None: valid = valid[:, -1]
        rendered = self.renderer(points, colors, valid)
        loc = self.pixel_loc.to(rendered).unsqueeze(0).expand(len(rendered), -1, -1, -1, -1)
        image = torch.cat((rendered, loc), dim=2).flatten(0, 1)
        patches = self.patchify(image)
        side = self.image_size // self.patch_size
        featmap = patches.reshape(len(points), 2, self.hidden_dim, side, side)
        sequence = featmap.permute(0, 1, 3, 4, 2).reshape(len(points), -1, self.hidden_dim)
        global_feat, visual_tokens = self.vit(sequence)
        encoded = visual_tokens.reshape(len(points), 2, side, side, self.hidden_dim)
        encoded = encoded.permute(0, 1, 4, 2, 3).contiguous()
        return global_feat, visual_tokens, encoded

    def _plan_targets(self, control: torch.Tensor, action_is_pad: torch.Tensor):
        """PushT-style temporal resampling of valid control-point trajectories.

        These are sparse geometric guides, not interpolated rigid transforms.
        As in PushT, reverse_plan only reverses the coarse sequence.
        """
        plans = []
        for trajectory, is_pad in zip(control, action_is_pad):
            valid = trajectory[~is_pad.bool()]
            if not len(valid):
                raise ValueError("a plan requires at least one valid target pose")
            channels = valid.flatten(1).transpose(0, 1).unsqueeze(0)
            sampled = torch.nn.functional.interpolate(
                channels, size=self.plan_steps, mode="linear",
                align_corners=self.plan_steps >= 3)
            plans.append(sampled[0].transpose(0, 1).reshape(self.plan_steps, 3, 3))
        plan = torch.stack(plans)
        return plan.flip(1) if self.reverse_plan else plan

    def _spatial_features(self, encoded: torch.Tensor, steps: int):
        # Match flattened (batch, time, anchor, view) token order.
        return encoded[:, None, None].expand(-1, steps, 3, -1, -1, -1, -1).reshape(
            -1, self.hidden_dim, encoded.shape[-2], encoded.shape[-1])

    def _heatmaps(self, pixels: torch.Tensor):
        flat_target = pixels.reshape(-1, 2)
        axis = torch.arange(self.image_size, device=pixels.device, dtype=pixels.dtype)
        dx = axis[None, None, :] - flat_target[:, 0, None, None]
        dy = axis[None, :, None] - flat_target[:, 1, None, None]
        distance2 = dx.square() + dy.square()
        heatmap = torch.exp(-distance2 / (2 * 1.5 ** 2))
        heatmap = heatmap.masked_fill(distance2 > (3 * 1.5) ** 2, 0)
        return heatmap / heatmap.sum((-2, -1), keepdim=True).clamp_min(1e-6)

    def _training_sequence(self, batch: dict[str, Any], encoded: torch.Tensor):
        bsz = len(encoded); views, anchors = 2, 3
        current = self.renderer.project(self.renderer.to_cube(
            batch["obs"]["control_points"][:, -1])).reshape(bsz, anchors * views, 2)
        target = self.renderer.project(self.renderer.to_cube(batch["target_control_points"]))
        target = target.reshape(bsz, self.horizon, anchors * views, 2)
        values = [current]; ids = [torch.zeros((bsz, anchors * views, 1), device=current.device)]
        chunks = [torch.arange(anchors * views, dtype=torch.long, device=current.device)]
        contexts = {}
        if self.plan_steps:
            plan = self._plan_targets(batch["target_control_points"], batch["action_is_pad"])
            plan_px = self.renderer.project(self.renderer.to_cube(plan)).reshape(bsz, -1, 2)
            # Rasterized supervision, matching PushT's coarse-plan labels.
            plan_px = plan_px.round().clamp(0, self.image_size - 1)
            values.append(plan_px)
            ids.append(torch.full((bsz, self.plan_steps * 6, 1),
                                  self.policy.token_name_2_ids["coarse-plan"], device=current.device))
            chunks.append(torch.full((self.plan_steps * 6,), 6, dtype=torch.long, device=current.device))
            contexts.update({"plan-featmap": self._spatial_features(encoded, self.plan_steps),
                             "plan-heatmap": self._heatmaps(plan_px)})
        action_start_chunk = anchors * views + bool(self.plan_steps)
        for step in range(self.horizon):
            values.append(target[:, step])
            ids.append(torch.ones((bsz, anchors * views, 1), device=current.device))
            if self.predict_gripper:
                values.append(batch["target_gripper"][:, step:step+1].reshape(bsz, 1, 1))
                ids.append(torch.full((bsz, 1, 1), 2, device=current.device))
            chunks.append(torch.full((self.action_tokens,), action_start_chunk + step // self.action_chunk_size,
                                     dtype=torch.long, device=current.device))
        vals = arp.cat_uneven_blc_tensors(*values)
        token_ids = torch.cat(ids, dim=1)
        tks = torch.cat((vals, token_ids), dim=-1)
        chk_ids = torch.cat(chunks)
        valid = torch.ones((bsz, tks.shape[1]), dtype=torch.bool, device=tks.device)
        action_valid = ~batch["action_is_pad"].bool()
        offset = anchors * views + self.plan_steps * 6
        for step in range(self.horizon):
            valid[:, offset:offset + self.action_tokens] = action_valid[:, step:step+1]
            offset += self.action_tokens
        spatial_feat = self._spatial_features(encoded, self.horizon)
        point_valid = action_valid[..., None].expand(-1, -1, 6).reshape(-1)
        contexts.update({"visual-featmap": spatial_feat,
                         "action-predict-featmap": spatial_feat[point_valid],
                         "target-heatmap": self._heatmaps(target.reshape(-1, 2)[point_valid])})
        return tks, chk_ids, valid, contexts

    def compute_loss(self, batch: dict[str, Any]):
        _, visual_tokens, encoded = self._visual(batch["obs"])
        tks, chunks, valid, contexts = self._training_sequence(batch, encoded)
        return self.policy.compute_loss(tks, chunks, valid_tk_mask=valid,
            contexts={**contexts, "visual-tokens": visual_tokens})

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, torch.Tensor]):
        self.eval()
        _, visual_tokens, encoded = self._visual(obs_dict)
        bsz, anchors, views = len(encoded), 3, 2
        current_px = self.renderer.project(self.renderer.to_cube(
            obs_dict["control_points"][:, -1])).reshape(bsz, anchors * views, 2)
        ids = torch.zeros((bsz, anchors * views, 1), device=current_px.device)
        prompt = torch.cat((current_px, ids), dim=-1)
        target_id = self.policy.token_name_2_ids["target-point"]
        grip_id = self.policy.token_name_2_ids["target-gripper"]
        future = []
        feature_context = {}
        predict_context = {}
        contexts = {"visual-tokens": visual_tokens}
        if self.plan_steps:
            plan_id = self.policy.token_name_2_ids["coarse-plan"]
            future.extend([{"tk_id": plan_id, "chk_id": 6}] * (self.plan_steps * 6))
            contexts["plan-featmap"] = self._spatial_features(encoded, self.plan_steps)
        action_start_chunk = anchors * views + bool(self.plan_steps)
        for step in range(self.horizon):
            chunk = action_start_chunk + step // self.action_chunk_size
            future.extend([{"tk_id": target_id, "chk_id": chunk}] * (anchors * views))
            if self.predict_gripper:
                future.append({"tk_id": grip_id, "chk_id": chunk})
            if step % self.action_chunk_size == 0:
                # Embedding sees previous actions; the predictor sees the new chunk.
                feature_context[str(chunk)] = self._spatial_features(encoded, step)
                predict_context[str(chunk)] = self._spatial_features(
                    encoded, min(self.action_chunk_size, self.horizon - step))
        generated = self.policy.generate(prompt, future, sample=False,
            contexts={**contexts, "visual-featmap": feature_context,
                      "action-predict-featmap": predict_context})
        plan_end = anchors * views + self.plan_steps * 6
        produced = generated[:, plan_end:]
        pixels, grippers = [], []
        for step in range(self.horizon):
            block = produced[:, step * self.action_tokens:(step + 1) * self.action_tokens]
            pixels.append(block[:, :6, :2].reshape(bsz, anchors, views, 2))
            if self.predict_gripper:
                grippers.append(block[:, 6, 0])
        control = self.renderer.from_cube(self.renderer.unproject_pair(torch.stack(pixels, 1)))
        origins = control[..., 0, :]
        rotations = _rotation_from_control_points(control)
        current_control = obs_dict["control_points"][:, -1]
        previous_origin = current_control[:, 0]
        previous_rotation = _rotation_from_control_points(current_control)
        previous_gripper = obs_dict["agent_pos"][:, -1, 7]
        actions = []
        for step in range(self.horizon):
            action = torch.zeros((bsz, 7), device=origins.device)
            action[:, :3] = origins[:, step] - previous_origin
            action[:, 3:6] = _matrix_to_rotvec(rotations[:, step] @ previous_rotation.transpose(-1, -2))
            if self.predict_gripper:
                action[:, 6] = grippers[step] - previous_gripper
                previous_gripper = grippers[step]
            actions.append(action)
            previous_origin, previous_rotation = origins[:, step], rotations[:, step]
        prediction = torch.stack(actions, 1)
        result = {"action_pred": prediction, "action": prediction[:, :self.n_action_steps],
                  "target_control_points": control}
        if self.plan_steps:
            plan_pixels = generated[:, anchors * views:plan_end, :2].reshape(
                bsz, self.plan_steps, anchors, views, 2)
            plan_control = self.renderer.from_cube(self.renderer.unproject_pair(plan_pixels))
            # Expose plans in chronological order for diagnostics, regardless of token order.
            result["plan_control_points"] = plan_control.flip(1) if self.reverse_plan else plan_control
        return result
