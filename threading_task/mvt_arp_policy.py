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
                 num_latents: int = 1) -> None:
        super().__init__()
        if n_obs_steps != 1 or image_size % patch_size:
            raise ValueError("MVT requires n_obs_steps=1 and image_size divisible by patch_size")
        self.horizon, self.n_action_steps, self.n_obs_steps = horizon, n_action_steps, n_obs_steps
        self.action_dim, self.action_mode = 7, "cartesian_delta"
        self.uses_pointcloud, self.requires_tcp_pos = True, False
        self.uses_mvt = True
        self.pointcloud_max_points = 131072
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
                               predictor_kwargs={"attn_with": "visual-featmap",
                                                 "upscale_ratio": patch_size,
                                                 "label_name": "target-heatmap"}),
            arp.TokenType.make(name="target-gripper", dim=1, is_continuous=True,
                               embedding="linear", predictor="gmm",
                               predictor_kwargs={"num_latents": num_latents,
                                                 "low_var_eval": True}),
        ]
        self.policy = arp.AutoRegressivePolicy(arp.ModelConfig(
            n_embd=hidden_dim, embd_pdrop=dropout, max_seq_len=6 + horizon * 7,
            max_chunk_size=7, layer_norm_every_block=False, tokens=tokens,
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

    def _training_sequence(self, batch: dict[str, Any], encoded: torch.Tensor):
        bsz = len(encoded); views, anchors = 2, 3
        current = self.renderer.project(self.renderer.to_cube(
            batch["obs"]["control_points"][:, -1])).reshape(bsz, anchors * views, 2)
        target = self.renderer.project(self.renderer.to_cube(batch["target_control_points"]))
        target = target.reshape(bsz, self.horizon, anchors * views, 2)
        gripper = batch["target_gripper"]
        values = [current]; ids = [torch.zeros((bsz, anchors * views, 1), device=current.device)]
        chunks = [torch.arange(anchors * views, dtype=torch.long, device=current.device)]
        for step in range(self.horizon):
            values.extend((target[:, step], gripper[:, step:step+1].reshape(bsz, 1, 1)))
            ids.extend((torch.ones((bsz, anchors * views, 1), device=current.device),
                        torch.full((bsz, 1, 1), 2, device=current.device)))
            chunks.append(torch.full((anchors * views + 1,), anchors * views + step,
                                     dtype=torch.long, device=current.device))
        vals = arp.cat_uneven_blc_tensors(*values)
        token_ids = torch.cat(ids, dim=1)
        tks = torch.cat((vals, token_ids), dim=-1)
        chk_ids = torch.cat(chunks)
        valid = torch.ones((bsz, tks.shape[1]), dtype=torch.bool, device=tks.device)
        action_valid = ~batch["action_is_pad"].bool()
        offset = anchors * views
        for step in range(self.horizon):
            valid[:, offset:offset + anchors * views + 1] = action_valid[:, step:step+1]
            offset += anchors * views + 1
        repeated = encoded[:, None, None].expand(-1, self.horizon, anchors, -1, -1, -1, -1)
        spatial_feat = repeated.reshape(bsz * self.horizon * anchors * views,
                                        self.hidden_dim, encoded.shape[-2], encoded.shape[-1])
        flat_target = target.reshape(-1, 2)
        axis = torch.arange(self.image_size, device=target.device, dtype=target.dtype)
        dx = axis[None, None, :] - flat_target[:, 0, None, None]
        dy = axis[None, :, None] - flat_target[:, 1, None, None]
        distance2 = dx.square() + dy.square()
        heatmap = torch.exp(-distance2 / (2 * 1.5 ** 2))
        heatmap = heatmap.masked_fill(distance2 > (3 * 1.5) ** 2, 0)
        heatmap = heatmap / heatmap.sum((-2, -1), keepdim=True).clamp_min(1e-6)
        return tks, chk_ids, valid, spatial_feat, heatmap

    def compute_loss(self, batch: dict[str, Any]):
        _, visual_tokens, encoded = self._visual(batch["obs"])
        tks, chunks, valid, spatial_feat, heatmap = self._training_sequence(batch, encoded)
        return self.policy.compute_loss(tks, chunks, valid_tk_mask=valid,
            contexts={"visual-featmap": spatial_feat, "visual-tokens": visual_tokens,
                      "target-heatmap": heatmap})

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
        repeated = encoded[:, None].expand(-1, anchors, -1, -1, -1, -1)
        repeated = repeated.reshape(bsz * anchors * views, self.hidden_dim,
                                    encoded.shape[-2], encoded.shape[-1])
        for step in range(self.horizon):
            chunk = anchors * views + step
            future.extend([{"tk_id": target_id, "chk_id": chunk}] * (anchors * views))
            future.append({"tk_id": grip_id, "chk_id": chunk})
            feature_context[str(chunk)] = repeated
        generated = self.policy.generate(prompt, future, sample=False,
            contexts={"visual-featmap": feature_context, "visual-tokens": visual_tokens})
        produced = generated[:, anchors * views:]
        pixels, grippers = [], []
        for step in range(self.horizon):
            block = produced[:, step * 7:(step + 1) * 7]
            pixels.append(block[:, :6, :2].reshape(bsz, anchors, views, 2))
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
            action[:, 6] = grippers[step] - previous_gripper
            actions.append(action)
            previous_origin, previous_rotation, previous_gripper = origins[:, step], rotations[:, step], grippers[step]
        prediction = torch.stack(actions, 1)
        return {"action_pred": prediction, "action": prediction[:, :self.n_action_steps],
                "target_control_points": control}
