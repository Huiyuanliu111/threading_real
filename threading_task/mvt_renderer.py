"""High-resolution orthographic multi-view rendering for Threading MVT.

The renderer follows the real-robot ARP representation: calibrated RGB-D is
fused in the robot base frame, normalized into a fixed cube, and rendered from
top and left virtual cameras.  It has no PyTorch3D dependency and preserves the
full 420x420 output used by the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OrthographicMVTRenderer(nn.Module):
    views = ("top", "left")

    def __init__(self, image_size: int = 420,
                 scene_bounds=(0.15, -0.40, -0.15, 0.75, 0.30, 0.50)) -> None:
        super().__init__()
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        bounds = torch.as_tensor(scene_bounds, dtype=torch.float32).reshape(2, 3)
        if torch.any(bounds[0] >= bounds[1]):
            raise ValueError("invalid scene bounds")
        self.image_size = int(image_size)
        self.register_buffer("scene_bounds", bounds)
        center = bounds.mean(0)
        scale = 2.0 / (bounds[1] - bounds[0]).max()
        self.register_buffer("cube_center", center)
        self.register_buffer("cube_scale", scale)

    def to_cube(self, points: torch.Tensor) -> torch.Tensor:
        return (points - self.cube_center) * self.cube_scale

    def from_cube(self, points: torch.Tensor) -> torch.Tensor:
        return points / self.cube_scale + self.cube_center

    def project(self, cube_points: torch.Tensor) -> torch.Tensor:
        """Project BxNx3 cube coordinates to BxNx2x2 pixel coordinates."""
        p = cube_points
        last = float(self.image_size - 1)
        top = torch.stack(((p[..., 0] + 1) * 0.5 * last,
                           (1 - p[..., 1]) * 0.5 * last), dim=-1)
        left = torch.stack(((p[..., 1] + 1) * 0.5 * last,
                            (1 - p[..., 2]) * 0.5 * last), dim=-1)
        return torch.stack((top, left), dim=-2)

    def unproject_pair(self, pixels: torch.Tensor) -> torch.Tensor:
        """Recover cube XYZ from paired top/left pixels (...,2,2)."""
        last = float(self.image_size - 1)
        top, left = pixels[..., 0, :], pixels[..., 1, :]
        x = top[..., 0] / last * 2 - 1
        y_top = 1 - top[..., 1] / last * 2
        y_left = left[..., 0] / last * 2 - 1
        z = 1 - left[..., 1] / last * 2
        return torch.stack((x, (y_top + y_left) * 0.5, z), dim=-1)

    @torch.no_grad()
    def forward(self, points: torch.Tensor, colors: torch.Tensor,
                valid_count: torch.Tensor | None = None) -> torch.Tensor:
        """Render BxNx3 points/RGB to Bx2x7xHxW MVT features."""
        if points.ndim != 3 or points.shape[-1] != 3 or colors.shape != points.shape:
            raise ValueError("points and colors must both have shape BxNx3")
        bsz, count, _ = points.shape
        cube = self.to_cube(points)
        in_cube = (cube.abs() <= 1).all(-1) & torch.isfinite(cube).all(-1)
        if valid_count is not None:
            rows = torch.arange(count, device=points.device)[None]
            in_cube &= rows < valid_count.reshape(-1, 1)
        pixels = self.project(cube).round().long()
        output = points.new_zeros((bsz, 2, 7, self.image_size, self.image_size))
        # Nearest-surface z-buffer. Top looks down +Z; left looks from +X.
        view_depth = torch.stack((-cube[..., 2], -cube[..., 0]), dim=1)
        feature = torch.cat((cube, colors * 2 - 1), dim=-1)
        cells = self.image_size * self.image_size
        for batch in range(bsz):
            for view in range(2):
                keep = in_cube[batch]
                xy = pixels[batch, :, view]
                keep &= (xy >= 0).all(-1) & (xy < self.image_size).all(-1)
                if not torch.any(keep):
                    continue
                xy = xy[keep]
                flat = xy[:, 1] * self.image_size + xy[:, 0]
                depth = view_depth[batch, view, keep]
                nearest = torch.full((cells,), torch.inf, device=points.device, dtype=points.dtype)
                nearest.scatter_reduce_(0, flat, depth, reduce="amin", include_self=True)
                surface = depth <= nearest[flat] + 1e-4
                flat, depth = flat[surface], depth[surface]
                feat = feature[batch, keep][surface]
                sums = points.new_zeros((6, cells))
                hits = points.new_zeros((cells,))
                sums.scatter_add_(1, flat[None].expand(6, -1), feat.T)
                hits.scatter_add_(0, flat, torch.ones_like(depth))
                occupied = hits > 0
                image = output[batch, view].flatten(1)
                image[:6, occupied] = sums[:, occupied] / hits[occupied]
                if occupied.any():
                    centered = nearest[occupied] - nearest[occupied].mean()
                    image[6, occupied] = centered
        return output


def pixel_location_channels(num_views: int, image_size: int, *, device=None,
                            dtype=torch.float32) -> torch.Tensor:
    result = torch.zeros((num_views, 3, image_size, image_size), device=device, dtype=dtype)
    result[:, 0] = torch.linspace(-1, 1, num_views, device=device, dtype=dtype)[:, None, None]
    result[:, 1] = torch.linspace(-1, 1, image_size, device=device, dtype=dtype)[None, :, None]
    result[:, 2] = torch.linspace(-1, 1, image_size, device=device, dtype=dtype)[None, None, :]
    return result
