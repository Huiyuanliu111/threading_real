"""Projection and triangulation primitives for spatial robot policies."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def load_projection_calibration(
    path: str | Path,
    camera_keys: Sequence[str],
    output_width: int,
    output_height: int,
    max_reprojection_rmse_px: float = 4.0,
) -> np.ndarray:
    """Load and resize base-to-pixel 3x4 projection matrices.

    The calibration JSON stores matrices in the resolution at which points
    were annotated. Rows 0 and 1 are scaled when the policy uses resized input.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"spatial camera calibration not found: {path}. Run "
            "scripts/calibration/fit.py first."
        )
    payload = json.loads(path.read_text())
    cameras = payload.get("cameras", {})
    matrices = []
    for key in camera_keys:
        if key not in cameras:
            raise KeyError(f"{path}: missing calibration for {key!r}")
        item = cameras[key]
        measured_rmse = item.get("reprojection_rmse_px")
        if measured_rmse is not None and float(measured_rmse) > max_reprojection_rmse_px:
            raise ValueError(
                f"{path}: {key} calibration RMSE {float(measured_rmse):.2f}px exceeds "
                f"the allowed {max_reprojection_rmse_px:.2f}px"
            )
        matrix = np.asarray(item["projection_matrix"], dtype=np.float64)
        if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{path}: {key} projection_matrix must be finite 3x4")
        source_width = int(item["image_width"])
        source_height = int(item["image_height"])
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"{path}: invalid image size for {key}")
        matrix = matrix.copy()
        matrix[0] *= output_width / source_width
        matrix[1] *= output_height / source_height
        matrices.append(matrix)
    return np.stack(matrices).astype(np.float32)


def project_points_numpy(
    points_xyz: np.ndarray,
    projection_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project N base-frame points into C cameras, returning [N,C,2]."""
    points = np.asarray(points_xyz, dtype=np.float64)
    matrices = np.asarray(projection_matrices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected Nx3 points, got {points.shape}")
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 4):
        raise ValueError(f"expected Cx3x4 projection matrices, got {matrices.shape}")
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    projected = np.einsum("cij,nj->nci", matrices, homogeneous)
    depth = projected[..., 2]
    pixels = projected[..., :2] / np.where(np.abs(depth[..., None]) > 1e-9, depth[..., None], np.nan)
    return pixels.astype(np.float32), depth.astype(np.float32)


def triangulate_dlt(
    pixels: torch.Tensor,
    projection_matrices: torch.Tensor,
) -> torch.Tensor:
    """Triangulate corresponding C-view pixels using homogeneous DLT.

    ``pixels`` is [B,C,2], matrices are [C,3,4]. The function is differentiable,
    although deployment normally calls it on hard heatmap maxima.
    """
    if pixels.ndim != 3 or pixels.shape[-1] != 2:
        raise ValueError(f"expected pixels [B,C,2], got {tuple(pixels.shape)}")
    if projection_matrices.shape != (pixels.shape[1], 3, 4):
        raise ValueError(
            f"expected matrices {(pixels.shape[1], 3, 4)}, got {tuple(projection_matrices.shape)}"
        )
    p = projection_matrices.unsqueeze(0).expand(pixels.shape[0], -1, -1, -1)
    u, v = pixels[..., 0, None], pixels[..., 1, None]
    rows_u = u * p[:, :, 2] - p[:, :, 0]
    rows_v = v * p[:, :, 2] - p[:, :, 1]
    system = torch.stack((rows_u, rows_v), dim=2).flatten(1, 2)
    _, _, vh = torch.linalg.svd(system, full_matrices=False)
    homogeneous = vh[:, -1]
    scale = homogeneous[:, 3:4]
    safe_scale = torch.where(scale.abs() > 1e-8, scale, torch.full_like(scale, 1e-8))
    return homogeneous[:, :3] / safe_scale


def spatial_argmax(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hard pixel coordinates and peak probability for [B,C,H,W]."""
    if logits.ndim != 4:
        raise ValueError(f"expected [B,C,H,W] logits, got {tuple(logits.shape)}")
    batch, cameras, height, width = logits.shape
    probabilities = logits.flatten(2).softmax(dim=-1)
    confidence, index = probabilities.max(dim=-1)
    pixels = torch.stack((index % width, index // width), dim=-1).to(logits.dtype)
    return pixels.reshape(batch, cameras, 2), confidence


def spatial_softargmax(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sub-pixel expectations and peak confidence for [B,C,H,W]."""
    if logits.ndim != 4:
        raise ValueError(f"expected [B,C,H,W] logits, got {tuple(logits.shape)}")
    _, _, height, width = logits.shape
    probabilities = logits.flatten(2).softmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    ys = torch.arange(height, device=logits.device, dtype=logits.dtype)
    xs = torch.arange(width, device=logits.device, dtype=logits.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack(
        (
            (probabilities * xx.flatten()).sum(dim=-1),
            (probabilities * yy.flatten()).sum(dim=-1),
        ),
        dim=-1,
    )
    return pixels, confidence


def gaussian_heatmaps(
    pixels: torch.Tensor,
    height: int,
    width: int,
    sigma: float,
) -> torch.Tensor:
    """Create normalized Gaussian target distributions for [B,C,2] pixels."""
    ys = torch.arange(height, device=pixels.device, dtype=pixels.dtype)
    xs = torch.arange(width, device=pixels.device, dtype=pixels.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    squared = (
        (xx[None, None] - pixels[..., 0, None, None]).square()
        + (yy[None, None] - pixels[..., 1, None, None]).square()
    )
    heatmaps = torch.exp(-0.5 * squared / float(sigma) ** 2)
    return heatmaps / heatmaps.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
