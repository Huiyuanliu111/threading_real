"""Inference-only AAC, arXiv:2604.04161, equations (2)--(10).

Core inputs use [dx, dy, dz, rx, ry, rz, binary_gripper]. Rotations are
axis-angle offsets in radians, composed in the base frame. No learned selector.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


AAC_IMPLEMENTATION_VERSION = "paper_candidate0_v7_threading_execution"


@dataclass(frozen=True)
class AACConfig:
    num_samples: int = 20
    alpha: float = 3.0
    covariance_eps: float = 1e-6
    execution_candidate_index: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.num_samples, bool) or not isinstance(self.num_samples, int) or self.num_samples < 2:
            raise ValueError("num_samples must be an integer >= 2")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if not math.isfinite(self.covariance_eps) or self.covariance_eps <= 0:
            raise ValueError("covariance_eps must be finite and positive")
        if (
            isinstance(self.execution_candidate_index, bool)
            or not isinstance(self.execution_candidate_index, int)
            or not 0 <= self.execution_candidate_index < self.num_samples
        ):
            raise ValueError("execution_candidate_index must be an integer in [0, num_samples)")


@dataclass(frozen=True)
class AACResult:
    h_star: int
    h_entropy: int
    xi: int
    magnitude_threshold_reached: bool
    entropy_translation: Tensor
    entropy_rotation: Tensor
    entropy_gripper: Tensor
    entropy_total: Tensor
    entropy_prefix_mean: Tensor
    entropy_delta: Tensor
    translation_magnitude: Tensor
    rotation_magnitude: Tensor
    gripper_magnitude: Tensor
    action_magnitude: Tensor

    def metrics(self) -> dict:
        return {
            key: value.detach().cpu().tolist() if isinstance(value, Tensor) else value
            for key, value in vars(self).items()
        }


def _finite_float(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value.to(dtype=torch.float64 if value.dtype == torch.float64 else torch.float32)


def gaussian_differential_entropy(samples: Tensor, eps: float = 1e-6) -> Tensor:
    """Eq. (3), unbiased full covariance per step, with diagonal regularization."""
    x = _finite_float(samples, "samples")
    if x.ndim != 3 or x.shape[0] < 2 or min(x.shape[1:]) < 1:
        raise ValueError("samples must have shape [N>=2,H>=1,D>=1]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    centered = x - x.mean(dim=0, keepdim=True)
    covariance = torch.einsum("nhd,nhe->hde", centered, centered) / (x.shape[0] - 1)
    covariance = covariance + eps * torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)
    sign, logdet = torch.linalg.slogdet(covariance)
    if not bool(((sign > 0) & torch.isfinite(logdet)).all()):
        raise ValueError("covariance is not numerically positive definite; check units or eps")
    return 0.5 * (x.shape[-1] * math.log(2 * math.pi * math.e) + logdet)


def binary_discrete_entropy(states: Tensor) -> Tensor:
    """Eq. (2), exact zero entropy at p=0 and p=1; natural logarithm."""
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 1:
        raise ValueError("gripper states must have shape [N>=2,H>=1]")
    if not bool(((states == 0) | (states == 1)).all()):
        raise ValueError("gripper states must be binary, not widths or width deltas")
    p = states.to(dtype=torch.float64 if states.dtype == torch.float64 else torch.float32).mean(0)
    return -torch.special.xlogy(p, p) - torch.special.xlogy(1 - p, 1 - p)


def compute_entropy_boundary(prefix_entropy: Tensor) -> tuple[int, Tensor]:
    """Eq. (5): diff index zero corresponds to prefix LENGTH one, not two."""
    prefix_entropy = _finite_float(prefix_entropy, "prefix_entropy")
    if prefix_entropy.ndim != 1 or prefix_entropy.numel() == 0:
        raise ValueError("prefix_entropy must be a nonempty vector")
    delta = prefix_entropy[1:] - prefix_entropy[:-1]
    return (int(delta.argmax().item()) + 1 if delta.numel() else 1), delta


def rotation_prefix_magnitude(rotations: Tensor) -> Tensor:
    """Eq. (9): norm of composed axis-angle, not norm of a unit quaternion.

    Matches the authors' delta_q * q_total order and [0, 2*pi] angle
    convention. Quaternion signs are not canonicalized between steps.
    """
    rotations = _finite_float(rotations, "rotations")
    if rotations.ndim != 2 or rotations.shape[1] != 3 or rotations.shape[0] == 0:
        raise ValueError("rotations must have shape [H>=1,3]")
    q = rotations.new_tensor([0., 0., 0., 1.])
    result = []
    for r in rotations:
        angle = torch.linalg.vector_norm(r)
        delta_xyz = r * (0.5 * torch.sinc(angle / (2 * math.pi)))
        delta_w = torch.cos(angle / 2)
        xyz = delta_w * q[:3] + q[3] * delta_xyz + torch.linalg.cross(delta_xyz, q[:3])
        w = delta_w * q[3] - torch.dot(delta_xyz, q[:3])
        q = torch.cat((xyz, w.reshape(1)))
        q = q / torch.linalg.vector_norm(q)
        norm_xyz = torch.linalg.vector_norm(q[:3])
        result.append(torch.where(norm_xyz > 1e-8, 2 * torch.atan2(norm_xyz, q[3]), norm_xyz * 0))
    return torch.stack(result)


@torch.no_grad()
def select_chunk_size(
    candidates: Tensor,
    *,
    nominal: Tensor | None = None,
    entropy_candidates: Tensor | None = None,
    config: AACConfig | None = None,
) -> AACResult:
    """Select an execution prefix from full-horizon Cartesian delta samples.

    candidates: [N,H,7], physical deltas and binary gripper states.
    nominal: [H,7], trajectory used for motion constraint (default configured candidate).
    entropy_candidates: optional [N,H,7] normalized policy actions. Only its
        first six coordinates are used; gripper entropy always uses candidates.

    No threshold crossing falls back to H (authors' implementation convention).
    There is no extra minimum length, maximum cap, or gripper movement weight.
    """
    cfg = config or AACConfig()
    candidates = _finite_float(candidates, "candidates")
    if candidates.ndim != 3 or candidates.shape[0] < 2 or candidates.shape[1] < 1 or candidates.shape[2] != 7:
        raise ValueError("candidates must have shape [N>=2,H>=1,7]")
    if cfg.execution_candidate_index >= candidates.shape[0]:
        raise ValueError("execution_candidate_index must be smaller than the actual sample count")
    nominal = (
        candidates[cfg.execution_candidate_index]
        if nominal is None else _finite_float(nominal, "nominal")
    )
    if nominal.shape != candidates.shape[1:] or nominal.device != candidates.device:
        raise ValueError("nominal must have shape [H,7] on the candidates device")
    if not bool(((nominal[:, 6] == 0) | (nominal[:, 6] == 1)).all()):
        raise ValueError("nominal gripper states must be binary")
    source = candidates if entropy_candidates is None else _finite_float(entropy_candidates, "entropy_candidates")
    if source.shape != candidates.shape or source.device != candidates.device:
        raise ValueError("entropy_candidates must match candidates shape and device")
    et = gaussian_differential_entropy(source[..., :3], cfg.covariance_eps)
    er = gaussian_differential_entropy(source[..., 3:6], cfg.covariance_eps)
    eg = binary_discrete_entropy(candidates[..., 6])
    entropy = et + er + eg
    horizon = candidates.shape[1]
    prefix = entropy.cumsum(0) / torch.arange(1, horizon + 1, device=entropy.device)
    h_entropy, delta = compute_entropy_boundary(prefix)
    mt = torch.linalg.vector_norm(nominal[:, :3].cumsum(0), dim=-1)
    mr = rotation_prefix_magnitude(nominal[:, 3:6])
    switches = torch.cat((nominal.new_zeros(1), (nominal[1:, 6] != nominal[:-1, 6]).to(nominal.dtype)))
    mg = switches.cumsum(0).clamp(max=1)
    magnitude = mt + mr + mg
    crossing = torch.nonzero(magnitude > cfg.alpha)
    reached = crossing.numel() > 0
    xi = int(crossing[0, 0].item()) + 1 if reached else horizon
    return AACResult(max(h_entropy, xi), h_entropy, xi, reached, et, er, eg,
                     entropy, prefix, delta, mt, mr, mg, magnitude)
