"""Validated adapter around the unmodified upstream AutoHorizon functions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .official import bidir_soft_pointer, pick_horizon_softpointer

UPSTREAM_COMMIT = "c7504f1"
IMPLEMENTATION_VERSION = "official_c7504f1_arp_full_prediction_v1"


@dataclass(frozen=True)
class AutoHorizonConfig:
    hold_thr: float = 0.3
    max_entropy_q: float = 0.9
    run_len: int = 1
    method: Literal["bidirectional", "forward"] = "bidirectional"

    def __post_init__(self):
        if not math.isfinite(self.hold_thr) or self.hold_thr < 0:
            raise ValueError("hold_thr must be finite and non-negative")
        if not 0 <= self.max_entropy_q <= 1:
            raise ValueError("max_entropy_q must lie in [0, 1]")
        if not isinstance(self.run_len, int) or self.run_len < 1:
            raise ValueError("run_len must be a positive integer")
        if self.method not in ("bidirectional", "forward"):
            raise ValueError("method must be bidirectional or forward")
        if self.method == "forward" and self.run_len != 1:
            raise ValueError("upstream forward method only supports run_len=1")


@dataclass
class AutoHorizonResult:
    h_star: int
    raw_horizon: int
    attention: Tensor
    diagnostics: dict
    method: str
    actions: Tensor | None = None
    generated_action_tokens: int = 0
    prediction_offset: int = 0

    def metrics(self) -> dict:
        def serializable(value):
            return value.detach().cpu().tolist() if isinstance(value, Tensor) else value
        return {
            "h_star": self.h_star,
            "raw_horizon": self.raw_horizon,
            "method": self.method,
            "generated_action_tokens": self.generated_action_tokens,
            "prediction_offset": self.prediction_offset,
            "implementation_version": IMPLEMENTATION_VERSION,
            **{key: serializable(value) for key, value in self.diagnostics.items()},
        }


@torch.no_grad()
def select_horizon(
    attention: Tensor, config: AutoHorizonConfig | None = None,
) -> AutoHorizonResult:
    """Average [P,P], [heads,P,P], or [layers,heads,P,P], then call upstream.

    No batch axis is accepted. Do not pre-normalize: upstream performs its own
    normalization (twice in bidirectional mode), including its epsilon terms.
    Invalid/singleton inputs are rejected instead of changing upstream behavior.
    """
    config = config or AutoHorizonConfig()
    if attention.ndim not in (2, 3, 4):
        raise ValueError("attention must have 2, 3 or 4 dimensions (no batch axis)")
    p = attention.shape[-1]
    if p < 2 or attention.shape[-2] != p or any(n == 0 for n in attention.shape):
        raise ValueError("official AutoHorizon requires a nonempty square map with P >= 2")
    # Upstream conv1d(...).squeeze() yields a scalar if the window covers P.
    if config.method == "bidirectional" and config.run_len >= p:
        raise ValueError("upstream bidirectional implementation requires run_len < P")
    a = attention.detach()
    if not a.is_floating_point() or not torch.isfinite(a).all() or (a < 0).any():
        raise ValueError("attention must contain finite non-negative floating weights")
    if a.ndim > 2:
        a = a.mean(dim=tuple(range(a.ndim - 2)))
    if (a.sum(-1) <= 0).any():
        raise ValueError("each attention row must have positive mass")
    kwargs = dict(hold_thr=config.hold_thr, max_entropy_q=config.max_entropy_q)
    if config.method == "bidirectional":
        horizon, diagnostics = bidir_soft_pointer(a, run_len=config.run_len, **kwargs)
    else:
        horizon, diagnostics = pick_horizon_softpointer(a, **kwargs)
    raw = int(horizon.item())
    return AutoHorizonResult(raw, raw, a, diagnostics, config.method)
