"""Transformer classifier for choosing an execution chunk size.

The selector is intentionally independent from an action policy.  Action policies
remain responsible for predicting their full, training-time action horizon; this
module only maps already-computed visual tokens to one of a small set of legal
execution chunks.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ChunkSelectorConfig:
    """Serializable architecture and deployment configuration."""

    input_dim: int
    candidate_chunks: tuple[int, ...]
    d_model: int = 256
    num_layers: int = 2
    n_heads: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_tokens: int = 256
    num_cameras: int = 0
    max_time_steps: int = 0
    max_spatial_positions: int = 0
    safe_chunk: int | None = None
    confidence_threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidates = tuple(int(value) for value in self.candidate_chunks)
        object.__setattr__(self, "candidate_chunks", candidates)
        if self.input_dim <= 0 or self.d_model <= 0:
            raise ValueError("input_dim and d_model must be positive")
        if self.num_layers <= 0 or self.n_heads <= 0 or self.dim_feedforward <= 0:
            raise ValueError("Transformer depth, heads, and feedforward width must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not candidates or any(value <= 0 for value in candidates):
            raise ValueError("candidate_chunks must contain positive integers")
        if tuple(sorted(set(candidates))) != candidates:
            raise ValueError("candidate_chunks must be unique and strictly increasing")
        if self.safe_chunk is not None and self.safe_chunk not in candidates:
            raise ValueError("safe_chunk must be one of candidate_chunks")
        if self.confidence_threshold is not None and not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1]")
        for name in ("num_cameras", "max_time_steps", "max_spatial_positions"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkSelectorConfig":
        data = dict(payload)
        data["candidate_chunks"] = tuple(data["candidate_chunks"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_chunks"] = list(self.candidate_chunks)
        return payload


@dataclass
class ChunkSelection:
    """Selector outputs in both class and execution-chunk space."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    class_ids: torch.Tensor
    chunk_sizes: torch.Tensor
    confidences: torch.Tensor
    used_safe_fallback: torch.Tensor


class ChunkSelector(nn.Module):
    """A small Transformer encoder over shared visual tokens."""

    CONFIG_NAME = "chunk_selector_config.json"
    WEIGHTS_NAME = "chunk_selector.safetensors"

    def __init__(self, config: ChunkSelectorConfig):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.position_embedding = nn.Embedding(config.max_tokens + 1, config.d_model)
        self.camera_embedding = (
            nn.Embedding(config.num_cameras, config.d_model) if config.num_cameras else None
        )
        self.time_embedding = (
            nn.Embedding(config.max_time_steps, config.d_model)
            if config.max_time_steps
            else None
        )
        self.spatial_embedding = (
            nn.Embedding(config.max_spatial_positions, config.d_model)
            if config.max_spatial_positions
            else None
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, len(config.candidate_chunks)),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        for embedding in (
            self.camera_embedding,
            self.time_embedding,
            self.spatial_embedding,
        ):
            if embedding is not None:
                nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    @property
    def candidate_chunks(self) -> tuple[int, ...]:
        return self.config.candidate_chunks

    def validate_for_policy(self, *, feature_dim: int, max_chunk: int) -> None:
        """Fail early when a sidecar selector does not match its action policy."""
        if self.config.input_dim != int(feature_dim):
            raise ValueError(
                f"Selector input_dim={self.config.input_dim}, policy feature_dim={feature_dim}"
            )
        invalid = [value for value in self.candidate_chunks if value > int(max_chunk)]
        if invalid:
            raise ValueError(
                f"Selector chunks {invalid} exceed policy maximum executable chunk {max_chunk}"
            )

    @staticmethod
    def _validate_ids(
        name: str,
        ids: torch.Tensor | None,
        *,
        batch_size: int,
        token_count: int,
        upper_bound: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if upper_bound == 0:
            if ids is not None:
                raise ValueError(f"{name} ids were provided but its embedding is disabled")
            return None
        if ids is None:
            raise ValueError(f"{name} ids are required by this selector configuration")
        ids = torch.as_tensor(ids, device=device, dtype=torch.long)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0).expand(batch_size, -1)
        if tuple(ids.shape) != (batch_size, token_count):
            raise ValueError(
                f"{name} ids must have shape {(batch_size, token_count)}, got {tuple(ids.shape)}"
            )
        if ids.numel() and (ids.min() < 0 or ids.max() >= upper_bound):
            raise ValueError(f"{name} ids must lie in [0, {upper_bound - 1}]")
        return ids

    def forward(
        self,
        features: torch.Tensor,
        *,
        camera_ids: torch.Tensor | None = None,
        time_ids: torch.Tensor | None = None,
        spatial_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected visual tokens [B, L, D], got {tuple(features.shape)}")
        batch_size, token_count, feature_dim = features.shape
        if feature_dim != self.config.input_dim:
            raise ValueError(
                f"Expected feature dim {self.config.input_dim}, got {feature_dim}"
            )
        if token_count <= 0 or token_count > self.config.max_tokens:
            raise ValueError(
                f"Token count must lie in [1, {self.config.max_tokens}], got {token_count}"
            )
        device = features.device
        camera_ids = self._validate_ids(
            "camera",
            camera_ids,
            batch_size=batch_size,
            token_count=token_count,
            upper_bound=self.config.num_cameras,
            device=device,
        )
        time_ids = self._validate_ids(
            "time",
            time_ids,
            batch_size=batch_size,
            token_count=token_count,
            upper_bound=self.config.max_time_steps,
            device=device,
        )
        spatial_ids = self._validate_ids(
            "spatial",
            spatial_ids,
            batch_size=batch_size,
            token_count=token_count,
            upper_bound=self.config.max_spatial_positions,
            device=device,
        )

        tokens = self.input_projection(self.input_norm(features))
        position_ids = torch.arange(token_count + 1, device=device)
        cls = self.cls_token.expand(batch_size, -1, -1)
        cls = cls + self.position_embedding(position_ids[:1]).unsqueeze(0)
        tokens = tokens + self.position_embedding(position_ids[1:]).unsqueeze(0)
        if camera_ids is not None:
            tokens = tokens + self.camera_embedding(camera_ids)
        if time_ids is not None:
            tokens = tokens + self.time_embedding(time_ids)
        if spatial_ids is not None:
            tokens = tokens + self.spatial_embedding(spatial_ids)
        tokens = torch.cat([cls, tokens], dim=1)

        transformer_padding_mask = None
        if padding_mask is not None:
            padding_mask = torch.as_tensor(padding_mask, device=device, dtype=torch.bool)
            if tuple(padding_mask.shape) != (batch_size, token_count):
                raise ValueError(
                    f"padding_mask must have shape {(batch_size, token_count)}, "
                    f"got {tuple(padding_mask.shape)}"
                )
            cls_mask = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
            transformer_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        encoded = self.transformer(
            tokens,
            src_key_padding_mask=transformer_padding_mask,
        )
        return self.head(self.output_norm(encoded[:, 0]))

    def select(
        self,
        features: torch.Tensor,
        *,
        camera_ids: torch.Tensor | None = None,
        time_ids: torch.Tensor | None = None,
        spatial_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        confidence_threshold: float | None = None,
        safe_chunk: int | None = None,
    ) -> ChunkSelection:
        logits = self(
            features,
            camera_ids=camera_ids,
            time_ids=time_ids,
            spatial_ids=spatial_ids,
            padding_mask=padding_mask,
        )
        probabilities = torch.softmax(logits, dim=-1)
        confidences, class_ids = probabilities.max(dim=-1)
        candidates = torch.as_tensor(
            self.candidate_chunks,
            device=logits.device,
            dtype=torch.long,
        )
        chunk_sizes = candidates[class_ids]

        threshold = (
            self.config.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        fallback = self.config.safe_chunk if safe_chunk is None else safe_chunk
        used_safe_fallback = torch.zeros_like(class_ids, dtype=torch.bool)
        if threshold is not None:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("confidence_threshold must lie in [0, 1]")
            if fallback is None:
                raise ValueError("safe_chunk is required when confidence fallback is enabled")
            if fallback not in self.candidate_chunks:
                raise ValueError("safe_chunk must be one of candidate_chunks")
            used_safe_fallback = confidences < threshold
            chunk_sizes = torch.where(
                used_safe_fallback,
                torch.full_like(chunk_sizes, fallback),
                chunk_sizes,
            )

        return ChunkSelection(
            logits=logits,
            probabilities=probabilities,
            class_ids=class_ids,
            chunk_sizes=chunk_sizes,
            confidences=confidences,
            used_safe_fallback=used_safe_fallback,
        )

    def save_pretrained(
        self,
        output_dir: str | Path,
        *,
        metadata: dict[str, str] | None = None,
    ) -> Path:
        """Save a sidecar selector without modifying an action checkpoint."""
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError("Saving ChunkSelector requires safetensors") from exc

        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        config_path = output_path / self.CONFIG_NAME
        config_path.write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.state_dict().items()
        }
        weights_path = output_path / self.WEIGHTS_NAME
        save_file(state, str(weights_path), metadata=metadata)
        return output_path

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "ChunkSelector":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading ChunkSelector requires safetensors") from exc

        model_path = Path(model_dir).expanduser().resolve()
        config = ChunkSelectorConfig.from_dict(
            json.loads((model_path / cls.CONFIG_NAME).read_text(encoding="utf-8"))
        )
        selector = cls(config)
        state = load_file(str(model_path / cls.WEIGHTS_NAME), device=str(device))
        selector.load_state_dict(state, strict=True)
        selector.to(device)
        return selector


def flatten_camera_feature_maps(
    feature_maps: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten `[B, D, H, W]` maps and generate camera/spatial ids."""
    if not feature_maps:
        raise ValueError("At least one camera feature map is required")
    batch_size = feature_maps[0].shape[0]
    feature_dim = feature_maps[0].shape[1]
    tokens: list[torch.Tensor] = []
    camera_ids: list[torch.Tensor] = []
    spatial_ids: list[torch.Tensor] = []
    for camera_index, feature in enumerate(feature_maps):
        if feature.ndim != 4:
            raise ValueError(f"Expected [B, D, H, W], got {tuple(feature.shape)}")
        if feature.shape[:2] != (batch_size, feature_dim):
            raise ValueError("Camera feature maps have inconsistent batch/feature dimensions")
        num_spatial = int(feature.shape[-2] * feature.shape[-1])
        tokens.append(feature.flatten(2).transpose(1, 2))
        camera_ids.append(
            torch.full(
                (batch_size, num_spatial),
                camera_index,
                device=feature.device,
                dtype=torch.long,
            )
        )
        spatial_ids.append(
            torch.arange(num_spatial, device=feature.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
    return (
        torch.cat(tokens, dim=1),
        torch.cat(camera_ids, dim=1),
        torch.cat(spatial_ids, dim=1),
    )
