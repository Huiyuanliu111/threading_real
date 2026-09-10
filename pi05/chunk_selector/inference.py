"""Shared-vision pi0.5 inference with adaptive action prediction horizons."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterator, Literal

import torch
import torch.nn.functional as F
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy

from chunk_selector.chunk_selector import ChunkSelector


PredictionMode = Literal["required_only", "full_then_truncate"]


@dataclass(frozen=True)
class AdaptivePrediction:
    actions: torch.Tensor
    execution_chunk: int
    predicted_chunk: int
    continuous_chunk: float
    probabilities: tuple[float, ...]
    vision_seconds: float
    selector_seconds: float
    action_seconds: float
    total_seconds: float


def _pool_visual_tokens(tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
    batch_size, token_count, feature_dim = tokens.shape
    source_grid = int(round(token_count**0.5))
    if source_grid * source_grid != token_count:
        raise ValueError(f"Expected square visual token grid, got {token_count} tokens")
    if not 1 <= grid_size <= source_grid:
        raise ValueError(f"pool grid must lie in [1, {source_grid}], got {grid_size}")
    maps = tokens.transpose(1, 2).reshape(batch_size, feature_dim, source_grid, source_grid)
    return F.adaptive_avg_pool2d(maps, (grid_size, grid_size)).flatten(2).transpose(1, 2)


class PI05SelectorInference:
    """Load frozen pi0.5 and a sidecar selector for the two prediction modes."""

    def __init__(
        self,
        checkpoint: str | Path,
        selector_dir: str | Path,
        *,
        device: str = "cuda",
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.selector_dir = Path(selector_dir).expanduser().resolve()
        self.device = torch.device(device)
        self.policy = PI05Policy.from_pretrained(self.checkpoint).to(self.device).eval()
        self.policy.requires_grad_(False)
        self.selector = ChunkSelector.from_pretrained(
            self.selector_dir, device=self.device
        ).eval()
        self.selector.requires_grad_(False)
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.full_horizon = int(self.policy.config.chunk_size)
        if self.full_horizon != 10:
            raise ValueError(f"Expected a 10-step pi0.5 checkpoint, got {self.full_horizon}")
        action_feature = self.policy.config.output_features["action"]
        if int(action_feature.shape[0]) != 7:
            raise ValueError(f"Expected 7D Cartesian actions, got {action_feature.shape}")
        if tuple(self.selector.candidate_chunks) != (4, 10):
            raise ValueError(
                f"Expected selector candidates (4, 10), got {self.selector.candidate_chunks}"
            )
        if self.selector.config.selection_mode != "expected":
            raise ValueError("Selector checkpoint must use selection_mode='expected'")
        if self.policy.config.use_visual_memory or self.policy.config.use_proprioceptive_memory:
            raise ValueError("Adaptive prediction comparison currently requires pi0.5 memory to be disabled")
        image_features = list(self.policy.config.image_features)
        if len(image_features) != 2:
            raise ValueError(f"Expected exactly two pi0.5 image features, got {image_features}")
        if self.selector.config.num_cameras != 2:
            raise ValueError("Selector must be trained with two camera embeddings")
        tokens_per_camera = self.selector.config.max_tokens // 2
        pool_grid = int(round(tokens_per_camera**0.5))
        if 2 * pool_grid * pool_grid != self.selector.config.max_tokens:
            raise ValueError(
                "Selector max_tokens must contain two equal square camera grids"
            )
        self.pool_grid = pool_grid

    def reset(self) -> None:
        self.policy.reset()

    def prepare(self, frame: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.preprocess(frame)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def _runtime_prediction_horizon(self, horizon: int) -> Iterator[None]:
        if not 1 <= horizon <= self.full_horizon:
            raise ValueError(f"prediction horizon must lie in [1, {self.full_horizon}]")
        original_policy_horizon = int(self.policy.config.chunk_size)
        original_model_horizon = int(self.policy.model.config.chunk_size)
        self.policy.config.chunk_size = horizon
        self.policy.model.config.chunk_size = horizon
        try:
            yield
        finally:
            self.policy.config.chunk_size = original_policy_horizon
            self.policy.model.config.chunk_size = original_model_horizon

    @contextmanager
    def _reuse_visual_embeddings(
        self, embeddings: list[torch.Tensor]
    ) -> Iterator[None]:
        image_encoder = self.policy.model.paligemma_with_expert
        original = image_encoder.embed_image
        queue = iter(embeddings)

        def cached_embed_image(_image: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
            try:
                return next(queue)
            except StopIteration as exc:
                raise RuntimeError("pi0.5 requested more cached camera embeddings than expected") from exc

        image_encoder.embed_image = cached_embed_image
        try:
            yield
            try:
                next(queue)
            except StopIteration:
                pass
            else:
                raise RuntimeError("pi0.5 did not consume every cached camera embedding")
        finally:
            image_encoder.embed_image = original

    @torch.inference_mode()
    def predict(
        self,
        batch: dict[str, torch.Tensor],
        *,
        mode: PredictionMode,
    ) -> AdaptivePrediction:
        if mode not in {"required_only", "full_then_truncate"}:
            raise ValueError(f"Unknown prediction mode: {mode}")
        tensor_values = [value for value in batch.values() if isinstance(value, torch.Tensor)]
        if not tensor_values:
            raise ValueError("Preprocessed pi0.5 batch contains no tensors")
        if tensor_values[0].shape[0] != 1:
            raise ValueError("Adaptive deployment inference currently requires batch size one")

        total_start = time.perf_counter()
        images, image_masks = self.policy._preprocess_images(batch)
        if any(mask.ndim != 1 or not bool(mask.all()) for mask in image_masks):
            raise ValueError("Adaptive selector requires both current camera images")

        self._synchronize()
        vision_start = time.perf_counter()
        visual_embeddings = [
            self.policy.model.paligemma_with_expert.embed_image(image) for image in images
        ]
        self._synchronize()
        vision_seconds = time.perf_counter() - vision_start

        pooled = [_pool_visual_tokens(tokens, self.pool_grid) for tokens in visual_embeddings]
        selector_features = torch.cat(pooled, dim=1).float()
        tokens_per_camera = self.pool_grid * self.pool_grid
        camera_ids = torch.arange(2, device=self.device).repeat_interleave(tokens_per_camera)
        spatial_ids = torch.arange(tokens_per_camera, device=self.device).repeat(2)
        self._synchronize()
        selector_start = time.perf_counter()
        selection = self.selector.select(
            selector_features,
            camera_ids=camera_ids,
            spatial_ids=spatial_ids,
        )
        self._synchronize()
        selector_seconds = time.perf_counter() - selector_start
        execution_chunk = int(selection.chunk_sizes.item())
        prediction_chunk = execution_chunk if mode == "required_only" else self.full_horizon

        self._synchronize()
        action_start = time.perf_counter()
        with self._runtime_prediction_horizon(prediction_chunk):
            with self._reuse_visual_embeddings(visual_embeddings):
                normalized_actions = self.policy.predict_action_chunk(batch)
        actions = self.postprocess(normalized_actions)
        actions = actions.clone()
        actions[..., 6] = 0.0
        self._synchronize()
        action_seconds = time.perf_counter() - action_start
        if actions.shape[1] != prediction_chunk:
            raise RuntimeError(
                f"pi0.5 returned {actions.shape[1]} actions for horizon {prediction_chunk}"
            )
        actions = actions[:, :execution_chunk]
        total_seconds = time.perf_counter() - total_start
        return AdaptivePrediction(
            actions=actions,
            execution_chunk=execution_chunk,
            predicted_chunk=prediction_chunk,
            continuous_chunk=float(selection.continuous_chunk_sizes.item()),
            probabilities=tuple(float(value) for value in selection.probabilities[0].tolist()),
            vision_seconds=vision_seconds,
            selector_seconds=selector_seconds,
            action_seconds=action_seconds,
            total_seconds=total_seconds,
        )
