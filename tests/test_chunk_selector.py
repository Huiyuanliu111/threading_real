from __future__ import annotations

import json

import pytest
import torch

from chunk_selector.chunk_selector import (
    ChunkSelector,
    ChunkSelectorConfig,
    flatten_camera_feature_maps,
)
from chunk_selector.execution import (
    execution_steps_from_chunk_label,
    full_plan_execution_lengths,
    prediction_execution_lengths,
    required_only_execution_lengths,
)


def _selector_config(**overrides) -> ChunkSelectorConfig:
    values = {
        "input_dim": 16,
        "candidate_chunks": (1, 2, 4),
        "d_model": 32,
        "num_layers": 2,
        "n_heads": 4,
        "dim_feedforward": 64,
        "dropout": 0.0,
        "max_tokens": 18,
        "num_cameras": 2,
        "max_spatial_positions": 9,
    }
    values.update(overrides)
    return ChunkSelectorConfig(**values)


def test_selector_maps_classes_to_execution_chunks():
    selector = ChunkSelector(_selector_config()).eval()
    with torch.no_grad():
        selector.head[-1].weight.zero_()
        selector.head[-1].bias.copy_(torch.tensor([0.0, 5.0, 0.0]))
    maps = [torch.randn(2, 16, 3, 3), torch.randn(2, 16, 3, 3)]
    tokens, camera_ids, spatial_ids = flatten_camera_feature_maps(maps)
    result = selector.select(
        tokens,
        camera_ids=camera_ids,
        spatial_ids=spatial_ids,
    )
    assert result.logits.shape == (2, 3)
    assert result.chunk_sizes.tolist() == [2, 2]
    assert not result.used_safe_fallback.any()


def test_selector_expected_mode_outputs_continuous_and_rounded_chunks():
    selector = ChunkSelector(
        _selector_config(candidate_chunks=(4, 10), selection_mode="expected")
    ).eval()
    with torch.no_grad():
        selector.head[-1].weight.zero_()
        selector.head[-1].bias.copy_(torch.log(torch.tensor([0.25, 0.75])))
    maps = [torch.randn(1, 16, 3, 3), torch.randn(1, 16, 3, 3)]
    tokens, camera_ids, spatial_ids = flatten_camera_feature_maps(maps)

    result = selector.select(tokens, camera_ids=camera_ids, spatial_ids=spatial_ids)

    assert result.continuous_chunk_sizes.item() == pytest.approx(8.5)
    assert result.chunk_sizes.item() == 9


def test_selector_uses_configured_low_confidence_fallback():
    config = _selector_config(
        confidence_threshold=0.9,
        safe_chunk=1,
    )
    selector = ChunkSelector(config).eval()
    with torch.no_grad():
        selector.head[-1].weight.zero_()
        selector.head[-1].bias.zero_()
    maps = [torch.randn(1, 16, 3, 3), torch.randn(1, 16, 3, 3)]
    tokens, camera_ids, spatial_ids = flatten_camera_feature_maps(maps)
    result = selector.select(
        tokens,
        camera_ids=camera_ids,
        spatial_ids=spatial_ids,
    )
    assert result.chunk_sizes.item() == 1
    assert result.used_safe_fallback.item()


def test_selector_sidecar_roundtrip(tmp_path):
    selector = ChunkSelector(_selector_config()).eval()
    selector.save_pretrained(tmp_path, metadata={"base_model": "unit-test"})
    loaded = ChunkSelector.from_pretrained(tmp_path).eval()
    maps = [torch.randn(1, 16, 3, 3), torch.randn(1, 16, 3, 3)]
    tokens, camera_ids, spatial_ids = flatten_camera_feature_maps(maps)
    expected = selector(
        tokens,
        camera_ids=camera_ids,
        spatial_ids=spatial_ids,
    )
    actual = loaded(
        tokens,
        camera_ids=camera_ids,
        spatial_ids=spatial_ids,
    )
    torch.testing.assert_close(actual, expected)
    payload = json.loads((tmp_path / ChunkSelector.CONFIG_NAME).read_text())
    assert payload["candidate_chunks"] == [1, 2, 4]


def test_selector_rejects_policy_mismatch():
    selector = ChunkSelector(_selector_config())
    with pytest.raises(ValueError, match="feature_dim"):
        selector.validate_for_policy(feature_dim=8, max_chunk=4)
    with pytest.raises(ValueError, match="exceed"):
        selector.validate_for_policy(feature_dim=16, max_chunk=2)


def test_selector_requires_ids_for_enabled_embeddings():
    selector = ChunkSelector(_selector_config()).eval()
    features = torch.randn(1, 18, 16)
    with pytest.raises(ValueError, match="camera ids are required"):
        selector(features)


@pytest.mark.parametrize("requested", [2, 4, 10])
def test_execution_chunk_does_not_change_prediction_horizon(requested):
    prediction, execution = full_plan_execution_lengths(
        prediction_horizon=20,
        default_execution_steps=20,
        requested_execution_steps=requested,
        max_execution_steps=19,
    )
    assert prediction == 20
    assert execution == requested


def test_execution_chunk_rejects_pushbox_stale_alignment_slot():
    with pytest.raises(ValueError, match=r"\[1, 19\]"):
        full_plan_execution_lengths(
            prediction_horizon=20,
            default_execution_steps=20,
            requested_execution_steps=20,
            max_execution_steps=19,
        )


@pytest.mark.parametrize("requested", [2, 4, 10])
def test_required_only_prediction_matches_execution_length(requested):
    prediction, execution = required_only_execution_lengths(
        prediction_horizon=10,
        default_execution_steps=2,
        requested_execution_steps=requested,
        max_execution_steps=10,
    )
    assert prediction == requested
    assert execution == requested


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("full_then_truncate", (10, 4)),
        ("required_only", (4, 4)),
    ],
)
def test_prediction_mode_resolves_fixed_execution(mode, expected):
    assert prediction_execution_lengths(
        mode=mode,
        prediction_horizon=10,
        default_execution_steps=4,
        requested_execution_steps=None,
        max_execution_steps=10,
    ) == expected


def test_prediction_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown prediction mode"):
        prediction_execution_lengths(
            mode="invalid",
            prediction_horizon=10,
            default_execution_steps=4,
            requested_execution_steps=None,
            max_execution_steps=10,
        )


@pytest.mark.parametrize(
    ("requested", "expected_prediction"),
    [(4, 5), (10, 11), (19, 20)],
)
def test_required_only_prediction_includes_pushbox_stale_prefix(
    requested, expected_prediction
):
    assert prediction_execution_lengths(
        mode="required_only",
        prediction_horizon=20,
        default_execution_steps=19,
        requested_execution_steps=requested,
        max_execution_steps=19,
        dropped_prediction_steps=1,
    ) == (expected_prediction, requested)


@pytest.mark.parametrize(
    ("chunk_label", "execution_steps"),
    [(10, 10), (15, 15), (19, 19), (20, 19)],
)
def test_pushbox_full_chunk_label_aliases_to_usable_prefix(
    chunk_label, execution_steps
):
    assert execution_steps_from_chunk_label(
        chunk_label,
        max_execution_steps=19,
        full_chunk_label=20,
    ) == execution_steps
