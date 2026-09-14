from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chunk_selector.mvt_features import predict_for_execution
from scripts.deployment.endpoint_schedule import EndpointExecutionSchedule


class RecordingPolicy:
    image_size = 28
    patch_size = 14

    def __init__(self):
        self.encodings = 0

    def eval(self):
        return self

    def _visual(self, obs):
        self.encodings += 1
        return None, torch.zeros(1, 8, 32), None

    def predict_action(self, obs, **kwargs):
        self.kwargs = kwargs
        return kwargs


class Selector:
    config = SimpleNamespace(metadata={}, num_cameras=0, max_spatial_positions=0)

    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def select(self, tokens, **kwargs):
        self.calls += 1
        return SimpleNamespace(chunk_sizes=torch.tensor([3]))


@pytest.mark.parametrize("mode", ["full_then_truncate", "required_only"])
def test_selector_runs_once_before_decode_and_reuses_vision(mode):
    policy, selector = RecordingPolicy(), Selector()
    prediction, selection = predict_for_execution(
        policy, {}, prediction_mode=mode, execute_steps=10, selector=selector)
    assert policy.encodings == 1 and selector.calls == 1
    assert prediction["requested_steps"] == 3
    assert prediction["prediction_mode"] == mode
    assert prediction["visual_features"][1].shape == (1, 8, 32)
    assert selection.chunk_sizes.item() == 3


def test_endpoint_only_requests_short_decode_after_latching():
    policy = RecordingPolicy()
    schedule = EndpointExecutionSchedule(np.zeros(3))
    kwargs = dict(prediction_mode="required_only", execute_steps=1, schedule=schedule)
    prediction, _ = predict_for_execution(policy, {}, **kwargs)
    assert prediction["requested_steps"] == 10
    assert not schedule.fine_mode
    poses = np.tile(np.eye(4), (10, 1, 1))
    steps, _ = schedule.select([.2, 0, 0], poses)
    assert steps == 3  # transition used the complete prediction
    prediction, _ = predict_for_execution(policy, {}, **kwargs)
    assert prediction["requested_steps"] == 3
    steps, diagnostics = schedule.select([.2, 0, 0], poses[:4])
    assert steps == 3 and diagnostics["execution_phase"] == "fine"
    schedule.reset()
    with pytest.raises(ValueError):
        schedule.select([.2, 0, 0], poses[:4])


def test_fixed_length_and_conflicting_selectors():
    policy = RecordingPolicy()
    prediction, selection = predict_for_execution(
        policy, {}, prediction_mode="required_only", execute_steps=3)
    assert selection is None and prediction["requested_steps"] == 3
    with pytest.raises(ValueError):
        predict_for_execution(policy, {}, prediction_mode="required_only", execute_steps=3,
                              selector=Selector(), schedule=EndpointExecutionSchedule(np.zeros(3)))
