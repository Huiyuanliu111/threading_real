import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from AAC import AACConfig, AACInference, select_chunk_size, threading_delta_to_aac
from AAC.aac import (
    binary_discrete_entropy,
    compute_entropy_boundary,
    gaussian_differential_entropy,
    rotation_prefix_magnitude,
)


def samples(horizon=8):
    generator = torch.Generator().manual_seed(24)
    x = torch.randn(20, horizon, 7, generator=generator, dtype=torch.float64) * 0.1
    x[..., 6] = (x[..., 6] > 0).double()
    return x


def test_entropy_matches_independent_numpy_full_covariance():
    x = samples()[..., :3]
    # Correlation makes a diagonal-only covariance implementation incorrect.
    x[..., 1] += 3 * x[..., 0]
    expected = []
    for t in range(x.shape[1]):
        cov = np.cov(x[:, t].numpy(), rowvar=False) + 1e-6 * np.eye(3)
        expected.append(0.5 * (3 * np.log(2 * np.pi * np.e) + np.linalg.slogdet(cov)[1]))
    np.testing.assert_allclose(gaussian_differential_entropy(x), expected, atol=1e-10)


def test_binary_entropy_endpoints_and_balanced_samples():
    states = torch.tensor([[0, 1, 0], [0, 1, 1]])
    torch.testing.assert_close(binary_discrete_entropy(states), torch.tensor([0., 0., math.log(2)]))


@pytest.mark.parametrize("prefix, expected", [([0., 8., 9.], 1), ([0., 1., 9.], 2), ([3.], 1), ([1., 1., 1.], 1)])
def test_boundary_is_length_before_largest_jump(prefix, expected):
    assert compute_entropy_boundary(torch.tensor(prefix))[0] == expected


def test_motion_is_displacement_not_path_length_and_strict_threshold():
    x = samples(4)
    nominal = torch.zeros(4, 7, dtype=torch.float64)
    nominal[:, 0] = torch.tensor([1., -1., 3., 0.01])
    decision = select_chunk_size(x, nominal=nominal)
    torch.testing.assert_close(decision.translation_magnitude, torch.tensor([1., 0., 3., 3.01], dtype=torch.float64), atol=1e-8, rtol=1e-7)
    assert decision.xi == 4  # Equality with alpha=3 is not a crossing.
    assert decision.h_star == 4


def test_rotation_composition_matches_scipy_noncommuting_rotations():
    r = np.array([[0.4, 0, 0], [0, 0.7, 0], [0, 0, -0.3]])
    total = Rotation.identity()
    expected = []
    for delta in r:
        total = Rotation.from_rotvec(delta) * total
        expected.append(np.linalg.norm(total.as_rotvec()))
    np.testing.assert_allclose(rotation_prefix_magnitude(torch.from_numpy(r)), expected, atol=1e-12)
    np.testing.assert_allclose(rotation_prefix_magnitude(torch.tensor([[0., 0., 0.], [0., 0., 0.]])), 0)


def test_gripper_magnitude_has_unit_weight_and_persists_after_switch():
    nominal = torch.zeros(4, 7, dtype=torch.float64)
    nominal[:, 6] = torch.tensor([0., 1., 0., 0.])
    result = select_chunk_size(samples(4), nominal=nominal, config=AACConfig(alpha=0.9))
    torch.testing.assert_close(result.action_magnitude, torch.tensor([0., 1., 1., 1.], dtype=torch.float64))
    assert result.xi == 2


def test_no_motion_fallback_and_one_step_horizon():
    for h in (1, 8):
        result = select_chunk_size(samples(h), nominal=torch.zeros(h, 7, dtype=torch.float64))
        assert result.xi == result.h_star == h
        assert not result.magnitude_threshold_reached
        assert torch.isfinite(result.entropy_total).all()


def test_separate_normalized_entropy_keeps_physical_gripper_and_motion():
    x = samples()
    normalized = x * 20 + 5
    result = select_chunk_size(x, entropy_candidates=normalized)
    reference = select_chunk_size(x)
    torch.testing.assert_close(result.entropy_gripper, reference.entropy_gripper)
    torch.testing.assert_close(result.action_magnitude, reference.action_magnitude)
    torch.testing.assert_close(result.entropy_translation, gaussian_differential_entropy(normalized[..., :3]))


def test_threading_gripper_width_deltas_are_integrated_per_candidate():
    actions = torch.zeros(2, 4, 7)
    actions[0, :, 6] = torch.tensor([-0.02, -0.03, 0., 0.05])
    converted = threading_delta_to_aac(actions, current_width=0.08, closed_width_threshold=0.04)
    torch.testing.assert_close(converted[0, :, 6], torch.tensor([0., 1., 1., 0.]))
    torch.testing.assert_close(converted[1, :, 6], torch.zeros(4))
    assert actions[0, 0, 6] == -0.02  # No mutation of executable actions.


def test_inference_one_batch_and_exact_original_prefix():
    x = samples()
    calls = []

    def sampler(n):
        calls.append(n)
        return x

    prediction = AACInference().predict(sampler)
    assert calls == [20]
    assert prediction.predicted_horizon == 8
    torch.testing.assert_close(prediction.actions, x[0, :prediction.decision.h_star])
    assert prediction.actions.data_ptr() != x.data_ptr()
    assert isinstance(prediction.decision.metrics()["action_magnitude"], list)


def test_identical_samples_rejected_by_sampler_but_regularized_in_core():
    x = torch.zeros(20, 4, 7)
    assert torch.isfinite(select_chunk_size(x).entropy_total).all()
    with pytest.raises(ValueError, match="identical"):
        AACInference().predict(lambda n: x)


def test_half_precision_sampler_preserves_executable_dtype():
    x = samples().half()
    prediction = AACInference().predict(lambda n: x)
    assert prediction.actions.dtype == torch.float16
    assert prediction.decision.entropy_total.dtype == torch.float32
    torch.testing.assert_close(prediction.actions, x[0, :prediction.decision.h_star])


@pytest.mark.parametrize("candidate_index", [0, 2])
def test_designated_candidate_supplies_motion_and_execution(candidate_index):
    x = samples(4)
    x[..., :6] = 0
    x[0, :, 0] = 0.01
    x[2, :, 0] = 4
    cfg = AACConfig(execution_candidate_index=candidate_index)
    calls = []

    def sampler(n):
        calls.append(n)
        return x

    prediction = AACInference(cfg).predict(sampler, to_entropy=lambda a: a * 2)
    expected_motion = x[candidate_index, :, :3].cumsum(0).norm(dim=-1)
    torch.testing.assert_close(prediction.decision.translation_magnitude, expected_motion)
    torch.testing.assert_close(prediction.actions, x[candidate_index, :prediction.decision.h_star])
    assert calls == [20]
    metrics = prediction.metrics()
    assert metrics["execution_candidate_index"] == candidate_index
    assert metrics["generated_action_tokens"] == 20 * 4
    variance = (x * 2).float().var(dim=0, unbiased=True)
    assert metrics["candidate_mean_variance"] == pytest.approx(variance.mean().item())
    assert metrics["candidate_max_variance"] == pytest.approx(variance.max().item())
    assert metrics["predict_action_elapsed_sec"] >= 0


def test_xi_keeps_first_crossing_when_motion_reverses():
    x = samples(3)
    nominal = torch.zeros(3, 7, dtype=torch.float64)
    nominal[:, 0] = torch.tensor([4., -3., 4.])
    decision = select_chunk_size(x, nominal=nominal)
    assert decision.xi == 1


@pytest.mark.parametrize("index", [-1, 20, 0.5, True])
def test_invalid_candidate_index_rejected_before_sampling(index):
    calls = []
    with pytest.raises(ValueError, match="execution_candidate_index"):
        AACInference(AACConfig(execution_candidate_index=index)).predict(
            lambda n: calls.append(n)
        )
    assert not calls


def test_candidate_index_checked_against_actual_core_batch():
    with pytest.raises(ValueError, match="actual sample count"):
        select_chunk_size(samples()[:2], config=AACConfig(execution_candidate_index=2))


def test_pi05_deployment_batches_same_observation_and_executes_selected_prefix():
    from types import SimpleNamespace
    from AAC.pi05 import predict_pi05_aac

    actions = samples(20).float()
    calls = []
    batch = {"image": torch.ones(1, 3, 4, 4), "tokens": torch.ones(1, 5, dtype=torch.long)}

    class Model:
        def predict_action_chunk(self, repeated):
            calls.append(repeated)
            return actions

    deployment = SimpleNamespace(
        preprocess=lambda frame: batch,
        postprocess=lambda value: value * 0.1,
        model=Model(), horizon=20,
        aac=AACInference(AACConfig(execution_candidate_index=2)),
    )
    result = predict_pi05_aac(
        deployment, {"observation.state": torch.zeros(8)}, closed_width_threshold=0.035,
    )
    assert len(calls) == 1
    assert calls[0]["image"].shape == (20, 3, 4, 4)
    torch.testing.assert_close(calls[0]["tokens"], batch["tokens"].expand(20, -1))
    metrics = deployment.last_prediction_diagnostics
    expected = actions[2, :metrics["execution_chunk"]].clone() * 0.1
    expected[:, 6] = 0
    torch.testing.assert_close(result["action"][0], expected)
    assert metrics["predicted_chunk"] == 20
    assert metrics["generated_action_tokens"] == 400
    assert metrics["execution_candidate_index"] == 2
    assert batch["image"].shape[0] == 1
    assert (actions[..., 6] != 0).any()  # Postprocessing did not mutate samples.


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("batch_size", [None, 1, 6])
def test_arp_shared_encoding_sample_restore_and_exact_candidate(planar, fail, batch_size):
    from AAC.arp import predict_arp_aac

    class Policy(torch.nn.Module):
        horizon = 4
        low_var_eval = True

        def __init__(self):
            super().__init__()
            self.encodings = self.calls = 0
            self.actions = samples(4).float()
            self.actions[..., 6] *= 0.001
            if planar:
                self.actions = self.actions[..., :2]

        def _visual(self, obs):
            self.encodings += 1
            return (torch.ones(1, 2), torch.ones(1, 3, 2), torch.ones(1, 2, 2, 2))

        def predict_action(self, obs, *, visual_features, sample, prediction_mode, requested_steps):
            start = self.calls * (batch_size or 20)
            count = min(batch_size or 20, 20 - start)
            self.calls += 1
            assert sample and not self.low_var_eval
            assert requested_steps == 4 and prediction_mode == "full_then_truncate"
            assert obs["state"].shape == (count, 1, 2)
            assert all(value.shape[0] == count for value in visual_features)
            if fail:
                raise RuntimeError("generation failed")
            return {"action_pred": self.actions[start:start + count]}

    policy = Policy()
    aac = AACInference(AACConfig(execution_candidate_index=2))
    obs = {"state": torch.zeros(1, 1, 2)}
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            predict_arp_aac(policy, obs, aac, planar=planar, sample_batch_size=batch_size)
    else:
        prediction, result = predict_arp_aac(policy, obs, aac, planar=planar, sample_batch_size=batch_size)
        torch.testing.assert_close(prediction["action_pred"], policy.actions[2:3])
        torch.testing.assert_close(prediction["action"], policy.actions[2:3, :result.decision.h_star])
        assert prediction["prediction_diagnostics"]["generated_action_tokens"] == 80
        diagnostics = prediction["prediction_diagnostics"]
        assert diagnostics["requested_steps"] == policy.horizon
        assert diagnostics["generated_steps"] == policy.horizon
        assert diagnostics["execution_chunk"] == result.decision.h_star
    assert policy.low_var_eval
    assert policy.encodings == 1
    assert policy.calls == (1 if fail else math.ceil(20 / (batch_size or 20)))


@pytest.mark.parametrize("kind", ["nan", "shape", "one_sample", "gripper"])
def test_invalid_core_inputs(kind):
    x = samples()
    if kind == "nan":
        x[0, 0, 0] = float("nan")
    elif kind == "shape":
        x = x[..., :6]
    elif kind == "one_sample":
        x = x[:1]
    else:
        x[0, 0, 6] = 0.2
    with pytest.raises(ValueError):
        select_chunk_size(x)


@pytest.mark.parametrize("kwargs", [{"alpha": float("nan")}, {"alpha": -1}, {"covariance_eps": 0}, {"num_samples": 1}, {"num_samples": 2.5}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        AACConfig(**kwargs)
