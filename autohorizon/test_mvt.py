import json

import pytest
import torch

from autohorizon import AutoHorizonConfig, predict_action_with_autohorizon, predict_mvt_with_autohorizon
from autohorizon.mvt import action_step_attention
from autohorizon import official
from threading_task.mvt_arp_policy import ThreadingMVTARPPolicy


def model_and_obs(*, grip=False, plan=2, chunk=3):
    torch.manual_seed(21)
    model = ThreadingMVTARPPolicy(
        horizon=3, n_action_steps=1, image_size=28, patch_size=14,
        hidden_dim=32, vit_mlp_dim=64, vit_depth=1, arp_depth=2, dropout=0,
        plan_steps=plan, action_chunk_size=chunk, predict_gripper=grip,
        pointcloud_views=("sideview",),
    ).eval()
    # Open residual gates so the test exercises meaningful attention paths.
    for block in model.policy.blocks:
        torch.nn.init.constant_(block.adaLN_modulation[-1].bias, 0.1)
    control = torch.tensor([[[[.4, 0., 0.], [.44, 0., 0.], [.4, .04, 0.]]]])
    obs = dict(points=control.clone(), colors=torch.ones_like(control) * .5,
               valid_points=torch.tensor([[3]]), agent_pos=torch.zeros(1, 1, 8),
               control_points=control)
    return model, obs


def test_step_attention_preserves_mass_and_temporal_axes():
    a = torch.arange(2 * 18 * 18, dtype=torch.float32).reshape(2, 18, 18)
    actual = action_step_attention(a, 3, 6)
    expected = torch.stack([torch.stack([
        a[:, i*6:(i+1)*6, j*6:(j+1)*6].sum(-1).mean(-1)
        for j in range(3)], -1) for i in range(3)], -2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.sum(-1), a.sum(-1).reshape(2, 3, 6).mean(-1))


@pytest.mark.parametrize('grip,plan,chunk', [(False, 0, 3), (False, 2, 3), (True, 2, 3),
                                                (False, 2, 2), (True, 2, 1)])
def test_real_single_camera_planarp_prefix_and_attention(monkeypatch, grip, plan, chunk):
    model, obs = model_and_obs(grip=grip, plan=plan, chunk=chunk)
    baseline = model.predict_action(obs, requested_steps=3)
    original = model.predict_action
    original_visual = model._visual
    calls, encodings = [], []
    def predict(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    def visual(*args, **kwargs):
        encodings.append(1)
        return original_visual(*args, **kwargs)
    monkeypatch.setattr(model, 'predict_action', predict)
    monkeypatch.setattr(model, '_visual', visual)
    captured = {}
    handles = [block.attn.attn_dropout.register_forward_pre_hook(
        lambda module, args, i=i: captured.setdefault(i, []).append(args[0].clone()))
        for i, block in enumerate(model.policy.blocks)]
    try:
        prediction, result = predict_mvt_with_autohorizon(model, obs)
        assert all(len(b.attn.attn_dropout._forward_pre_hooks) == 1 for b in model.policy.blocks)
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == len(encodings) == 1
    assert calls[0] == dict(prediction_mode='full_then_truncate', requested_steps=3, sample=False)
    assert model.n_action_steps == 1
    assert model.pointcloud_views == ('sideview',)
    torch.testing.assert_close(prediction['action_pred'], baseline['action_pred'], rtol=0, atol=0)
    torch.testing.assert_close(result.actions, baseline['action_pred'][0, :result.h_star], rtol=0, atol=0)
    assert prediction['action'].shape == (1, result.h_star, 7)
    k, start = 6 + int(grip), 6 + plan * 6
    # Independently form each temporal block, excluding current and plan tokens.
    layers = []
    for passes in captured.values():
        full = torch.zeros_like(passes[-1])
        previous = start
        for a in passes:
            end = a.shape[-1]
            if end <= start:
                continue
            full[:, :, previous:end, :end] = a[:, :, previous:end, :end]
            previous = end
        layers.append(torch.stack([torch.stack([
            full[0, :, start+i*k:start+(i+1)*k, start+j*k:start+(j+1)*k].sum(-1).mean(-1)
            for j in range(3)], -1) for i in range(3)], -2))
    if chunk < 3:
        # First group cannot attend to later groups, even though the last pass can.
        assert result.attention[:chunk, chunk:].count_nonzero() == 0
    expected = torch.stack(layers).mean((0, 1))
    torch.testing.assert_close(result.attention, expected, rtol=1e-6, atol=1e-7)
    assert result.h_star == int(official.bidir_soft_pointer(expected)[0])
    assert result.generated_action_tokens == 3 * k
    assert result.prediction_offset == 0
    assert result.metrics()['pointcloud_views'] == ['sideview']
    json.dumps(prediction['prediction_diagnostics'], allow_nan=False)
    public = predict_action_with_autohorizon(model, obs)
    torch.testing.assert_close(public.actions, result.actions, rtol=0, atol=0)


def test_mvt_cleanup_on_prediction_failure(monkeypatch):
    model, obs = model_and_obs()
    def fail(*args, **kwargs):
        raise RuntimeError('deliberate')
    monkeypatch.setattr(model, 'predict_action', fail)
    with pytest.raises(RuntimeError, match='deliberate'):
        predict_mvt_with_autohorizon(model, obs)
    assert all(not b.attn.attn_dropout._forward_pre_hooks for b in model.policy.blocks)
    assert model.n_action_steps == 1


@pytest.mark.parametrize('case', ['batch', 'training', 'run_len'])
def test_mvt_rejects_incompatible_settings_before_generation(monkeypatch, case):
    model, obs = model_and_obs()
    config = AutoHorizonConfig(run_len=3 if case == 'run_len' else 1)
    if case == 'batch':
        obs = {k: v.repeat_interleave(2, dim=0) for k, v in obs.items()}
    if case == 'training':
        model.train()
    def unexpected(*args, **kwargs):
        pytest.fail('invalid call must not generate actions')
    monkeypatch.setattr(model, 'predict_action', unexpected)
    with pytest.raises(ValueError):
        predict_mvt_with_autohorizon(model, obs, config)
