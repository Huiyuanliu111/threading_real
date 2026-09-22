from pathlib import Path
import os
import ast
import math

import pytest
import torch
import torch.nn.functional as F

from autohorizon import AutoHorizonConfig, predict_action_with_autohorizon, select_horizon
from autohorizon import official

UPSTREAM = Path(os.environ.get('AUTOHORIZON_REFERENCE',
    '/home/huiyuan/pushbox/scripts/autohorizon/official.py'))


def upstream_functions():
    if not UPSTREAM.exists():
        pytest.skip('set AUTOHORIZON_REFERENCE to reference official.py or upstream pi0_pytorch.py')
    tree = ast.parse(UPSTREAM.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             {'pick_horizon_softpointer', '_soft_pointer_prefix', 'bidir_soft_pointer'}]
    namespace = dict(torch=torch, math=math, F=F)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(UPSTREAM), 'exec'), namespace)
    return namespace


def test_vendored_functions_are_identical_to_clone():
    if not UPSTREAM.exists():
        pytest.skip('reference source unavailable')
    def functions(path):
        return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef)}
    original, copied = functions(UPSTREAM), functions(Path(official.__file__))
    assert len(copied) == 3
    for name, value in copied.items():
        assert value == original[name]


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
@pytest.mark.parametrize('p', [2, 10, 20, 50])
@pytest.mark.parametrize('method', ['forward', 'bidirectional'])
def test_independent_upstream_parity(device, p, method):
    funcs = upstream_functions()
    torch.manual_seed(17)
    maps = [torch.rand(p, p, device=device), torch.eye(p, device=device),
            torch.ones(p, p, device=device), torch.softmax(torch.randn(p, p, device=device) * 8, -1)]
    for a in maps:
        for q, threshold in [(0.9, 0.3), (1.0, 0.0), (0.5, 0.7)]:
            cfg = AutoHorizonConfig(method=method, max_entropy_q=q, hold_thr=threshold)
            name = 'bidir_soft_pointer' if method == 'bidirectional' else 'pick_horizon_softpointer'
            expected, diags = funcs[name](a, hold_thr=threshold, max_entropy_q=q)
            actual = select_horizon(a, cfg)
            assert actual.raw_horizon == int(expected)
            assert actual.diagnostics.keys() == diags.keys()
            for key, value in diags.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual.diagnostics[key], value, rtol=0, atol=0)
                else:
                    assert actual.diagnostics[key] == value


@pytest.mark.parametrize('bad', [torch.ones(1, 1), torch.zeros(3, 3),
    torch.full((3, 3), float('nan')), -torch.ones(3, 3), torch.ones(3, 2)])
def test_invalid_attention(bad):
    with pytest.raises(ValueError):
        select_horizon(bad)


def test_average_layers_heads_before_normalization():
    a = torch.rand(3, 2, 20, 20)
    a[0] *= 100
    actual = select_horizon(a)
    expected, _ = official.bidir_soft_pointer(a.mean((0, 1)))
    assert actual.raw_horizon == int(expected)
    torch.testing.assert_close(actual.attention, a.mean((0, 1)), rtol=0, atol=0)


class TinyPolicy(torch.nn.Module):
    """Exercise real ARP Attention hooks without vision encoders/simulation."""
    def __init__(self, fail=False):
        super().__init__()
        from pushbox.arp import Attention
        self.policy = torch.nn.Module()
        self.policy.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
        for block in self.policy.blocks:
            block.attn = Attention(16, 2)
        self.horizon = self.action_chunk_size = self.max_selector_chunk_label = 20
        self.n_obs_steps, self.plan_steps = 2, 0
        self.inference_chunk_size = 5
        self.prediction_mode = 'required_only'
        object.__setattr__(self, 'chunk_selector', object())
        self.fail, self.calls, self.use_sample = fail, 0, False
        self.eval()

    def set_prediction_mode(self, value):
        self.prediction_mode = value

    def predict_action(self, observation):
        self.calls += 1
        assert self.prediction_mode == 'full_then_truncate'
        assert self.inference_chunk_size == 20
        assert self.chunk_selector is None
        x = observation['x']
        for block in self.policy.blocks:
            x = block.attn(x)
        if self.fail:
            raise RuntimeError('deliberate failure')
        actions = x[:, -20:, :7]
        return {'action_pred': actions, 'action': actions[:, 1:]}


def test_policy_alignment_single_call_and_state_restore():
    torch.manual_seed(9)
    policy = TinyPolicy()
    selector = policy.chunk_selector
    obs = {'x': torch.randn(1, 22, 16)}
    result = predict_action_with_autohorizon(policy, obs, prediction_offset=1)
    assert result.attention.shape == (20, 20)
    assert result.h_star == max(1, result.raw_horizon - 1)
    assert result.actions.shape == (result.h_star, 7)
    assert result.generated_action_tokens == 20
    assert policy.calls == 1 and not policy.use_sample
    assert policy.chunk_selector is selector
    assert policy.inference_chunk_size == 5 and policy.prediction_mode == 'required_only'
    assert all(not b.attn.attn_dropout._forward_pre_hooks for b in policy.policy.blocks)
    policy.inference_chunk_size, policy.prediction_mode = 20, 'full_then_truncate'
    object.__setattr__(policy, 'chunk_selector', None)
    baseline = policy.predict_action(obs)['action'][0]
    torch.testing.assert_close(result.actions, baseline[:result.h_star], rtol=0, atol=0)


def test_cleanup_on_failure():
    policy = TinyPolicy(fail=True)
    selector = policy.chunk_selector
    with pytest.raises(RuntimeError, match='deliberate'):
        predict_action_with_autohorizon(policy, {'x': torch.randn(1, 22, 16)}, prediction_offset=1)
    assert policy.chunk_selector is selector
    assert policy.prediction_mode == 'required_only' and policy.inference_chunk_size == 5
    assert all(not b.attn.attn_dropout._forward_pre_hooks for b in policy.policy.blocks)


def test_run_length_restriction_and_nondefault_parity():
    with pytest.raises(ValueError):
        select_horizon(torch.eye(3), AutoHorizonConfig(run_len=3))
    a = torch.rand(20, 20)
    expected, _ = upstream_functions()['bidir_soft_pointer'](a, run_len=2)
    assert select_horizon(a, AutoHorizonConfig(run_len=2)).raw_horizon == int(expected)


def test_threading_public_full_label_caps_tail_without_dropping_first_action():
    class ThreadingLikePolicy(TinyPolicy):
        def predict_action(self, observation):
            output = super().predict_action(observation)
            output['action'] = output['action_pred'][:, :19]
            return output

    policy = ThreadingLikePolicy()
    result = predict_action_with_autohorizon(
        policy, {'x': torch.randn(1, 22, 16)}, prediction_offset=0
    )
    assert result.h_star == min(result.raw_horizon, 19)
    assert result.prediction_offset == 0
    assert result.attention.shape == (20, 20)


@pytest.mark.parametrize('plan_steps', [0, 2])
def test_real_threading_policy_full_prediction_and_default_offset(plan_steps):
    from threading_task.policy import ThreadingARPolicy
    from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
    torch.manual_seed(12)
    policy = ThreadingARPolicy(
        shape_meta={'action': {'shape': [7]}, 'obs': {
            'agent_pos': {'shape': [8], 'type': 'low_dim'},
            'sideview': {'shape': [3, 32, 32], 'type': 'rgb'},
            'frontview': {'shape': [3, 32, 32], 'type': 'rgb'},
        }}, horizon=4, n_action_steps=2, n_obs_steps=2, pretrained=False,
        arp_cfg={'n_embd': 16, 'num_layers': 2, 'layer_cfg': {'n_head': 2},
                 'plan_steps': plan_steps, 'action_chunk_size': 4, 'sample': False},
    ).eval()
    normalizer = LinearNormalizer()
    normalizer.fit({'agent_pos': torch.randn(30, 8), 'action': torch.randn(30, 7)})
    policy.set_normalizer(normalizer)
    policy.set_prediction_mode('required_only')
    policy.inference_chunk_size = 1
    obs = {'agent_pos': torch.randn(1, 2, 8), 'sideview': torch.rand(1, 2, 3, 32, 32),
           'frontview': torch.rand(1, 2, 3, 32, 32)}
    result = predict_action_with_autohorizon(policy, obs)
    assert result.prediction_offset == 0
    assert result.h_star == result.raw_horizon
    assert result.attention.shape == (4, 4)
    assert result.generated_action_tokens == 4
    assert policy.inference_chunk_size == 1 and policy.prediction_mode == 'required_only'
    assert policy.chunk_selector is None and not policy.use_sample and policy.low_var_eval
    assert all(not b.attn.attn_dropout._forward_pre_hooks for b in policy.policy.blocks)
    policy.set_prediction_mode('full_then_truncate')
    policy.inference_chunk_size = 4
    baseline = policy.predict_action(obs)['action'][0]
    torch.testing.assert_close(result.actions, baseline[:result.h_star], rtol=0, atol=0)
    import json
    json.dumps(result.metrics(), allow_nan=False)


@pytest.mark.parametrize('case', ['training', 'batch', 'multichunk', 'multitoken', 'offset'])
def test_unsupported_inputs_fail_before_prediction(case):
    policy = TinyPolicy()
    observation = {'x': torch.randn(1, 22, 16)}
    offset = 1
    if case == 'training':
        policy.train()
    elif case == 'batch':
        observation['x'] = torch.randn(2, 22, 16)
    elif case == 'multichunk':
        policy.action_chunk_size = 5
    elif case == 'multitoken':
        policy.action_tokens = 6
    else:
        offset = 0.5
    with pytest.raises(ValueError):
        predict_action_with_autohorizon(policy, observation, prediction_offset=offset)
    assert policy.calls == 0
    assert all(not b.attn.attn_dropout._forward_pre_hooks for b in policy.policy.blocks)


def test_cleanup_on_attention_validation_failure():
    policy = TinyPolicy()
    selector = policy.chunk_selector
    # An extra prompt token must not silently shift the action-attention slice.
    with pytest.raises(RuntimeError, match='full ARP token sequence'):
        predict_action_with_autohorizon(
            policy, {'x': torch.randn(1, 23, 16)}, prediction_offset=1)
    assert policy.chunk_selector is selector
    assert policy.inference_chunk_size == 5 and policy.prediction_mode == 'required_only'
    assert all(not b.attn.attn_dropout._forward_pre_hooks for b in policy.policy.blocks)
