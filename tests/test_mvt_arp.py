import pytest
import torch

from threading_task.mvt_arp_policy import (
    ThreadingMVTARPPolicy, _rotation_from_control_points,
)
from threading_task.mvt_renderer import OrthographicMVTRenderer


def test_mvt_projection_pair_round_trip():
    renderer = OrthographicMVTRenderer(image_size=420)
    points = torch.tensor([[[0.30, -0.10, 0.00], [0.50, 0.10, 0.20]]])
    recovered = renderer.from_cube(renderer.unproject_pair(
        renderer.project(renderer.to_cube(points))))
    torch.testing.assert_close(recovered, points, atol=1e-6, rtol=0)


def test_control_points_recover_full_rotation():
    points = torch.tensor([[[0.4, 0.0, 0.0], [0.4, 0.04, 0.0],
                            [0.36, 0.0, 0.0]]])
    rotation = _rotation_from_control_points(points)
    expected = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, -0.0],
                              [0.0, 0.0, 1.0]]])
    torch.testing.assert_close(rotation, expected, atol=1e-6, rtol=0)


def test_vit_receives_spatial_loss_gradient():
    model = ThreadingMVTARPPolicy(horizon=1, n_action_steps=1,
        image_size=28, patch_size=14, vit_depth=1, arp_depth=1)
    points = torch.tensor([[[[0.30, -0.10, 0.00], [0.40, 0.00, 0.10],
                             [0.50, 0.10, 0.20]]]])
    control = torch.tensor([[[[0.4, 0.0, 0.0], [0.44, 0.0, 0.0],
                              [0.4, 0.04, 0.0]]]])
    obs = {"points": points, "colors": torch.ones_like(points) * 0.5,
           "valid_points": torch.tensor([[3]]), "agent_pos": torch.zeros(1, 1, 8),
           "control_points": control}
    batch = {"obs": obs, "target_control_points": control,
             "target_gripper": torch.zeros(1, 1, 1), "action": torch.zeros(1, 1, 7),
             "action_is_pad": torch.zeros(1, 1, dtype=torch.bool)}
    loss = sum(model.compute_loss(batch).values())
    loss.backward()
    assert model.vit.layers[0].attn.to_qkv.weight.grad is not None
    assert model.vit.layers[0].attn.to_qkv.weight.grad.abs().sum() > 0


def _plan_batch(horizon=3):
    control = torch.tensor([[[[0.4, 0.0, 0.0], [0.44, 0.0, 0.0],
                              [0.4, 0.04, 0.0]]]]).repeat(2, 1, 1, 1)
    targets = control.repeat(1, horizon, 1, 1)
    targets += torch.arange(horizon)[None, :, None, None] * 0.01
    points = control.clone().reshape(2, 1, 3, 3)
    return {"obs": {"points": points, "colors": torch.ones_like(points) * 0.5,
                    "valid_points": torch.tensor([[3], [3]]),
                    "agent_pos": torch.zeros(2, 1, 8), "control_points": control},
            "target_control_points": targets, "target_gripper": torch.zeros(2, horizon, 1),
            "action_is_pad": torch.zeros(2, horizon, dtype=torch.bool)}


def _small_model(**kwargs):
    return ThreadingMVTARPPolicy(horizon=3, n_action_steps=2,
        image_size=28, patch_size=14, hidden_dim=32, vit_mlp_dim=64,
        vit_depth=1, arp_depth=1, dropout=0, **kwargs)


def test_plan_resampling_excludes_padding_and_reverses_only_plan():
    model = _small_model(plan_steps=3)
    batch = _plan_batch()
    batch["action_is_pad"][0, -1] = True
    batch["target_control_points"][0, -1] = 100  # must never enter the plan
    plan = model._plan_targets(batch["target_control_points"], batch["action_is_pad"])
    torch.testing.assert_close(plan[:, -1], batch["target_control_points"][:, 0])
    torch.testing.assert_close(plan[0, 0], batch["target_control_points"][0, 1])
    torch.testing.assert_close(plan[0, 1], batch["target_control_points"][0, :2].mean(0))
    assert plan.max() < 1
    batch["action_is_pad"][0] = True
    with pytest.raises(ValueError, match="at least one valid"):
        model._plan_targets(batch["target_control_points"], batch["action_is_pad"])


@pytest.mark.parametrize("plan_steps,action_chunk_size", [(0, 1), (3, 1), (3, 2), (3, 3)])
def test_plan_training_and_observation_only_inference(plan_steps, action_chunk_size):
    model = _small_model(plan_steps=plan_steps, action_chunk_size=action_chunk_size)
    batch = _plan_batch()
    batch["action_is_pad"][0, -1] = True
    losses = model.compute_loss(batch)
    assert all(torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()
    if plan_steps:
        assert "coarse-plan.2d_ce_loss" in losses
        head = model.policy.token_predictors[model.policy.token_name_2_ids["coarse-plan"]]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    result = model.predict_action(batch["obs"])
    assert result["action"].shape == (2, 2, 7)
    assert result["action_pred"].shape == (2, 3, 7)
    assert torch.isfinite(result["action_pred"]).all()
    if plan_steps:
        assert result["plan_control_points"].shape == (2, plan_steps, 3, 3)
        assert torch.isfinite(result["plan_control_points"]).all()
    else:
        assert "plan_control_points" not in result


def test_plan_is_causal_condition_for_actions():
    torch.manual_seed(7)
    model = _small_model(plan_steps=3, action_chunk_size=3).eval()
    # AdaLN residual gates start at zero; open them to exercise attention.
    for block in model.policy.blocks:
        torch.nn.init.constant_(block.adaLN_modulation[-1].bias, 0.1)
    batch = _plan_batch()
    with torch.no_grad():
        _, visual_tokens, encoded = model._visual(batch["obs"])
        tokens, chunks, valid, contexts = model._training_sequence(batch, encoded)
        contexts["visual-tokens"] = visual_tokens
        captured = {}
        handles = []
        for name in ("coarse-plan", "target-point"):
            head = model.policy.token_predictors[model.policy.token_name_2_ids[name]]
            handles.append(head.register_forward_pre_hook(
                lambda module, args, name=name: captured.__setitem__(name, args[0].clone())))
        try:
            model.policy.compute_loss(tokens, chunks, valid_tk_mask=valid, contexts=contexts)
            original = dict(captured)
            changed_actions = tokens.clone()
            changed_actions[:, 6 + 3 * 6:, :2] += 2
            model.policy.compute_loss(changed_actions, chunks, valid_tk_mask=valid, contexts=contexts)
            torch.testing.assert_close(captured["coarse-plan"], original["coarse-plan"], atol=0, rtol=0)
            changed_plan = tokens.clone()
            changed_plan[:, 6:6 + 3 * 6, :2] += 2
            model.policy.compute_loss(changed_plan, chunks, valid_tk_mask=valid, contexts=contexts)
            assert not torch.allclose(captured["target-point"], original["target-point"])
        finally:
            for handle in handles:
                handle.remove()


def test_inference_actions_use_generated_plan(monkeypatch):
    torch.manual_seed(11)
    model = _small_model(plan_steps=3, action_chunk_size=3).eval()
    for block in model.policy.blocks:
        torch.nn.init.constant_(block.adaLN_modulation[-1].bias, 0.1)
    original_generate = model.policy.generate
    plan_pixels = torch.arange(18).reshape(1, 18, 1).expand(2, -1, 2).float() + 1
    captured = []
    action_head = model.policy.token_predictors[model.policy.token_name_2_ids["target-point"]]
    handle = action_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].clone()))

    def generate_with_plan(prompt, future, **kwargs):
        assert {token["chk_id"] for token in future[:18]} == {6}
        assert {token["chk_id"] for token in future[18:]} == {7}
        # Intervene only on coarse sampling; execute the real action decoder.
        return original_generate(prompt, future, sample_function={6: lambda _: plan_pixels}, **kwargs)

    monkeypatch.setattr(model.policy, "generate", generate_with_plan)
    try:
        result = model.predict_action(_plan_batch()["obs"])
        expected = model.renderer.from_cube(model.renderer.unproject_pair(
            plan_pixels.reshape(2, 3, 3, 2, 2))).flip(1)
        torch.testing.assert_close(result["plan_control_points"], expected)
        original_action_input = captured[0]
        captured.clear()
        plan_pixels = plan_pixels + 3
        model.predict_action(_plan_batch()["obs"])
        assert not torch.allclose(captured[0], original_action_input)
    finally:
        handle.remove()


@pytest.mark.parametrize("action_chunk_size", [1, 3])
def test_disabled_gripper_has_no_loss_or_generation(action_chunk_size):
    model = _small_model(plan_steps=3, action_chunk_size=action_chunk_size,
                         predict_gripper=False)
    batch = _plan_batch()
    del batch["target_gripper"]  # training must not even require these labels
    batch["obs"]["agent_pos"][:, :, 7] = torch.tensor([[0.021], [0.025]])
    grip_id = model.policy.token_name_2_ids["target-gripper"]
    head = model.policy.token_predictors[grip_id]

    def unexpected_gripper(*args):
        raise AssertionError("disabled gripper predictor was called")

    handle = head.register_forward_pre_hook(unexpected_gripper)
    try:
        losses = model.compute_loss(batch)
        assert set(losses) == {"target-point.2d_ce_loss", "coarse-plan.2d_ce_loss"}
        sum(losses.values()).backward()
        assert all(p.grad is None for p in head.parameters())
        result = model.predict_action(batch["obs"])
        assert result["action_pred"].shape == (2, 3, 7)
        assert torch.isfinite(result["action_pred"]).all()
        assert torch.count_nonzero(result["action_pred"][..., 6]) == 0
    finally:
        handle.remove()
