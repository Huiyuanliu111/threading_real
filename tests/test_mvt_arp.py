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
