import pytest
import torch

from pushbox.workspace import evaluate_loss, validation_batch_boundaries


def test_half_epoch_validation_waits_for_optimizer_update():
    assert validation_batch_boundaries(4151, 0.5, 8, 0) == {2080, 4151}
    assert validation_batch_boundaries(16, 0.5, 8, 0) == {8, 16}
    assert validation_batch_boundaries(3, 0.5, 8, 0) == {3}
    assert validation_batch_boundaries(100, 1, 8, 0) == {100}
    assert validation_batch_boundaries(100, 5, 8, 3) == set()
    assert validation_batch_boundaries(100, 5, 8, 4) == {100}


@pytest.mark.parametrize("interval", [0, -1, 1.5, 0.3, float("nan")])
def test_invalid_validation_interval(interval):
    with pytest.raises(ValueError):
        validation_batch_boundaries(100, interval, 8, 0)


def test_validation_restores_training_and_preserves_gradients():
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def compute_loss(self, batch):
            assert not self.training
            assert not torch.is_grad_enabled()
            return {"loss": batch["action"].mean() * self.weight}

    policy = Policy().train()
    policy.weight.grad = torch.tensor(2.0)
    result = evaluate_loss(policy, [{"action": torch.ones(2, 1)},
                                    {"action": torch.full((1, 1), 4.0)}], "cpu")
    assert result == {"val.loss": 2.0, "val_loss": 2.0}
    assert policy.training
    assert policy.weight.grad.item() == 2
    policy.eval()
    evaluate_loss(policy, [], "cpu")
    assert not policy.training
