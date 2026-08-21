"""Training workspace hook that adds periodic Threading video rollouts."""
from __future__ import annotations

import hydra
import wandb

from pushbox.workspace import PushBoxARPWorkspace


class ThreadingARPWorkspace(PushBoxARPWorkspace):
    """Reuse the PushBox trainer and run Threading evaluation at checkpoint epochs.

    The base workspace currently has no rollout callback. Checkpoint saving is a
    stable epoch hook, so Threading uses it without changing PushBox behavior.
    Set ``checkpoint_every`` to a divisor of ``rollout_every``.
    """

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        self._threading_runner = None
        self._last_rollout_epoch = -1

    def _maybe_rollout(self) -> dict | None:
        training = self.cfg.training
        every = int(training.rollout_every)
        if (
            not bool(getattr(training, "enable_rollout", False))
            or self.epoch <= 0
            or self.epoch % every != 0
            or self._last_rollout_epoch == self.epoch
        ):
            return None
        if self._threading_runner is None:
            self._threading_runner = hydra.utils.instantiate(self.cfg.task.env_runner)
        policy = self.ema_model if training.use_ema and self.ema_model is not None else self.model
        was_training = policy.training
        policy.eval()
        rollout_log = self._threading_runner.run(policy)
        rollout_log["global_step"] = self.global_step
        wandb.log(rollout_log)
        if was_training:
            policy.train()
        self._last_rollout_epoch = self.epoch
        return rollout_log

    def save_checkpoint(self, *args, **kwargs):
        # Run rollout BEFORE saving so the success rate can drive topk selection.
        rollout_log = self._maybe_rollout()
        if rollout_log is not None:
            # Expose the rollout success rate for PushBoxARPWorkspace.run() to
            # use as the topk checkpoint metric.
            self._last_rollout_sr = rollout_log.get("test_success_rate", 0.0)
        path = super().save_checkpoint(*args, **kwargs)
        return path
