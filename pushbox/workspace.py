"""PushBox ARP training workspace (self-contained under pushbox/)."""
from __future__ import annotations

from collections import defaultdict
import copy
import math
import os
import pathlib
import random

import hydra
import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from pushbox.diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from pushbox.diffusion_policy.common.json_logger import JsonLogger
from pushbox.diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from pushbox.diffusion_policy.dataset.base_dataset import BaseImageDataset
from pushbox.diffusion_policy.model.common.lr_scheduler import get_scheduler
from pushbox.diffusion_policy.model.diffusion.ema_model import EMAModel
from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from pushbox.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from pushbox.diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def compute_grad_norm(model):
    if isinstance(model, nn.Module):
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
    else:
        grads = [param.grad.detach().flatten() for param in model if param.grad is not None]
    return torch.cat(grads).norm()


def validation_batch_boundaries(train_batches, val_every, accumulation, epoch):
    """Validation intervals in epochs, rounded up to optimizer boundaries."""
    interval = float(val_every)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("val_every must be positive and finite")
    if interval >= 1:
        if not interval.is_integer():
            raise ValueError("val_every >= 1 must be an integer number of epochs")
        return {train_batches} if (epoch + 1) % int(interval) == 0 else set()
    parts = round(1 / interval)
    if not math.isclose(parts * interval, 1):
        raise ValueError("fractional val_every must evenly divide one epoch")
    return {min(train_batches, math.ceil(train_batches * part / parts / accumulation) * accumulation)
            for part in range(1, parts + 1)}


@torch.no_grad()
def evaluate_loss(policy, dataloader, device, max_steps=None, description="Validation"):
    """Evaluate held-out losses and restore the policy's training mode."""
    was_training = policy.training
    totals = defaultdict(float)
    count = 0
    policy.eval()
    try:
        for index, batch in enumerate(tqdm.tqdm(dataloader, desc=description, leave=False)):
            if max_steps is not None and index >= max_steps:
                break
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            losses = policy.compute_loss(batch)
            size = len(batch["action"])
            for key, value in losses.items():
                totals["val." + key] += value.item() * size
            totals["val_loss"] += sum(losses.values()).item() * size
            count += size
    finally:
        policy.train(was_training)
    return {key: value / count for key, value in totals.items()} if count else {}


def aligned_action_target(
    model: BaseImagePolicy,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the actions aligned with the latest observation and their valid mask."""
    start = int(model.n_obs_steps) - 1
    end = start + int(model.horizon)
    target = batch["action"][:, start:end]
    if target.shape[1] != int(model.horizon):
        raise ValueError(
            f"Expected {model.horizon} aligned actions, got {target.shape[1]}"
        )
    valid = None
    if "action_is_pad" in batch:
        valid = ~batch["action_is_pad"][:, start:end].bool()
    return target, valid


def masked_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Action prediction/target shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    squared_error = (prediction - target).square()
    if valid is None:
        return squared_error.mean()
    if valid.shape != prediction.shape[:2]:
        raise ValueError(
            f"Action valid mask shape {tuple(valid.shape)} does not match "
            f"{tuple(prediction.shape[:2])}"
        )
    expanded_valid = valid.unsqueeze(-1).expand_as(squared_error)
    if not expanded_valid.any():
        return squared_error.new_zeros(())
    return squared_error[expanded_valid].mean()


class PushBoxARPWorkspace(BaseWorkspace):
    include_keys = ["global_step", "optimizer_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: BaseImagePolicy | None = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.optimizer_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Capture training params before resume (load_payload may corrupt cfg via shared refs)
        _num_epochs = int(cfg.training.num_epochs)
        _stop_at_epoch = int(getattr(cfg.training, "stop_at_epoch", 5000))

        if cfg.training.resume:
            resume_optimizer = bool(
                getattr(cfg.training, "resume_optimizer", True)
            )
            resume_path = getattr(cfg.training, 'resume_path', None)
            if resume_path:
                ckpt_path = pathlib.Path(resume_path)
            else:
                ckpt_path = self.get_checkpoint_path()
            if ckpt_path.is_file():
                print(f"Resuming from checkpoint {ckpt_path}")
                import dill as _dill
                payload = torch.load(ckpt_path.open('rb'), pickle_module=_dill)
                # Strip normalizer keys from model state_dicts (normalizer is a plain attr, not a submodule)
                all_norm_sd = {}
                for sd_key in ('model', 'ema_model'):
                    if sd_key not in payload['state_dicts']:
                        continue
                    model_sd = payload['state_dicts'][sd_key]
                    norm_sd = {k[len('normalizer.'):]: v for k, v in model_sd.items() if k.startswith('normalizer.')}
                    clean_sd = {k: v for k, v in model_sd.items() if not k.startswith('normalizer.')}
                    payload['state_dicts'][sd_key] = clean_sd
                    if not all_norm_sd:
                        all_norm_sd = norm_sd
                self.load_payload(
                    payload,
                    exclude_keys=() if resume_optimizer else ("optimizer",),
                )
                if not resume_optimizer:
                    # Keep the recovered model/EMA and epoch, but start the
                    # fine-tuning optimizer and its schedule from the new config.
                    self.optimizer_step = 0
                # Restore cfg from CLI (checkpoint's cfg is stale)
                self.cfg = cfg
                # Re-capture after cfg restore to guard against OmegaConf deepcopy aliasing
                _num_epochs = int(self.cfg.training.num_epochs)
                _stop_at_epoch = int(getattr(self.cfg.training, "stop_at_epoch", 5000))
                print(f"Resumed at epoch {self.epoch}, global_step {self.global_step}; "
                      f"target stop_at_epoch={_stop_at_epoch}, num_epochs={_num_epochs}")
                if all_norm_sd:
                    normalizer = LinearNormalizer()
                    normalizer.load_state_dict(all_norm_sd)
                    self.model.set_normalizer(normalizer)
                    if self.ema_model is not None:
                        self.ema_model.set_normalizer(normalizer)
            else:
                print(f"Checkpoint not found at {ckpt_path}, starting fresh")

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # Normalizer: used for joint-space state/action (non-pixel)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        lr_schedule_kwargs = {}
        if cfg.training.lr_scheduler == "cosine_with_restarts":
            lr_schedule_kwargs["num_cycles"] = cfg.training.lr_num_cycles

        gradient_accumulate_every = int(
            cfg.training.gradient_accumulate_every
        )
        if gradient_accumulate_every <= 0:
            raise ValueError("gradient_accumulate_every must be positive")
        batches_per_epoch = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            batches_per_epoch = min(
                batches_per_epoch,
                int(cfg.training.max_train_steps),
            )
        optimizer_steps_per_epoch = math.ceil(
            batches_per_epoch / gradient_accumulate_every
        )
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=optimizer_steps_per_epoch
            * cfg.training.num_epochs,
            last_epoch=self.optimizer_step - 1,
            **lr_schedule_kwargs,
        )

        ema: EMAModel | None = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        # A resumed W&B run may continue in a new Hydra output directory.
        wandb.config.update({"output_dir": self.output_dir}, allow_val_change=True)
        # Keep W&B's internal _step monotonic across fresh and resumed
        # processes. Training progress is tracked by the explicit global_step
        # metric instead of forcing _step to equal a checkpoint-local value.
        wandb_run.define_metric("global_step")
        wandb_run.define_metric("*", step_metric="global_step")

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            _num_epochs = 2
            _stop_at_epoch = 2

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        # Use the local runtime copy. Debug-mode limits and resumed CLI overrides
        # are applied to ``cfg`` above, not to the constructor-time config.
        train_cfg = cfg.training
        with JsonLogger(log_path) as json_logger:
            for _ in range(_num_epochs):
                self.model.train()
                step_log = {"rollout_sr": 0.0}
                train_losses = list()
                train_batches = len(train_dataloader)
                if train_cfg.max_train_steps is not None:
                    train_batches = min(
                        train_batches,
                        int(train_cfg.max_train_steps),
                    )
                validation_boundaries = validation_batch_boundaries(
                    train_batches, train_cfg.val_every, gradient_accumulate_every, self.epoch)
                self.optimizer.zero_grad(set_to_none=True)
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=train_cfg.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        if batch_idx >= train_batches:
                            break
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss_dict = self.model.compute_loss(batch)
                        raw_loss = sum(raw_loss_dict.values())
                        group_start = (
                            batch_idx // gradient_accumulate_every
                        ) * gradient_accumulate_every
                        group_size = min(
                            gradient_accumulate_every,
                            train_batches - group_start,
                        )
                        loss = raw_loss / group_size
                        loss.backward()

                        should_update = (
                            (batch_idx + 1) % gradient_accumulate_every == 0
                            or (batch_idx + 1) == train_batches
                        )
                        grad_norm = compute_grad_norm(self.model).item()
                        if should_update:
                            self.optimizer.step()
                            lr_scheduler.step()
                            self.optimizer_step += 1
                            if train_cfg.use_ema:
                                ema.step(self.model)
                            self.optimizer.zero_grad(set_to_none=True)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "optimizer_step": self.optimizer_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "grad_norm": grad_norm,
                        }
                        for key, value in raw_loss_dict.items():
                            step_log["train." + key] = value.item()

                        wandb_run.log(step_log)
                        json_logger.log(step_log)
                        self.global_step += 1

                        if batch_idx + 1 in validation_boundaries:
                            policy = self.ema_model if cfg.training.use_ema else self.model
                            metrics = evaluate_loss(
                                policy, val_dataloader, device, train_cfg.max_val_steps,
                                description=f"Validation epoch {self.epoch + (batch_idx + 1) / train_batches:.3f}")
                            metrics.update({"global_step": self.global_step,
                                            "optimizer_step": self.optimizer_step,
                                            "epoch": self.epoch,
                                            "epoch_progress": self.epoch + (batch_idx + 1) / train_batches})
                            wandb_run.log(metrics)
                            json_logger.log(metrics)
                            if batch_idx + 1 == train_batches:
                                step_log.update(metrics)

                step_log["train_loss"] = np.mean(train_losses)

                policy = self.model
                if train_cfg.use_ema:
                    policy = self.ema_model
                policy.eval()

                if self.epoch > 0 and (self.epoch % train_cfg.sample_every) == 0:
                    with torch.no_grad():
                        batch = dict_apply(
                            train_sampling_batch, lambda x: x.to(device, non_blocking=True)
                        )
                        result = policy.predict_action(batch["obs"])
                        target, valid = aligned_action_target(
                            self.model,
                            batch,
                        )
                        mse = masked_action_mse(
                            result["action_pred"],
                            target,
                            valid,
                        )
                        step_log["train_action_mse_error"] = mse.item()

                # Persist the next epoch so resuming does not repeat this completed one.
                self.epoch += 1
                if (self.epoch % train_cfg.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    # Subclass (e.g. ThreadingARPWorkspace) may store a rollout
                    # success rate during save_checkpoint and expose it here.
                    rollout_sr = getattr(self, '_last_rollout_sr', None)
                    if rollout_sr is not None:
                        step_log['rollout_sr'] = rollout_sr
                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    monitor_key = cfg.checkpoint.topk.monitor_key
                    if monitor_key in metric_dict:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                    else:
                        print(
                            f"Skipping top-k checkpoint at epoch {self.epoch}: "
                            f"metric {monitor_key!r} is unavailable"
                        )

                wandb_run.log(step_log)
                json_logger.log(step_log)

                if self.epoch == _stop_at_epoch:
                    print("Reached max epoch limit")
                    break
