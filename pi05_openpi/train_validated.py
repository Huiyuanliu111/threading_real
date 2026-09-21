"""Official OpenPI train_step plus episode validation and best-only checkpoints."""
import functools
import json
import logging
import math
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
import wandb

import train as upstream
from openpi.training import checkpoints, data_loader, sharding
from validation import (MetricsSaver, eval_loss, evaluate, finite_loader, is_improvement, raw_subset)


def main(config, settings):
    upstream.init_logging()
    if jax.process_count() != 1:
        raise ValueError('This entry point supports single-process, multi-GPU training')
    directory = config.checkpoint_dir
    if directory.exists() and not config.resume:
        raise FileExistsError(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manager = ocp.CheckpointManager(directory, item_handlers={
        'assets': checkpoints.CallbackHandler(),
        'train_state': ocp.PyTreeCheckpointHandler(),
        'params': ocp.PyTreeCheckpointHandler(),
    }, options=ocp.CheckpointManagerOptions(
        max_to_keep=1, keep_period=None, best_fn=lambda metrics: metrics['val_loss'],
        best_mode='min', create=False, async_options=ocp.AsyncOptions(timeout_secs=7200)))
    try:
        resuming = config.resume and manager.latest_step() is not None
        best = float(manager.metrics(manager.latest_step())['val_loss']) if resuming else math.inf
        if config.wandb_enabled:
            os.environ["WANDB_MODE"] = "online"
        upstream.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        mesh = sharding.make_mesh(config.fsdp_devices)
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        dc, train_raw = raw_subset(config, settings, 'train')
        _, val_raw = raw_subset(config, settings, 'val')
        train_data = data_loader.transform_dataset(train_raw, dc)
        val_data = data_loader.transform_dataset(val_raw, dc)
        training = data_loader.DataLoaderImpl(dc, data_loader.TorchDataLoader(
            train_data, config.batch_size, sharding=data_sharding, shuffle=True,
            num_workers=config.num_workers, seed=config.seed))
        validation = finite_loader(val_data, config.batch_size, config.num_workers)
        train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
        state, state_sharding = upstream.init_train_state(config, init_rng, mesh, resume=resuming)
        if resuming:
            state = checkpoints.restore_state(manager, state, training)
        jax.block_until_ready(state)
        step_fn = jax.jit(functools.partial(upstream.train_step, config),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        val_fn = jax.jit(eval_loss, in_shardings=(state_sharding, replicated, data_sharding),
                         out_shardings=replicated)
        batches = iter(training)
        for step in range(int(state.step) + 1, config.num_train_steps + 1):
            with sharding.set_mesh(mesh):
                state, info = step_fn(train_rng, state, next(batches))
            if step == 1 or step % config.log_interval == 0:
                metrics = {k: float(v) for k, v in jax.device_get(info).items()}
                logging.info('Step %d: %s', step, metrics)
                wandb.log(metrics, step=step)
            if step % config.save_interval:
                continue
            loss = evaluate(state, validation, val_fn, config.seed + 1, config.batch_size, data_sharding, mesh)
            improved = is_improvement(loss, best)
            record = {'step': step, 'val_loss': loss if math.isfinite(loss) else None,
                      'improved': improved, 'evaluated_weights': 'ema' if config.ema_decay is not None else 'params'}
            logging.info('Validation: %s', record)
            wandb.log({'val_loss': loss}, step=step)
            if improved:
                checkpoints.save_state(MetricsSaver(manager, loss), state, training, step)
                manager.wait_until_finished()
                best = loss
            with (directory / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
        manager.wait_until_finished()
    finally:
        manager.close()
        wandb.finish()
