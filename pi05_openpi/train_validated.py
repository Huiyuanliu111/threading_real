"""Official OpenPI train_step with independent validation and periodic checkpoints."""
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


def create_checkpoint_manager(directory, max_to_keep=None):
    return ocp.CheckpointManager(directory, item_handlers={
        'assets': checkpoints.CallbackHandler(),
        'train_state': ocp.PyTreeCheckpointHandler(),
        'params': ocp.PyTreeCheckpointHandler(),
    }, options=ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep, keep_period=None,
        best_fn=(lambda metrics: metrics['val_loss'] if metrics['val_loss'] is not None else math.inf)
                if max_to_keep is None else None,
        best_mode='min',
        create=False, async_options=ocp.AsyncOptions(timeout_secs=7200)))


def main(config, settings):
    upstream.init_logging()
    if jax.process_count() != 1:
        raise ValueError('This entry point supports single-process, multi-GPU training')
    eval_interval = settings.get('eval_interval', config.save_interval)
    evaluation_scope = 'train_reconstruction' if settings.get('overfit_episodes', 0) else 'held_out'
    eval_metric = 'train_eval_loss' if evaluation_scope == 'train_reconstruction' else 'val_loss'
    if eval_interval < 1 or config.save_interval % eval_interval:
        raise ValueError('save_interval must be a multiple of the positive eval_interval')
    directory = config.checkpoint_dir
    if directory.exists() and not config.resume:
        raise FileExistsError(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manager = create_checkpoint_manager(directory, settings.get('max_checkpoints'))
    try:
        resuming = config.resume and manager.latest_step() is not None
        best = math.inf
        if resuming:
            history = directory / 'validation.jsonl'
            if history.exists():
                for line in history.read_text().splitlines():
                    row = json.loads(line)
                    if row['step'] <= manager.latest_step() and row[eval_metric] is not None:
                        best = min(best, row[eval_metric])
        if config.wandb_enabled:
            os.environ["WANDB_MODE"] = "online"
        upstream.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        if config.wandb_enabled:
            wandb.config.update({'evaluation_scope': evaluation_scope,
                                'overfit_episodes': settings.get('overfit_episodes', 0),
                                'validation_split': settings.get('validation_split')})
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
            if step <= 5 or step % config.log_interval == 0:
                metrics = {k: float(v) for k, v in jax.device_get(info).items()}
                logging.info('Step %d: %s', step, metrics)
                wandb.log(metrics, step=step, commit=bool(step % eval_interval))
            if step % eval_interval:
                continue
            loss = evaluate(state, validation, val_fn, config.seed + 1, config.batch_size, data_sharding, mesh)
            improved = is_improvement(loss, best)
            record = {'step': step, eval_metric: loss if math.isfinite(loss) else None,
                      'evaluation_scope': evaluation_scope,
                      'improved': improved, 'evaluated_weights': 'ema' if config.ema_decay is not None else 'params'}
            logging.info('Evaluation: %s', record)
            wandb.log({eval_metric: loss}, step=step, commit=True)
            if improved:
                best = loss
            with (directory / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            if step % config.save_interval == 0:
                checkpoints.save_state(MetricsSaver(manager, record[eval_metric]), state, training, step)
                manager.wait_until_finished()
        manager.wait_until_finished()
    finally:
        manager.close()
        wandb.finish()
