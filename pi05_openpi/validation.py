"""Episode-held-out data and reproducible best-checkpoint validation."""
import json
import math
from pathlib import Path

import numpy as np


def episode_split(root, fraction, seed, overfit_episodes=0):
    episodes = [json.loads(line) for line in (Path(root) / 'meta/episodes.jsonl').read_text().splitlines()]
    episodes.sort(key=lambda row: row['episode_index'])
    if isinstance(overfit_episodes, bool) or not isinstance(overfit_episodes, int) or not 0 <= overfit_episodes <= len(episodes):
        raise ValueError('overfit_episodes must be an integer between 0 and the dataset episode count')
    if not overfit_episodes and (not 0 < fraction < 1 or len(episodes) < 2):
        raise ValueError('Validation requires >=2 episodes and 0 < val_fraction < 1')
    ids = [row['episode_index'] for row in episodes]
    if ids != list(range(len(ids))) or any(row['length'] <= 0 for row in episodes):
        raise ValueError('Expected contiguous episode IDs and positive episode lengths')
    if overfit_episodes:
        selected = sorted(np.random.default_rng(seed).permutation(ids)[:overfit_episodes].tolist())
        selected_set = set(selected)
        frames = []
        offset = 0
        for row in episodes:
            if row['episode_index'] in selected_set:
                frames.extend(range(offset, offset + row['length']))
            offset += row['length']
        manifest = {'seed': seed, 'overfit_episodes': overfit_episodes,
                    'evaluation_scope': 'train_reconstruction',
                    'train_episodes': selected, 'val_episodes': selected.copy(),
                    'train_frames': len(frames), 'val_frames': len(frames),
                    'episode_lengths': [row['length'] for row in episodes]}
        return manifest, {'train': frames, 'val': frames.copy()}
    count = max(1, min(len(ids) - 1, math.ceil(len(ids) * fraction)))
    held_out = set(np.random.default_rng(seed).permutation(ids)[:count].tolist())
    indices = {'train': [], 'val': []}
    offset = 0
    for row in episodes:
        part = 'val' if row['episode_index'] in held_out else 'train'
        indices[part].extend(range(offset, offset + row['length']))
        offset += row['length']
    manifest = {'seed': seed, 'val_fraction': fraction,
                'train_episodes': [i for i in ids if i not in held_out],
                'val_episodes': sorted(held_out),
                'train_frames': len(indices['train']), 'val_frames': len(indices['val']),
                'episode_lengths': [row['length'] for row in episodes]}
    return manifest, indices


def split_from_settings(settings):
    return episode_split(settings['dataset_root'], settings['val_fraction'], settings['seed'],
                         settings.get('overfit_episodes', 0))


def is_improvement(loss, best):
    return math.isfinite(loss) and loss < best


def raw_subset(config, settings, part):
    from openpi.training import data_loader as loader
    from torch.utils.data import Subset
    _, indices = split_from_settings(settings)
    data_config = config.data.create(config.assets_dirs, config.model)
    raw = loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    return data_config, Subset(raw, indices[part])


def finite_loader(dataset, batch_size, workers):
    """One complete deterministic pass, including the short final batch."""
    from torch.utils.data import DataLoader
    from openpi.training.data_loader import _collate_fn, _worker_init_fn
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=workers, collate_fn=_collate_fn, worker_init_fn=_worker_init_fn,
                      multiprocessing_context='spawn' if workers else None,
                      persistent_workers=workers > 0)


def compute_norm(config, settings):
    from compute_norm_stats import RemoveStrings
    from openpi.shared import normalize
    from openpi.training import data_loader as loader
    from tqdm import tqdm
    data_config, raw = raw_subset(config, settings, 'train')
    dataset = loader.TransformedDataset(raw, [*data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs, RemoveStrings()])
    stats = {key: normalize.RunningStats() for key in ('state', 'actions')}
    for batch in tqdm(finite_loader(dataset, config.batch_size, config.num_workers), desc='Training-only stats'):
        for key, running in stats.items():
            running.update(np.asarray(batch[key]))
    destination = config.assets_dirs / data_config.asset_id
    normalize.save(destination, {key: value.get_statistics() for key, value in stats.items()})
    split, _ = split_from_settings(settings)
    (destination / 'provenance.json').write_text(json.dumps({
        'split': split, 'horizon': config.model.action_horizon, 'fps': 30,
        'all_train_frames_included': True, 'gripper_removed': True, 'physical_action_dim': 6, 'physical_state_dim': 9,
    }, indent=2) + '\n')


def validate_norm(config, settings):
    destination = config.assets_dirs / config.data.create(config.assets_dirs, config.model).asset_id
    provenance = json.loads((destination / 'provenance.json').read_text())
    split, _ = split_from_settings(settings)
    expected = {'split': split, 'horizon': config.model.action_horizon, 'fps': 30,
                'all_train_frames_included': True, 'gripper_removed': True, 'physical_action_dim': 6, 'physical_state_dim': 9}
    if provenance != expected:
        raise ValueError('Normalization provenance differs: rerun norm for this experiment')


def eval_loss(state, rng, batch):
    import flax.nnx as nnx
    import jax.numpy as jnp
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, actions = batch
    losses = model.compute_loss(rng, observation, actions, train=False)
    return jnp.mean(losses, axis=tuple(range(1, losses.ndim)))


def evaluate(state, dataset_loader, eval_step, seed, batch_size, data_sharding, mesh):
    import jax
    from openpi.models.model import Observation
    from openpi.training import sharding
    total = 0.0
    count = 0
    # Identical RNG per batch at every checkpoint: same noise/time samples, no augmentation.
    for index, batch in enumerate(dataset_loader):
        actual = len(batch['actions'])
        batch = jax.tree.map(lambda x: np.concatenate([x, np.repeat(x[-1:], batch_size - actual, axis=0)])
                             if actual < batch_size else x, batch)
        batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), batch)
        with sharding.set_mesh(mesh):
            losses = eval_step(state, jax.random.fold_in(jax.random.key(seed), index),
                               (Observation.from_dict(batch), batch['actions']))
        total += float(np.asarray(losses)[:actual].sum(dtype=np.float64))
        count += actual
    if not count:
        raise ValueError('Empty validation dataset')
    return total / count


class MetricsSaver:
    """Attach selection metrics while using upstream checkpoint serialization."""
    def __init__(self, manager, loss):
        self.manager, self.loss = manager, loss

    def save(self, step, items):
        return self.manager.save(step, items=items, metrics={'val_loss': self.loss})
