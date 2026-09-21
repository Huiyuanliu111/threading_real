#!/usr/bin/env python3
"""Paired image interventions for OpenPI; fixed state, prompt, noise and RNG. No robot I/O."""
import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pi05_openpi'))
from run import bind_dataset, setup
from transforms import FRONT, SIDE


def difference_metrics(actions, reference):
    delta = np.asarray(actions, dtype=np.float64) - reference
    return {
        'first_translation_mm': float(np.linalg.norm(delta[0, :3]) * 1000),
        'first_rotation_deg': float(np.linalg.norm(delta[0, 3:6]) * 180 / np.pi),
        'chunk_translation_rms_mm': float(np.sqrt(np.mean(np.sum(delta[:, :3]**2, axis=1))) * 1000),
        'chunk_rotation_rms_deg': float(np.sqrt(np.mean(np.sum(delta[:, 3:6]**2, axis=1))) * 180 / np.pi),
        'summed_translation_difference_mm': float(np.linalg.norm(delta[:, :3].sum(axis=0)) * 1000),
    }


def summarize(records):
    result = {}
    for condition in sorted({r['condition'] for r in records}):
        selected = [r for r in records if r['condition'] == condition]
        metrics = {}
        for key in selected[0]['response']:
            values = np.array([r['response'][key] for r in selected])
            metrics[key] = {'mean': float(values.mean()), 'median': float(np.median(values)),
                            'p90': float(np.quantile(values, .9)), 'max': float(values.max())}
        for key in ('label_translation_rms_mm', 'label_rotation_rms_deg'):
            values = np.array([r[key] for r in selected])
            metrics[key] = {'mean': float(values.mean()), 'median': float(np.median(values))}
        result[condition] = {'count': len(selected), 'metrics': metrics}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--run-config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--openpi-root', type=Path)
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--episodes', type=int, nargs='+')
    p.add_argument('--phases', type=float, nargs='+', default=[.1, .5, .85])
    p.add_argument('--noise-seeds', type=int, default=2)
    p.add_argument('--seed', type=int, default=20260920)
    p.add_argument('--num-steps', type=int, default=10)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.noise_seeds < 2 or args.num_steps < 1 or any(not 0 <= f <= 1 for f in args.phases):
        p.error('Require >=2 noise seeds, positive num-steps, and phases in [0,1]')
    settings = json.loads(args.run_config.read_text())
    for key in ('openpi_root','dataset_root'):
        if getattr(args,key) is not None:
            settings[key] = str(getattr(args,key).resolve())
    if settings.get('physical_action_dim') != 6 or not settings.get('gripper_removed'):
        raise ValueError('Requires the no-gripper 6D OpenPI experiment')
    config = setup(settings)
    info = bind_dataset(settings)
    episodes = args.episodes or settings['validation_split']['val_episodes']
    if len(episodes) < 2 or len(set(episodes)) != len(episodes):
        raise ValueError('Need >=2 distinct episodes for image substitution')
    meta = [json.loads(line) for line in (Path(settings['dataset_root'])/'meta/episodes.jsonl').read_text().splitlines()]
    lengths = {row['episode_index']: row['length'] for row in meta}
    if any(ep not in lengths for ep in episodes):
        raise ValueError('Unknown episode')
    offsets = {}; count = 0
    for row in sorted(meta, key=lambda x: x['episode_index']):
        offsets[row['episode_index']] = count; count += row['length']
    horizon = config.model.action_horizon
    if any(lengths[ep] < horizon for ep in episodes):
        raise ValueError('Episode shorter than action chunk')
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from openpi.policies.policy_config import create_trained_policy
    import jax
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Use the OpenPI CUDA environment')
    dataset = LeRobotDataset(settings['repo_id'], delta_timestamps={'action':[i/info['fps'] for i in range(horizon)]})
    samples = {}
    for ep in episodes:
        for phase in args.phases:
            frame = min(round(phase*(lengths[ep]-1)), lengths[ep]-horizon)
            row = dataset[offsets[ep]+frame]
            obs = {'front': row[FRONT].numpy(), 'side': row[SIDE].numpy(),
                   'state': row['observation.state'].numpy(), 'prompt': row['task']}
            samples[ep,phase] = (frame, obs, row['action'].numpy())
    args.output.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO)
    started = time.monotonic()
    policy = create_trained_policy(config, args.checkpoint, sample_kwargs={'num_steps':args.num_steps})
    load_seconds = time.monotonic()-started
    variants = ['baseline','identical_repeat','swap_front','swap_side','swap_both','black_front','black_side','black_both']
    records, arrays, comparisons = [], {}, []
    baselines = defaultdict(list)
    total = len(episodes)*len(args.phases)*args.noise_seeds*len(variants)
    done = 0
    for ep_index, ep in enumerate(episodes):
        donor = episodes[(ep_index+1)%len(episodes)]
        for phase_index, phase in enumerate(args.phases):
            frame, base, target = samples[ep,phase]
            donor_frame, donor_obs, _ = samples[donor,phase]
            for noise_index in range(args.noise_seeds):
                seed = args.seed + (ep_index*len(args.phases)+phase_index)*args.noise_seeds+noise_index
                noise = np.random.default_rng(seed).standard_normal((horizon,config.model.action_dim)).astype(np.float32)
                baseline = None
                for condition in variants:
                    obs = {**base}
                    if condition.startswith('swap_'):
                        for camera in ('front','side'):
                            if condition in (f'swap_{camera}','swap_both'):
                                obs[camera] = donor_obs[camera]
                    if condition.startswith('black_'):
                        for camera in ('front','side'):
                            if condition in (f'black_{camera}','black_both'):
                                obs[camera] = np.zeros_like(base[camera])
                    # Pin all randomness for each paired intervention, not only the initial flow noise.
                    policy._rng = jax.random.key(seed)
                    call_start = time.monotonic()
                    actions = np.asarray(policy.infer(obs, noise=noise)['actions'])
                    elapsed = time.monotonic()-call_start
                    if actions.shape != (horizon,6) or not np.isfinite(actions).all():
                        raise ValueError(f'Invalid output {actions.shape}')
                    if baseline is None:
                        baseline = actions.copy()
                        baselines[ep,phase].append(baseline)
                    if condition == 'identical_repeat' and not np.allclose(actions,baseline,atol=1e-7,rtol=0):
                        raise AssertionError('Identical input/noise must reproduce the baseline')
                    key = f'ep{ep}_phase{phase_index}_seed{noise_index}_{condition}'
                    arrays[key] = actions
                    errors = difference_metrics(actions,target)
                    record = {'episode':ep,'frame':frame,'phase':phase,'donor_episode':donor,
                              'donor_frame':donor_frame,'noise_seed':seed,'condition':condition,
                              'response':difference_metrics(actions,baseline),
                              'label_translation_rms_mm':errors['chunk_translation_rms_mm'],
                              'label_rotation_rms_deg':errors['chunk_rotation_rms_deg'],
                              'inference_seconds':elapsed,'actions_key':key}
                    records.append(record)
                    with (args.output/'records.jsonl').open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
                    done += 1
                print(f'{done}/{total}: episode={ep}, phase={phase}, seed={noise_index}',flush=True)
            comparisons.append({'episode':ep,'phase':phase,'condition':'noise_only',
                                'response':difference_metrics(baselines[ep,phase][1],baselines[ep,phase][0]),
                                'label_translation_rms_mm':0.,'label_rotation_rms_deg':0.})
    np.savez_compressed(args.output/'actions.npz',**arrays)
    report = {'checkpoint':str(args.checkpoint.resolve()),'run_config':str(args.run_config.resolve()),
              'dataset':settings['dataset_root'],'episodes':episodes,'phases':args.phases,
              'samples':len(episodes)*len(args.phases),'noise_seeds':args.noise_seeds,
              'num_inferences':total,'num_steps':args.num_steps,'fps':info['fps'],'horizon':horizon,
              'load_seconds':load_seconds,'elapsed_seconds':time.monotonic()-started,
              'summary':summarize(records),'noise_only_summary':summarize(comparisons)['noise_only'],
              'method':{'fixed':['state','prompt','flow_noise','rng'],
                        'swap':'next selected validation episode at the same relative phase; state remains from base episode',
                        'black':'zero RGB pixels with image mask still true, not a missing-camera mask',
                        'output':'physical 6D TCP increments, full 50-step chunk; no gripper',
                        'notes':['No training augmentation or robot commands.',
                                 'Image replacement can be inconsistent with the fixed state; black images are out of distribution.',
                                 'Output sensitivity does not prove correct visual grounding or real-world task success.',
                                 'Reference-label error after substitution uses the original base episode actions.',
                                 'Summary combines samples/seeds descriptively; no independence or significance claim.']}}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)
    print('Report:',args.output/'report.json',flush=True)


if __name__ == '__main__':
    main()
