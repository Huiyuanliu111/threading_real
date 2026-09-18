#!/usr/bin/env python3
"""Wait for a completed crop checkpoint, then run and summarize sensitivity."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--timeout-hours', type=float, default=12)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = root / 'pi05/outputs/visual_sensitivity'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'after_checkpoint.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = output / 'status.json'

    def status(state: str, **details) -> None:
        record = dict(state=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(status_path)
        print(json.dumps(record), flush=True)

    try:
        checkpoints = root / 'pi05/outputs/threading_crop_v1/checkpoints'
        checkpoint = checkpoints / '005000/pretrained_model'
        status('waiting_for_checkpoint', checkpoint=str(checkpoint))
        deadline = time.monotonic() + args.timeout_hours * 3600
        while True:
            # LeRobot updates this symlink only after save_checkpoint returns.
            last = checkpoints / 'last'
            if last.is_symlink() and last.resolve() == checkpoint.parent:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Checkpoint did not complete within the configured wait')
            try:
                os.kill(args.training_pid, 0)
            except ProcessLookupError:
                # Recheck to handle completion concurrent with launcher exit.
                if last.is_symlink() and last.resolve() == checkpoint.parent:
                    break
                raise RuntimeError('Training exited without a completed step-5000 checkpoint')
            time.sleep(30)

        from safetensors import safe_open

        required = ['config.json', 'model.safetensors', 'policy_preprocessor.json',
                    'policy_postprocessor.json', 'visual_preprocessing.json',
                    'state_representation.json']
        for name in required:
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
        with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as weights:
            if not list(weights.keys()):
                raise RuntimeError('Checkpoint contains no tensors')
        visual = json.loads((checkpoint / 'visual_preprocessing.json').read_text())
        expected = json.loads((root / 'data/threading_crop_15hz/meta/visual_preprocessing.json').read_text())
        if visual != expected:
            raise ValueError('Checkpoint crop metadata differs from the dataset')
        status('running_sensitivity', checkpoint=str(checkpoint))
        crop_path = output / 'crop_005000.json'
        command = [sys.executable, '-u', str(root / 'pi05/diagnostics/condition_sensitivity.py'),
                   '--checkpoint', str(checkpoint), '--dataset-root', str(root / 'data/threading_crop_15hz'),
                   '--repo-id', 'threading_real/threading_crop_15hz', '--pairs', '6',
                   '--seed', '20260909', '--output', str(crop_path)]
        with (output / 'crop_005000.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        baseline = json.loads((output / 'full_image_001000.json').read_text())
        crop = json.loads(crop_path.read_text())
        keys = ['base_index', 'donor_index', 'base_episode', 'donor_episode']
        samples = lambda result: [[record[key] for key in keys] for record in result['records']]
        if baseline['seed'] != crop['seed'] or samples(baseline) != samples(crop):
            raise ValueError('Sensitivity runs used different samples or noise seeds')
        lines = ['# Visual sensitivity comparison', '',
                 'Six matched pairs; seed 20260909. State and frame indices match across datasets.', '',
                 '| Model | Image/noise RMSE ratio | Image-swap endpoint (mm) | Noise-swap endpoint (mm) |',
                 '|---|---:|---:|---:|']
        for name, result in [('Full image, step 1000', baseline), ('Crop, step 5000', crop)]:
            ratio = result['condition_to_noise_ratio']['image_swap']
            image_mm = result['mean']['image_swap']['translation_endpoint_m'] * 1000
            noise_mm = result['mean']['noise_swap']['translation_endpoint_m'] * 1000
            lines.append(f'| {name} | {ratio:.4f} | {image_mm:.3f} | {noise_mm:.3f} |')
        lines += ['', 'This measures dependence on conditioning, not task success or correct visual grounding.',
                  'Training steps/configurations and action data differ; this is not an isolated crop ablation.',
                  'The six pairs include training and validation episodes; this is a small diagnostic sample.',
                  'See both JSON files for state swaps, per-dimension differences and repeat determinism.']
        (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
        status('complete', checkpoint=str(checkpoint), report=str(output / 'comparison.md'))
    except Exception as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
