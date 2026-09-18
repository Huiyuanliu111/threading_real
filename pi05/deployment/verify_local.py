#!/usr/bin/env python3
"""Validate downloaded models and preprocessing without hardware or GPU access."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
from threading_real.pi05.visual.crop import load


def main():
    torch.set_num_threads(1)
    manifest = json.loads((ROOT / 'outputs/deployment_preparation_20260918/download_manifest.json').read_text())
    dataset = LeRobotDataset('threading_real/threading_crop_15hz', root=ROOT / 'data/threading_crop_15hz', video_backend='pyav')
    results = {}
    for name, files in manifest.items():
        checkpoint = ROOT / 'outputs' / name / 'checkpoints/005000/pretrained_model'
        for relative, expected in files.items():
            path = checkpoint / relative
            digest = hashlib.sha256()
            with path.open('rb') as file:
                for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            assert path.stat().st_size == expected['bytes'], path
            assert digest.hexdigest() == expected['sha256'], path
        config = PreTrainedConfig.from_pretrained(checkpoint)
        assert config.chunk_size == (20 if 'chunk20' in name else 10)
        crop = load(checkpoint / 'visual_preprocessing.json')
        assert crop == load(ROOT / 'data/threading_crop_15hz/meta/visual_preprocessing.json')
        # Build shapes only: no model allocation on CPU/GPU and no forward pass.
        config.device = 'meta'
        with torch.device('meta'):
            policy = PI05Policy(config)
        config.device = 'cpu'
        expected_state = policy.state_dict()
        with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as weights:
            actual_keys = set(weights.keys())
            assert actual_keys == set(expected_state), (actual_keys - set(expected_state), set(expected_state) - actual_keys)
            for key, value in expected_state.items():
                assert tuple(weights.get_slice(key).get_shape()) == tuple(value.shape), key
        pre, post = make_pre_post_processors(config, str(checkpoint), preprocessor_overrides={'device_processor': {'device': 'cpu'}})
        batch = pre(dict(dataset[60]))
        for key in config.image_features:
            assert batch[key].isfinite().all() and batch[key].min() >= 0 and batch[key].max() <= 1
        action = post(torch.zeros(1, config.chunk_size, 7))
        assert action.shape == (1, config.chunk_size, 7) and action.isfinite().all()
        results[name] = dict(sha256_files=len(files), tensor_shapes=len(expected_state), chunk_size=config.chunk_size,
                             preprocessing='passed', model_forward='not run: GPU occupied by live robot task')
        print(name, results[name], flush=True)
        del policy, expected_state, pre, post
    (ROOT / 'outputs/deployment_preparation_20260918/local_validation.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
