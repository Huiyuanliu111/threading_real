"""Robot-free full-model smoke check for native RGB, using pi05_base weights."""
import json
import os
from pathlib import Path
import time

from run import setup


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-views', choices=['both', 'cam1'], default='both')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    settings = json.loads((root/'checkpoints/pi05_threading_lora/threading_lora_tcp6_30hz_h50_best_v4.json').read_text())
    settings.update(mode='vision_lora_action_full', vision_lora_rank=16, image_profile='native640',
                    freeze_vision=True, freeze_language=True, camera_views=args.camera_views)
    config = setup(settings)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx, traverse_util
    from openpi.models import model as model_lib
    from openpi.policies import policy as policy_lib
    from openpi import transforms as T
    from openpi.shared import normalize
    t = time.monotonic()
    template = nnx.eval_shape(lambda: config.model.create(jax.random.key(0)))
    ref = traverse_util.flatten_dict(nnx.state(template, nnx.Param).to_pure_dict())
    weights_path = root/'.cache/openpi/openpi-assets/checkpoints/pi05_base/params'
    weights = traverse_util.flatten_dict(model_lib.restore_params(weights_path, dtype=jnp.bfloat16))
    for key, value in ref.items():
        if key not in weights:
            assert 'lora_' in '/'.join(key), key
            weights[key] = jnp.zeros(value.shape, dtype=jnp.bfloat16)
    model = config.model.load(traverse_util.unflatten_dict(weights))
    dc = config.data.create(config.assets_dirs, config.model)
    norm_stats = normalize.load(root/'assets/threading_lora_tcp6_30hz_h50_best_v4/pi05_threading_lora/threading_real/threading_tcp6_nosmooth_30hz')
    policy = policy_lib.Policy(model,
        transforms=[*dc.data_transforms.inputs, T.Normalize(norm_stats, use_quantiles=dc.use_quantile_norm),
                    *dc.model_transforms.inputs],
        output_transforms=[*dc.model_transforms.outputs, T.Unnormalize(norm_stats, use_quantiles=dc.use_quantile_norm),
                           *dc.data_transforms.outputs], sample_kwargs={'num_steps': 10})
    inputs = np.load(root/'outputs/input_native640_examples/inputs.npz')
    payload = dict(front=inputs['front'], side=inputs['side'], state=inputs['state'],
                   prompt='insert the grasped block through the needle')
    if args.camera_views == 'cam1':
        payload.pop('front')
    print('Loaded base weights; beginning full 644x490 inference', flush=True)
    noise = np.random.default_rng(42).normal(size=(50,32)).astype(np.float32)
    result = policy.infer(payload, noise=noise)
    actions = np.asarray(result['actions'])
    assert actions.shape == (50, 6) and np.isfinite(actions).all()
    elapsed = time.monotonic() - t
    # Verify the real policy transform path preserves every source RGB pixel.
    import copy
    obs = policy._input_transform(copy.deepcopy(payload))
    expected = [('left_wrist_0_rgb', inputs['side'])]
    if args.camera_views == 'both':
        expected.append(('base_0_rgb', inputs['front']))
    for key, source in expected:
        np.testing.assert_array_equal(obs['image'][key], source)
    if args.camera_views == 'cam1':
        assert set(obs['image']) == {'left_wrist_0_rgb'}
        changed = policy.infer({**payload, 'front': np.full_like(inputs['front'], 255)}, noise=noise)
        np.testing.assert_array_equal(actions, np.asarray(changed['actions']))
    record = dict(weights='pi05_base with zero vision LoRA deltas, not task-finetuned',
                  model_input_hw=[490,644], tokens_per_camera=1610, physical_action_shape=list(actions.shape),
                  camera_views=args.camera_views, cam3_invariance_checked=args.camera_views=='cam1',
                  actions_finite=True, load_compile_infer_seconds=elapsed, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    output=root/('outputs/cam1_native640_setup' if args.camera_views=='cam1' else 'outputs/native640_setup');output.mkdir(exist_ok=True)
    (output/'model_smoke.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
