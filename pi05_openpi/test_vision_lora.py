"""Compatibility, gradients, and freeze coverage for the v6 training recipe."""
import math
import unittest

from flax import linen as nn, nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import siglip, pi0_config

import siglip_lora
from config import make_config
from run import parser


class VisionLoRATests(unittest.TestCase):
    def test_pretrained_equivalence_and_adapter_gradients(self):
        for scan in (False, True):
            with self.subTest(scan=scan):
                kwargs = dict(variant='mu/4', num_classes=16, pool_type='none',
                              scan=scan, dtype_mm='float32')
                base = siglip.Module(**kwargs)
                adapted = siglip_lora.Module(**kwargs, lora_rank=4, lora_alpha=4.)
                x = jax.random.normal(jax.random.key(0), (1, 8, 8, 3))
                bp = base.init(jax.random.key(1), x)['params']
                bp['head']['kernel'] = jax.random.normal(jax.random.key(3), bp['head']['kernel'].shape) * .01
                ap = adapted.init(jax.random.key(2), x)['params']
                bf, af = map(traverse_util.flatten_dict, (bp, ap))
                self.assertEqual(set(bf), {k for k in af if 'lora_' not in '/'.join(k)})
                for k, v in bf.items():
                    self.assertEqual(v.shape, af[k].shape)
                    af[k] = v
                params = traverse_util.unflatten_dict(af)
                y = base.apply({'params': bp}, x)[0]
                np.testing.assert_allclose(adapted.apply({'params': params}, x)[0], y, atol=1e-6)
                np.testing.assert_allclose(adapted.apply({'params': params}, x)[1]['encoded'],
                                           base.apply({'params': bp}, x)[1]['encoded'], atol=1e-6)
                # Probe encoder features; pretrained image head is initialized to zero.
                def loss(p):
                    return jnp.mean(adapted.apply({'params': p}, x)[1]['encoded'] ** 3)
                gradients = traverse_util.flatten_dict(jax.grad(loss)(params))
                bgrads = [v for k, v in gradients.items() if k[-1] == 'lora_b']
                self.assertEqual(len(bgrads), 6)
                self.assertTrue(all(float(jnp.linalg.norm(v)) > 0 for v in bgrads))
                updated = {k: v - .01 * gradients[k] if 'lora_' in '/'.join(k) else v
                           for k, v in af.items()}
                self.assertNotEqual(float(loss(params)), float(loss(traverse_util.unflatten_dict(updated))))
                for k in bf:
                    np.testing.assert_array_equal(updated[k], bf[k])

    def test_full_model_shapes_and_training_scope(self):
        cfg = make_config(vars(parser().parse_args(['train'])))
        model = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))
        original_cfg = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50)
        original = nnx.eval_shape(lambda: original_cfg.create(jax.random.key(0)))
        def flatten(model, filter):
            return traverse_util.flatten_dict(nnx.state(model, filter).to_pure_dict(), sep='/')
        full = flatten(model, nnx.Param)
        base = flatten(original, nnx.Param)
        self.assertEqual(set(base), {k for k in full if 'lora_' not in k})
        for k in base:
            self.assertEqual(base[k].shape, full[k].shape)
        trainable = flatten(model, cfg.trainable_filter)
        for k in full:
            expected = (k.startswith('PaliGemma/img/') and 'lora_' in k
                        or k.startswith('PaliGemma/llm/') and '_1' in k
                        or not k.startswith('PaliGemma/'))
            self.assertEqual(k in trainable, expected, k)
        self.assertFalse(any('lora' in k for k in full if '/llm/' in k))
        counts = {}
        for k,v in trainable.items():
            group = 'vision_lora' if '/img/' in k else 'action_expert' if '/llm/' in k else 'projections_time'
            counts[group] = counts.get(group, 0) + math.prod(v.shape)
        print('V6_TRAINABLE_PARAMETERS', counts, 'TOTAL', sum(counts.values()), flush=True)
        self.assertEqual(counts['vision_lora'], 8_695_296)
        self.assertEqual(counts['action_expert'], 427_932_672)
        self.assertEqual(sum(counts.values()), 438_793_760)


if __name__ == '__main__':
    unittest.main()
