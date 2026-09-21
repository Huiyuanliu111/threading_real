"""Run with the pinned OpenPI src and scripts on PYTHONPATH."""
import dataclasses
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('openpi'), 'requires OpenPI environment')
class IntegrationTests(unittest.TestCase):
    def test_action_only_lora_and_historical_freeze_filters(self):
        import math
        import jax
        from flax import nnx
        from config import make_config
        from run import parser
        settings = vars(parser().parse_args(['train', '--mode', 'lora']))
        config = make_config(settings)
        model = nnx.eval_shape(lambda: config.model.create(jax.random.key(0)))
        trainable = nnx.state(model, config.trainable_filter)
        count = lambda state: sum(math.prod(v.shape) for v in jax.tree.leaves(state))
        self.assertEqual(count(trainable), 24_284_192)
        for path, value in jax.tree_util.tree_leaves_with_path(trainable):
            name = '/'.join(str(getattr(p, 'key', getattr(p, 'name', p))) for p in path)
            self.assertNotIn('/img/', name)
            if '/llm/' in name:
                self.assertIn('_1', name)
                self.assertIn('lora', name)
        settings.pop('freeze_vision')
        settings.pop('freeze_language')
        historical = make_config(settings)
        self.assertEqual(count(nnx.state(model, historical.trainable_filter)), 466_957_072)

    def test_periodic_checkpoint_retention_and_resume(self):
        import jax.numpy as jnp
        import optax
        from openpi.training import checkpoints
        from openpi.training.utils import TrainState
        from validation import MetricsSaver
        import flax.nnx as nnx
        model = nnx.Dict(w=nnx.Param(jnp.array([1.])))
        graph, params = nnx.split(model)
        model.w.value = jnp.array([2.])
        state = TrainState(step=0, params=params, model_def=graph,
                           tx=optax.sgd(.1), opt_state=(), ema_decay=.99,
                           ema_params=nnx.state(model))
        class Loader:
            def data_config(self):
                from types import SimpleNamespace
                return SimpleNamespace(norm_stats=None, asset_id=None)
        from train_validated import create_checkpoint_manager as manager
        with tempfile.TemporaryDirectory() as path:
            with manager(path) as m:
                for step, loss in [(2000, 3.), (4000, 4.), (6000, 2.), (8000, 2.), (10000, 5.)]:
                    updated = dataclasses.replace(state, step=step)
                    checkpoints.save_state(MetricsSaver(m, loss), updated, Loader(), step)
                    m.wait_until_finished()
                self.assertEqual(list(m.all_steps()), [2000, 4000, 6000, 8000, 10000])
                self.assertEqual({p.name for p in Path(path).iterdir() if p.name.isdigit()},
                                 {'2000', '4000', '6000', '8000', '10000'})
            with manager(path) as m:
                self.assertEqual(m.metrics(m.latest_step())['val_loss'], 5.)
                restored = checkpoints.restore_state(m, state, Loader())
                self.assertEqual(int(restored.step), 10000)
                np.testing.assert_array_equal(restored.ema_params['w'].value, [2.])

    def test_eval_ema_no_augmentation_and_full_tail(self):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        from openpi.training import sharding
        from openpi.training.utils import TrainState
        from validation import eval_loss, evaluate
        class Model(nnx.Module):
            def __init__(self):
                self.weight = nnx.Param(jnp.array(1.))
            def compute_loss(self, rng, observation, actions, train=False):
                assert not train
                return actions[..., 0] + self.weight.value + jax.random.uniform(rng, ())
        model = Model()
        graph, params = nnx.split(model)
        model.weight.value = jnp.array(10.)
        from openpi.shared import array_typing as at
        with at.disable_typechecking():
            state = TrainState(step=0, params=params, model_def=graph, tx=None, opt_state=(),
                           ema_decay=.99, ema_params=nnx.state(model))
        mesh = sharding.make_mesh(1)
        ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        def batch(xs):
            n = len(xs)
            return {'state': np.zeros((n, 32), np.float32), 'image': {}, 'image_mask': {},
                    'actions': np.asarray(xs, np.float32).reshape(n, 1, 1)}
        batches = [batch([1, 2]), batch([3])]
        fn = jax.jit(eval_loss)
        first = evaluate(state, batches, fn, 43, 2, ds, mesh)
        second = evaluate(state, batches, fn, 43, 2, ds, mesh)
        self.assertEqual(first, second)
        noise = [float(jax.random.uniform(jax.random.fold_in(jax.random.key(43), i), ())) for i in range(2)]
        self.assertAlmostEqual(first, 12 + (2 * noise[0] + noise[1]) / 3, places=5)


if __name__ == '__main__':
    unittest.main()
