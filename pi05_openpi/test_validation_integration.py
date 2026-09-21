"""Run with the pinned OpenPI src and scripts on PYTHONPATH."""
import dataclasses
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('openpi'), 'requires OpenPI environment')
class IntegrationTests(unittest.TestCase):
    def test_best_checkpoint_retention_and_resume(self):
        import jax.numpy as jnp
        import optax
        import orbax.checkpoint as ocp
        from openpi.training import checkpoints
        from openpi.training.utils import TrainState
        from validation import MetricsSaver, is_improvement
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
        def manager(path):
            return ocp.CheckpointManager(path, item_handlers={
                'assets': checkpoints.CallbackHandler(),
                'train_state': ocp.PyTreeCheckpointHandler(),
                'params': ocp.PyTreeCheckpointHandler(),
            }, options=ocp.CheckpointManagerOptions(max_to_keep=1, keep_period=None,
                best_fn=lambda m: m['val_loss'], best_mode='min'))
        with tempfile.TemporaryDirectory() as path:
            with manager(path) as m:
                best = math.inf
                for step, loss in [(2000, 3.), (4000, 4.), (6000, 2.), (8000, 2.), (10000, 5.)]:
                    if is_improvement(loss, best):
                        updated = dataclasses.replace(state, step=step)
                        checkpoints.save_state(MetricsSaver(m, loss), updated, Loader(), step)
                        m.wait_until_finished()
                        best = loss
                self.assertEqual(list(m.all_steps()), [6000])
                self.assertEqual(sorted(p.name for p in Path(path).iterdir() if p.name.isdigit()), ['6000'])
            with manager(path) as m:
                self.assertEqual(m.metrics(m.latest_step())['val_loss'], 2.)
                restored = checkpoints.restore_state(m, state, Loader())
                self.assertEqual(int(restored.step), 6000)
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
