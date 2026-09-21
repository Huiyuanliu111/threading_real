import dataclasses
import unittest
import numpy as np

from native_images import MODEL_HW, PadNativeImages, pad_native_rgb


class NativeImageTests(unittest.TestCase):
    def test_pixel_exact_padding_and_reject_downscaled_input(self):
        rgb = np.random.default_rng(42).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        padded = pad_native_rgb(rgb)
        self.assertEqual(padded.shape, (490, 644, 3))
        np.testing.assert_array_equal(padded[5:485, 2:642], rgb)
        self.assertFalse(padded[:5].any())
        self.assertFalse(padded[-5:].any())
        self.assertFalse(padded[:, :2].any())
        self.assertFalse(padded[:, -2:].any())
        self.assertIs(PadNativeImages()({'image': {'x': padded}})['image']['x'], padded)
        with self.assertRaises(ValueError):
            pad_native_rgb(np.zeros((224, 224, 3), np.uint8))

    def test_model_shapes_weights_and_no_internal_resize(self):
        import jax
        from flax import nnx, traverse_util
        from config import make_config
        from run import parser
        from pi0_native import _preprocess_native, NativePi0
        settings = vars(parser().parse_args(['train', '--image-profile', 'native640']))
        cfg = make_config(settings)
        model = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))
        self.assertIsInstance(model, NativePi0)
        shapes = traverse_util.flatten_dict(nnx.state(model, nnx.Param).to_pure_dict(), sep='/')
        self.assertEqual(shapes['PaliGemma/img/pos_embedding'].shape, (1, 256, 1152))
        obs = cfg.model.fake_obs()
        self.assertEqual(obs.images['base_0_rgb'].shape, (1, 490, 644, 3))
        checked = _preprocess_native(obs)
        self.assertEqual(set(checked.images), {'base_0_rgb', 'left_wrist_0_rgb'})
        self.assertIs(checked.images['base_0_rgb'], obs.images['base_0_rgb'])
        # Both retained camera slots produce 35x46 image tokens without losing border pixels.
        vision = nnx.eval_shape(lambda m, x: m(x, train=False), model.PaliGemma.img, obs.images['base_0_rgb'])
        self.assertEqual(vision[0].shape, (1, 1610, 2048))
        loss = nnx.eval_shape(lambda m, o, a: m.compute_loss(jax.random.key(1), o, a, train=True),
                             model, obs, cfg.model.fake_act())
        self.assertEqual(loss.shape, (1, 50))

    def test_position_interpolation_preserves_checkpoint_shape_and_gradients(self):
        import jax
        import jax.numpy as jnp
        import siglip_lora
        from flax import traverse_util
        kwargs = dict(variant='mu/4', num_classes=16, pool_type='none', scan=True,
                      lora_rank=4, lora_alpha=4., dtype_mm='float32')
        base = siglip_lora.Module(**kwargs)
        native = siglip_lora.Module(**kwargs, pretrained_grid=(2, 2))
        params = base.init(jax.random.key(0), jnp.zeros((1, 8, 8, 3)))
        x = jax.random.normal(jax.random.key(1), (1, 12, 20, 3))
        result = native.apply(params, x)
        self.assertEqual(result[0].shape, (1, 15, 16))
        grads = jax.grad(lambda p: jnp.mean(native.apply(p, x)[1]['encoded'] ** 3))(params)
        leaves = traverse_util.flatten_dict(grads)
        self.assertTrue(any(float(jnp.linalg.norm(v)) > 0 for k, v in leaves.items() if k[-1] == 'lora_b'))


if __name__ == '__main__':
    unittest.main()
