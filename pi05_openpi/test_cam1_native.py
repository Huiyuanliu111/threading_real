"""Cam1-only native input regression checks; no weights, cameras, or network."""
import dataclasses
import math
import unittest
from unittest import mock

import numpy as np

from native_images import PadNativeImages
from transforms import SIDE, ThreadingInputs


CAM1_KEY = "left_wrist_0_rgb"


def cam1_config():
    from config import make_config
    from run import parser

    settings = vars(parser().parse_args([
        "train", "--image-profile", "native640", "--camera-views", "cam1",
    ]))
    return make_config(settings)


class Cam1InputTests(unittest.TestCase):
    def test_missing_or_invalid_cam3_is_ignored_and_cam1_pixels_are_preserved(self):
        rgb = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        sample = {"side": rgb, "state": np.zeros(9, dtype=np.float32)}
        transform = ThreadingInputs(camera_views="cam1")
        clean = transform(sample)
        invalid_front = transform({**sample, "front": object()})
        for result in (clean, invalid_front):
            self.assertEqual(set(result["image"]), {CAM1_KEY})
            self.assertEqual(set(result["image_mask"]), {CAM1_KEY})
            self.assertTrue(result["image_mask"][CAM1_KEY])
            padded = PadNativeImages()(result)["image"][CAM1_KEY]
            self.assertEqual(padded.shape, (490, 644, 3))
            np.testing.assert_array_equal(padded[5:485, 2:642], rgb)

    def test_dataset_repack_does_not_require_front(self):
        import config as local_config
        from openpi import transforms as T

        cfg = cam1_config()
        # Tokenization and asset loading are unrelated to camera selection and
        # may download files; exercise the real repack/data transforms offline.
        with mock.patch.object(local_config.C, "ModelTransformFactory") as factory, \
             mock.patch.object(local_config.ThreadingDataConfig, "create_base_config",
                               return_value=local_config.C.DataConfig()):
            factory.return_value.return_value = T.Group(inputs=[T.ResizeImages(224, 224)])
            dc = cfg.data.create(cfg.assets_dirs, cfg.model)
        sample = {
            SIDE: np.zeros((480, 640, 3), dtype=np.uint8),
            "observation.state": np.zeros(9, dtype=np.float32),
            "action": np.zeros((50, 6), dtype=np.float32),
            "prompt": "insert the grasped block through the needle",
        }
        repacked = T.compose(dc.repack_transforms.inputs)(sample)
        self.assertNotIn("front", repacked)
        transformed = T.compose(dc.data_transforms.inputs)(repacked)
        transformed = T.compose(dc.model_transforms.inputs)(transformed)
        self.assertEqual(set(transformed["image"]), {CAM1_KEY})
        self.assertEqual(transformed["image"][CAM1_KEY].shape, (490, 644, 3))


class Cam1ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        from flax import nnx

        cls.cfg = cam1_config()
        cls.model = nnx.eval_shape(lambda: cls.cfg.model.create(jax.random.key(0)))

    def test_single_camera_prefix_loss_and_sampling_shapes(self):
        import jax
        from flax import nnx
        from pi0_native import _preprocess_native

        obs = self.cfg.model.fake_obs()
        self.assertEqual(set(obs.images), {CAM1_KEY})
        self.assertEqual(set(obs.image_masks), {CAM1_KEY})
        self.assertEqual(obs.images[CAM1_KEY].shape, (1, 490, 644, 3))
        self.assertEqual(self.model.image_keys, (CAM1_KEY,))
        # A stray front image must be removed before prefix embedding.
        polluted = dataclasses.replace(obs, images={**obs.images, "base_0_rgb": obs.images[CAM1_KEY]})
        clean = _preprocess_native(polluted, self.model.image_keys)
        self.assertEqual(set(clean.images), {CAM1_KEY})
        self.assertIs(clean.images[CAM1_KEY], obs.images[CAM1_KEY])
        prefix, mask, ar_mask = nnx.eval_shape(lambda m, o: m.embed_prefix(o), self.model, clean)
        tokens = 1610 + self.cfg.model.max_token_len
        self.assertEqual(prefix.shape, (1, tokens, 2048))
        self.assertEqual(mask.shape, (1, tokens))
        self.assertEqual(ar_mask.shape, (tokens,))
        loss = nnx.eval_shape(
            lambda m, o, a: m.compute_loss(jax.random.key(1), o, a, train=True),
            self.model, obs, self.cfg.model.fake_act(),
        )
        self.assertEqual(loss.shape, (1, 50))
        actions = nnx.eval_shape(
            lambda m, o: m.sample_actions(jax.random.key(2), o, num_steps=2), self.model, obs,
        )
        self.assertEqual(actions.shape, (1, 50, 32))

    def test_training_scope_and_parameter_count_are_unchanged(self):
        from flax import nnx, traverse_util

        trainable = traverse_util.flatten_dict(
            nnx.state(self.model, self.cfg.trainable_filter).to_pure_dict(), sep="/",
        )
        counts = {"vision_lora": 0, "action_expert": 0, "projections_time": 0}
        for key, value in trainable.items():
            if key.startswith("PaliGemma/img/"):
                self.assertIn("lora_", key)
                group = "vision_lora"
            elif key.startswith("PaliGemma/llm/"):
                self.assertIn("_1", key)
                self.assertNotIn("lora_", key)
                group = "action_expert"
            else:
                group = "projections_time"
            counts[group] += math.prod(value.shape)
        self.assertEqual(counts, {"vision_lora": 8_695_296, "action_expert": 427_932_672,
                                  "projections_time": 2_165_792})
        self.assertEqual(sum(counts.values()), 438_793_760)


if __name__ == "__main__":
    unittest.main()
