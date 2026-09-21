import unittest

import numpy as np

from transforms import ACTION_NAMES, FRONT, SIDE, STATE_NAMES, ThreadingInputs, ThreadingOutputs, parse_image, validate_info


class TransformTests(unittest.TestCase):
    def test_dataset_requires_30hz(self):
        info = {"fps": 30, "features": {
            "observation.state": {"shape": [9], "names": STATE_NAMES},
            "action": {"shape": [6], "names": ACTION_NAMES}, FRONT: {}, SIDE: {},
        }}
        validate_info(info)
        info["fps"] = 15
        with self.assertRaises(ValueError):
            validate_info(info)

    def test_camera_slots_and_physical_actions(self):
        actions = np.arange(300, dtype=np.float32).reshape(50, 6)
        sample = {"front": np.ones((3, 12, 16), np.float32),
                  "side": np.zeros((12, 16, 3), np.uint8),
                  "state": np.arange(9), "actions": actions, "prompt": b"thread"}
        output = ThreadingInputs()(sample)
        np.testing.assert_array_equal(output["actions"][:, :6], actions[:, :6])
        np.testing.assert_array_equal(ThreadingInputs()(sample)["actions"], actions)
        self.assertEqual(actions[0, 5], 5)  # Dataset is never modified in place.
        self.assertTrue((output["image"]["base_0_rgb"] == 255).all())
        self.assertFalse(output["image_mask"]["right_wrist_0_rgb"])
        self.assertTrue(output["image_mask"]["left_wrist_0_rgb"])
        self.assertEqual(output["prompt"], "thread")

    def test_output_unpadding_and_gripper(self):
        output = ThreadingOutputs()({"actions": np.ones((50, 32))})["actions"]
        self.assertEqual(output.shape, (50, 6))
        np.testing.assert_array_equal(output[:, :6], 1)
        np.testing.assert_array_equal(ThreadingOutputs()({"actions": np.ones((50, 32))})["actions"], 1)

    def test_old_gripper_schema_rejected(self):
        sample = {"front": np.zeros((12,16,3),np.uint8), "side": np.zeros((12,16,3),np.uint8),
                  "state": np.zeros(10), "actions": np.zeros((50,7))}
        with self.assertRaises(ValueError):
            ThreadingInputs()(sample)
        sample["state"] = np.zeros(9)
        with self.assertRaises(ValueError):
            ThreadingInputs()(sample)

    def test_bad_image_rejected(self):
        with self.assertRaises(ValueError):
            parse_image(np.full((3, 12, 16), np.nan))
        with self.assertRaises(ValueError):
            parse_image(np.ones((12, 16)))


if __name__ == "__main__":
    unittest.main()
