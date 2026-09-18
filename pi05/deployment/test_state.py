import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'remote_controller/src'))
from threading_real.pi05.deployment.state import make_state_adapter
from threading_real.pi05.visual.crop import load, make_observation_adapter
from threading_real.scripts.deployment.joint import make_observation, stack_observations
from remote_controller.robot_kinematics import RobotModel
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors


@pytest.mark.parametrize('run', ['threading_crop_v1', 'threading_crop_chunk20_v1'])
def test_live_state_matches_tcp_fk_and_checkpoint_processor(run):
    checkpoint = ROOT / 'threading_real/pi05/outputs' / run / 'checkpoints/005000/pretrained_model'
    if not checkpoint.exists():
        pytest.skip('Downloaded checkpoint required')
    observation = make_state_adapter(make_observation_adapter(
        make_observation, load(checkpoint / 'visual_preprocessing.json')), checkpoint)
    robot = RobotModel()
    config = PreTrainedConfig.from_pretrained(checkpoint)
    preprocess, _ = make_pre_post_processors(config, str(checkpoint),
        preprocessor_overrides={'device_processor': {'device': 'cpu'}})
    raw = np.zeros((480, 640, 3), np.uint8)
    for q in [[0, -.4, 0, -2, 0, 1.6, .7], [.2, -.7, .3, -1.8, -.2, 1.3, .4]]:
        pose = robot.get_frame_pose(q, 'panda_hand_tcp')['T']
        frame = observation(raw, None, raw, q, .023, 224, tcp_position=pose[:3, 3])
        expected = np.concatenate([pose[:3, 3], pose[:3, 0], pose[:3, 1], [.023]])
        np.testing.assert_allclose(frame.agent_pos.numpy(), expected, atol=2e-7)
        stacked = stack_observations([frame], 'cpu', rgb_keys=('sideview', 'frontview'))
        assert stacked['agent_pos'].shape == (1, 1, 10)
        batch = preprocess({
            'observation.state': stacked['agent_pos'][0, -1],
            'observation.images.exterior_image_2_right': stacked['sideview'][0, -1],
            'observation.images.exterior_image_1_left': stacked['frontview'][0, -1],
            'task': 'insert the grasped block through the needle',
        })
        assert batch['observation.state'].shape == (1, 10)
        assert torch.isfinite(batch['observation.state']).all()


def test_mismatched_metadata_rejected(tmp_path):
    (tmp_path / 'state_representation.json').write_text(json.dumps(
        dict(state_representation='tcp_pose_6d', state_dim=10)))
    (tmp_path / 'config.json').write_text(json.dumps(
        {'input_features': {'observation.state': {'shape': [8]}}}))
    with pytest.raises(ValueError, match='disagree'):
        make_state_adapter(lambda: None, tmp_path)
