"""Apply the checkpoint's training-time robot state representation to live RGB frames."""
import json
from pathlib import Path

import torch


def make_state_adapter(original, checkpoint: Path):
    from threading_real.pi05.training.prepare_dataset import STATE_FEATURES, convert_state
    from convert_lerobot_v3_to_cartesian import UrdfForwardKinematics

    metadata = json.loads((checkpoint / 'state_representation.json').read_text())
    model_config = json.loads((checkpoint / 'config.json').read_text())
    representation = metadata['state_representation']
    if representation not in STATE_FEATURES:
        raise ValueError(f'Unsupported checkpoint state representation: {representation}')
    dimension = STATE_FEATURES[representation]['shape'][0]
    configured = model_config['input_features']['observation.state']['shape']
    if metadata['state_dim'] != dimension or configured != [dimension]:
        raise ValueError('Checkpoint state metadata and model input dimension disagree')
    root = Path(__file__).resolve().parents[3]
    fk = None if representation == 'joint' else UrdfForwardKinematics(
        root / 'remote_controller/src/remote_controller/assets/panda/panda_arm.urdf'
    )

    def observe(*args, **kwargs):
        frame = original(*args, **kwargs)
        # The shared RGB runner returns [q1..q7, gripper_width]. Reuse exactly
        # the dataset conversion, including TCP frame and rotation-column order.
        state = convert_state(frame.agent_pos, representation, fk)
        frame.agent_pos = torch.from_numpy(state)
        return frame

    print(f'[state] checkpoint representation={representation}, dimension={dimension}', flush=True)
    return observe
