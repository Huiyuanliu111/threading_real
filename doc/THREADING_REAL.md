# Threading Real-Robot Training

This pipeline trains the existing `threading_task.policy.ThreadingARPolicy` on
real Franka recordings converted to official LeRobot Dataset v3.0 format.

The `vla_finetune` recorder creates raw trial folders first. Convert a session
from the `teleoperation` repository root:

```bash
python convert_vla_to_lerobot_v3.py \
  vla_finetune/data/<session> \
  data/threading_vla_lerobot_v3
```

The converter uses the official `lerobot` writer and creates v3 `meta/`,
`data/`, and `videos/` shards. The current raw recorder does not persist camera
or robot timestamps, so conversion uses normalized episode progress for 30 FPS
alignment and records this limitation in `meta/vla_conversion_report.json`.
The default raw-camera mapping is `cam1.mp4` = side view and `cam2.mp4` =
wrist; pass explicit `--camera` arguments to the converter if the recording
machine uses the opposite numbering.

Expected LeRobot features:

| Model input | LeRobot feature |
| --- | --- |
| `sideview` | `observation.images.exterior_image_2_right` |
| `wrist` | `observation.images.wrist_image_left` |
| `agent_pos` | `observation.state` = `[q1..q7, gripper_width]` |
| `action` | `action` = next `[q1..q7, gripper_width]` |

Validate a converted dataset:

```bash
cd /home/tele/threading_real
python scripts/validate_threading_real_lerobot.py \
  /path/to/threading_vla_lerobot_v3 \
  --max-episodes 3
```

Train:

```bash
cd /home/tele/threading_real
python pushbox/train.py \
  --config-name=threading_real_arp \
  task.dataset.dataset_path=/path/to/record_layer_lerobot_dataset \
  training.device=cuda:0
```

Real-robot rollout is disabled in `threading_real_arp.yaml`; evaluation should
use held-out validation loss or a separate hardware deployment script.
