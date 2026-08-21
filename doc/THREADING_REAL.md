# Threading Real-Robot Training

This pipeline trains the existing `threading_task.policy.ThreadingARPolicy` on
real Franka recordings converted to LeRobot format by `Record_layer`.

The launcher `Record_layer/run_dual_robot_recording.sh` records raw trial
folders first. Convert those trials with `Record_layer/build_latest_lerobot_dataset.py`
before training here.

Expected LeRobot features:

| Model input | LeRobot feature |
| --- | --- |
| `top45` | `observation.images.exterior_image_1_left` |
| `wrist` | `observation.images.wrist_image_left` |
| `sideview` | `observation.images.exterior_image_2_right` |
| `agent_pos` | `observation.state` = `[q1..q7, gripper_width]` |
| `action` | `action` = next `[q1..q7, gripper_width]` |

Validate a converted dataset:

```bash
cd /home/tele/threading_real
python scripts/validate_threading_real_lerobot.py \
  /home/tele/Thesis_project/Record_layer/data_lerobot_latest \
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
