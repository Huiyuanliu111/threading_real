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

### Delta-action retraining

`threading_real_arp_delta_aug.yaml` predicts the next-state delta
`[q(t+1), width(t+1)] - [q(t), width(t)]`. It also enables stronger training
image augmentation (color, hue, pixel noise, and small translations). This is
a different action representation, so do **not** resume an absolute-action
checkpoint. Start a fresh run:

```bash
cd /home/huiyuan/teleoperation/threading_real
python pushbox/train.py \
  --config-name=threading_real_arp_delta_aug \
  task.dataset.dataset_path=/home/huiyuan/teleoperation/data/threading_lerobot_v3 \
  training.device=cuda:0
```

The delta checkpoint is automatically converted back to absolute joint targets
by `scripts/deploy_threading_real.py` before TrackJ receives a plan.

Real-robot rollout is disabled in `threading_real_arp.yaml`; evaluation should
use held-out validation loss or the separate follower deployment runner below.

## Single-follower deployment

The two-arm leader/follower executable is only used to collect demonstrations.
At deployment time, run `remote_controller_server` on the follower controller
PC and run the Python policy client on the inference PC. The server IP used by
the examples below depends on whether these are the same machine.

The runner imports the repository-local controller client automatically. To use
it independently elsewhere, install it in the policy environment:

```bash
python -m pip install -e ../remote_controller
python -m pip install pyrealsense2
```

Start the server on the follower controller PC:

```bash
cd /home/huiyuan/teleoperation/remote_controller
./run_server.sh
```

First run observation and inference only. This mode never starts TrackJ or
sends gripper commands:

```bash
cd /home/huiyuan/teleoperation/threading_real
python scripts/deploy_threading_real.py \
  outputs/2026-08-31/11-10-27/checkpoints/epoch=0005-val_loss=-18.758.ckpt \
  --server-url http://localhost:8008/RPC2 \
  --server-ip 127.0.0.1 \
  --udp-ip 127.0.0.1 \
  --max-cycles 20
```

If inference runs on a different PC, `--server-url` and `--server-ip` must use
the follower controller PC address, while `--udp-ip` must be an address of the
inference PC reachable by the server.

After inspecting the dry-run predictions and clearing the workspace, real
motion requires both acknowledgement flags:

```bash
python scripts/deploy_threading_real.py \
  outputs/2026-08-31/11-10-27/checkpoints/epoch=0005-val_loss=-18.758.ckpt \
  --execute --confirm-real-robot \
  --move-to-training-start
```

The runner uses the same RealSense serial mapping as data collection: camera
`233722072293` is `sideview`/`cam1`, and camera `233622071984` is
`wrist`/`cam2`. Override `--sideview-serial` and `--wrist-serial` if the
hardware mapping changes. Keep the Franka user stop reachable. Any camera,
fresh-state, inference, or controller error exits the loop and requests TrackJ
to stop. Hardware inference defaults to deterministic GMM MAP output. Raw
targets with a first-point or adjacent-point jump above `0.15 rad` abort the
rollout before that chunk is sent; smaller targets are additionally rate
limited by `--max-first-delta` and `--max-step-delta`.

`--move-to-training-start` first moves the follower at low speed to the default
evaluation posture captured from the follower on 2026-09-04:
`[0.307272, 0.323924, -0.112529, -2.501686, -0.012559, 2.764401, 0.833281]`
rad, then opens the gripper. Override it with `--training-start-q` when a
different safe posture is required. It is intentionally unavailable in dry-run
mode.

### Cartesian-delta deployment

`scripts/deploy_threading_real_cartesian.py` is for checkpoints trained with
`action_mode: cartesian_delta`. It reads the live joint state, computes the
`panda_hand_tcp` pose with the packaged URDF, integrates base-frame TCP deltas,
and streams the resulting poses through TrackC (UDP port 9200 by default).
The runner requires `pinocchio` in the policy environment. Run dry-run first;
real execution additionally requires explicit `--workspace-min X Y Z` and
`--workspace-max X Y Z` bounds.
