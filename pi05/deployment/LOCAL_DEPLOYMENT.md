# Local cropped pi0.5 deployment

Both runs use their final step-5000 checkpoint. Downloaded deployment bundles include weights,
normalization processors, tokenizer, TCP state metadata, and the exact ROI configuration.
Optimizer states are kept on the training server and are not needed for deployment.

| Model | Local directory under `threading_real/pi05/outputs` | Training horizon |
|---|---|---|
| chunk 10 | `threading_crop_v1/checkpoints/005000/pretrained_model` | 10 steps |
| chunk 20 | `threading_crop_chunk20_v1/checkpoints/005000/pretrained_model` | 20 steps |

Run from `/home/huiyuan/teleoperation`:

```bash
# Camera/state dry run (no action execution). Requires the robot state service.
bash threading_real/pi05/deployment/run_crop.sh 10 --max-cycles 20
bash threading_real/pi05/deployment/run_crop.sh 20 --max-cycles 20

# Real execution, explicitly enabled by the operator:
bash threading_real/pi05/deployment/run_crop.sh 10 --execute --confirm-real-robot
bash threading_real/pi05/deployment/run_crop.sh 20 --execute --confirm-real-robot
```

The launcher selects `pi05/.venv-deploy/bin/python`, 15 Hz action timing, the training task prompt,
and the checkpoint's raw-camera ROIs. The crop entry point also reads `state_representation.json` and converts live joint observations to the same TCP state used for training (xyz, rotation column 0, rotation column 1, gripper width). Default execution is **one action per replan** for both models,
while predicting the full trained horizon. To execute more, append `--execute-steps N` (N <= horizon).
Thus model chunk size 20 does not automatically execute 20 actions. Do not run both deployment
processes at once or alongside another process controlling the robot/using the same cameras.

The environment was copied from the training server's Python 3.12 environment and its executable
paths were relocated. Core versions: LeRobot 0.6.2 (training revision
`3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`), torch 2.11.0+cu128, transformers 5.5.4.
Added pin 4.1.0, pyrealsense2 2.58.4.10922 and pytest. Existing conda environments were not changed.

Checks and source-file SHA256 manifest are in `pi05/outputs/deployment_preparation_20260918/`.
No robot session was started during preparation. Both models passed full local GPU inference using synthetic live RGB/joint observations after fixing the TCP state adapter. No real camera/robot session was started. Results: `live_state_smoke.json`; 10 regression tests passed.

Offline inference, after the GPU is free:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 threading_real/pi05/.venv-deploy/bin/python \
  threading_real/pi05/diagnostics/infer_one.py \
  --checkpoint threading_real/pi05/outputs/threading_crop_v1/checkpoints/005000/pretrained_model \
  --dataset-root threading_real/pi05/data/threading_crop_15hz \
  --repo-id threading_real/threading_crop_15hz --frame 60 --warmup-runs 1 --benchmark-runs 3
```

For chunk 20, replace `threading_crop_v1` with `threading_crop_chunk20_v1`.
