#!/usr/bin/env bash
# Cameras and TrackC on this workstation; OpenPI weights on inference server.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export CUDA_VISIBLE_DEVICES=""
exec "${SCRIPT_DIR}/.venv-deploy/bin/python" "${SCRIPT_DIR}/deploy_remote.py" \
  /home/huiyuan/threading_real/pi05_openpi/checkpoints/pi05_threading_vision_lora_action_full/threading_tcp6_30hz_h10_vision_lora_action_full_v11/2000 \
  --policy-host 127.0.0.1 --policy-port 18000 --inference-timeout 5 \
  --device cpu --weights model --policy-hz 30 --stream-hz 480 --gripper-force 70 \
  --execute-steps 10 --no-sync-require-target --prediction-mode full_then_truncate --synchronous --sync-timeout 5.0 \
  "$@"
