#!/usr/bin/env bash
# Local 4060 Ti + local cameras. Defaults to one executed action per full H50 prediction.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${SCRIPT_DIR}/vendor/lerobot${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "${SCRIPT_DIR}/.venv-deploy/bin/python" "${SCRIPT_DIR}/deploy.py" \
  "${SCRIPT_DIR}/checkpoints/pi05_threading_lora/threading_lora_tcp6_30hz_h50_best_v4/2000" \
  --run-config "${SCRIPT_DIR}/deployment_config.json" \
  --device cpu --weights model --policy-hz 30 --stream-hz 480 --gripper-force 70 \
  --execute-steps 1 --prediction-mode full_then_truncate --synchronous --sync-timeout 5.0 \
  "$@"
