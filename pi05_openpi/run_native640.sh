#!/usr/bin/env bash
# Cam1-only native 640x480 RGB, black padding only; independent v8 experiment.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${OPENPI_PY:-"$HERE/.venv/bin/python"}
if [[ $# -lt 1 ]]; then
  echo "Usage: OPENPI_PY=<python> bash run_native640.sh {check|norm|train} [options]" >&2
  exit 2
fi
COMMAND=$1
shift
export PYTHONPATH="$HERE/vendor/lerobot${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" "$HERE/run.py" "$COMMAND" \
  --mode vision_lora_action_full --image-profile native640 --vision-lora-rank 16 \
  --camera-views cam1 --exp-name threading_tcp6_cam1_native640_vision_lora_action_full_v8 \
  --dataset-root "$HERE/data/threading_tcp6_native640_30hz" \
  --repo-id threading_real/threading_tcp6_native640_30hz \
  --batch-size 6 --fsdp-devices 2 --eval-interval 1000 --save-interval 2000 --wandb "$@"
