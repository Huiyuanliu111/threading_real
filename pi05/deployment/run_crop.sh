#!/usr/bin/env bash
# Usage: bash run_crop.sh 10|20 [cartesian deployment arguments]
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PI05_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
CHUNK=${1:?Specify trained chunk size: 10 or 20}
shift
case "${CHUNK}" in
  10) RUN=threading_crop_v1 ;;
  20) RUN=threading_crop_chunk20_v1 ;;
  *) echo 'Supported trained chunk sizes: 10, 20' >&2; exit 2 ;;
esac
CHECKPOINT=${PI05_ROOT}/outputs/${RUN}/checkpoints/005000/pretrained_model
PYTHON=${PYTHON:-${PI05_ROOT}/.venv-deploy/bin/python}
[[ -s ${CHECKPOINT}/model.safetensors ]] || { echo "Model not downloaded: ${CHECKPOINT}" >&2; exit 2; }
[[ -x ${PYTHON} ]] || { echo "Compatible deployment environment not installed: ${PYTHON}" >&2; exit 2; }
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
exec "${PYTHON}" "${SCRIPT_DIR}/cropped.py" "${CHECKPOINT}" \
  --policy-kind pi05 --weights model \
  --task 'insert the grasped block through the needle' \
  --policy-hz 15 --execute-steps 1 --prediction-mode full_then_truncate \
  "$@"
