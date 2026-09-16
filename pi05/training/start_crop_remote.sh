#!/usr/bin/env bash
# Launch the isolated crop experiment on 10.157.174.249 after data/env preparation.
set -euo pipefail
RUN_ROOT=${RUN_ROOT:-/home/huiyuan/threading_real/pi05/crop_20260915}
ENV_DIR=${ENV_DIR:-${RUN_ROOT}/.venv-pi05}
export DATASET_ROOT=${DATASET_ROOT:-${RUN_ROOT}/data/threading_crop_15hz}
export REPO_ID=${REPO_ID:-threading_real/threading_crop_15hz}
export OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/pi05/outputs/threading_crop_v1}
export JOB_NAME=${JOB_NAME:-threading_crop_v1}
export HF_HOME=${HF_HOME:-/home/huiyuan/pi05/.cache/huggingface}
export WANDB_DIR=${WANDB_DIR:-${RUN_ROOT}/logs}
export GPU_IDS=${GPU_IDS:-0,1,3,4,5,6}
export NUM_PROCESSES=${NUM_PROCESSES:-6}
export BATCH_SIZE=${BATCH_SIZE:-2}
export GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-1}
export STEPS=${STEPS:-5000}
# Keep one final training checkpoint to limit disk use; validation still runs every 500 updates.
export SAVE_FREQ=${SAVE_FREQ:-${STEPS}}
export EVAL_FREQ=${EVAL_FREQ:-500}
export STATE_REPRESENTATION=tcp_pose_6d
export FINETUNE_MODE=visual_full_expert
export CHUNK_SIZE=10
export N_ACTION_STEPS=10
MIN_DISK_FREE_GIB=${MIN_DISK_FREE_GIB:-40}

if [[ "${PRINT_CONFIG_ONLY:-false}" == true ]]; then
  bash "${RUN_ROOT}/pi05/training/train.sh"
  exit 0
fi
[[ -x "${ENV_DIR}/bin/python" ]] || { echo "Missing isolated environment: ${ENV_DIR}" >&2; exit 2; }
[[ -f "${DATASET_ROOT}/meta/visual_preprocessing.json" ]] || { echo "Missing cropped dataset metadata: ${DATASET_ROOT}" >&2; exit 2; }
available_bytes=$(df -B1 --output=avail "${RUN_ROOT}" | tail -n 1 | tr -d ' ')
if ((available_bytes < MIN_DISK_FREE_GIB * 1024 * 1024 * 1024)); then
  echo "Need at least ${MIN_DISK_FREE_GIB} GiB free for the final checkpoint; available bytes=${available_bytes}." >&2
  exit 2
fi
source "${ENV_DIR}/bin/activate"
cd "${RUN_ROOT}"
exec bash "${RUN_ROOT}/pi05/training/train.sh"
