#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=${RUN_ROOT:-/home/huiyuan/threading_real/pi05/crop_20260915}
export HF_HOME=${HF_HOME:-/home/huiyuan/pi05/.cache/huggingface}
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export DATASET_ROOT=${RUN_ROOT}/data/threading_crop_15hz
export REPO_ID=threading_real/threading_crop_15hz
export OUTPUT_DIR=${RUN_ROOT}/pi05/outputs/threading_crop_chunk20_v1
export JOB_NAME=threading_crop_chunk20_v1
export MODEL_ID=/home/huiyuan/pi05/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba
export WANDB_DIR=${RUN_ROOT}/logs
export CHUNK_SIZE=20 N_ACTION_STEPS=20
export STEPS=5000 SAVE_FREQ=500 EVAL_FREQ=500
export GPU_IDS=0,1,3,4,5,6 NUM_PROCESSES=6 BATCH_SIZE=2
export STATE_REPRESENTATION=tcp_pose_6d FINETUNE_MODE=visual_full_expert
export PI05_KEEP_CHECKPOINTS=2
export TRAIN_ENTRYPOINT=${RUN_ROOT}/pi05/training/train_checkpointed.py
if [[ ${PRINT_CONFIG_ONLY:-false} != true ]]; then
  [[ ! -e ${OUTPUT_DIR} ]] || { echo "Output already exists: ${OUTPUT_DIR}" >&2; exit 2; }
  available_bytes=$(df -B1 --output=avail "${RUN_ROOT}" | tail -n 1 | tr -d ' ')
  (( available_bytes >= 55 * 1024 * 1024 * 1024 )) || { echo 'Need 55 GiB free for rolling saves' >&2; exit 2; }
fi
source "${RUN_ROOT}/.venv-pi05/bin/activate"
cd "${RUN_ROOT}"
exec bash "${RUN_ROOT}/pi05/training/train.sh"
