#!/usr/bin/env bash
# Server launch: original dataset remains on the workstation; RAM-backed copy/cache
# reserve persistent disk space for all five periodic checkpoints.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
export HF_HOME="$HERE/.cache/huggingface"
export HF_DATASETS_CACHE=/dev/shm/huiyuan_pi05_v8/hf_datasets
export HF_LEROBOT_HOME="$HERE/.cache/lerobot"
export OPENPI_DATA_HOME="$HERE/.cache/openpi"
export XDG_CACHE_HOME="$HERE/.cache"
export WANDB_MODE=online
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
DATASET=/dev/shm/huiyuan_pi05_v8/data/threading_tcp6_native640_30hz
test -f "$DATASET/meta/native_verification.json"
JAX_PLATFORMS=cpu bash run_native640.sh norm --dataset-root "$DATASET"
exec bash start_remote.sh \
  --mode vision_lora_action_full --image-profile native640 --vision-lora-rank 16 \
  --camera-views cam1 --exp-name threading_tcp6_cam1_native640_vision_lora_action_full_v8 \
  --dataset-root "$DATASET" --repo-id threading_real/threading_tcp6_native640_30hz \
  --batch-size 6 --fsdp-devices 2 --eval-interval 1000 --save-interval 2000 --wandb
