#!/usr/bin/env bash
# Isolated 10-episode memorization diagnostic, initialized from pi05_base.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
export HF_HOME="$HERE/.cache/huggingface"
export HF_DATASETS_CACHE=/dev/shm/huiyuan_pi05_v8/hf_datasets
export HF_LEROBOT_HOME="$HERE/.cache/lerobot"
export OPENPI_DATA_HOME="$HERE/.cache/openpi"
export XDG_CACHE_HOME="$HERE/.cache"
export WANDB_MODE=online
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
DATASET=/dev/shm/huiyuan_pi05_v8/data/threading_tcp6_native640_30hz
test -f "$DATASET/meta/native_verification.json"
ARGS=(
  --mode vision_lora_action_full --image-profile native640 --vision-lora-rank 16
  --camera-views cam1 --exp-name threading_tcp6_cam1_native640_overfit10_v9
  --dataset-root "$DATASET" --repo-id threading_real/threading_tcp6_native640_30hz
  --overfit-episodes 10 --seed 42 --batch-size 6 --fsdp-devices 2
  --steps 10000 --eval-interval 500 --save-interval 500 --max-checkpoints 5 --wandb
)
export PYTHONPATH="$HERE/vendor/lerobot${PYTHONPATH:+:$PYTHONPATH}"
JAX_PLATFORMS=cpu .venv/bin/python -u run.py norm "${ARGS[@]}"
exec bash start_remote.sh "${ARGS[@]}"
