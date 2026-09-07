#!/usr/bin/env bash
set -euo pipefail

# Single-GPU SmolVLA fine-tuning: keep the pretrained VLM/vision tower frozen
# and train the action expert plus state projection.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
DATASET_ROOT=${DATASET_ROOT:-${PROJECT_ROOT}/data/block_grasp_smolvla_6hz}
REPO_ID=${REPO_ID:-threading_real/block_grasp_smolvla_6hz}
OUTPUT_DIR=${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/block_grasp_expert}
MODEL_ID=${MODEL_ID:-lerobot/smolvla_base}
GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
STEPS=${STEPS:-10000}
SAVE_FREQ=${SAVE_FREQ:-1000}
MIN_FREE_MIB=${MIN_FREE_MIB:-12000}
RENAME_MAP=${RENAME_MAP:-'{"observation.images.exterior_image_1_left":"observation.images.camera1","observation.images.exterior_image_2_right":"observation.images.camera2","observation.images.wrist_image_left":"observation.images.camera3"}'}

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one physical GPU index; got ${GPU_ID}." >&2
  exit 2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  free_mib=$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
    echo "Could not read free memory for GPU ${GPU_ID}." >&2
    exit 2
  fi
  if ((free_mib < MIN_FREE_MIB)); then
    echo "GPU ${GPU_ID} has only ${free_mib} MiB free; need at least ${MIN_FREE_MIB} MiB." >&2
    exit 2
  fi
fi

python - <<'PY'
import torch
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy  # noqa: F401

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the active Python environment.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected GPU/PyTorch build does not support bfloat16.")
PY

python "${SCRIPT_DIR}/preflight.py" \
  --dataset-root "${DATASET_ROOT}" \
  --repo-id "${REPO_ID}" \
  --chunk-size 10

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  echo "Set OUTPUT_DIR to a new path, or use the resume command in README.md." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$(command -v lerobot-train)" \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --rename_map="${RENAME_MAP}" \
  --policy.path="${MODEL_ID}" \
  --policy.chunk_size=10 \
  --policy.n_action_steps=2 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.load_vlm_weights=true \
  --policy.compile_model=false \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --accelerator.mixed_precision=bf16 \
  --checkpoint_format=safetensors \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=block_grasp_smolvla_expert \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --env_eval_freq=0 \
  --log_freq=10 \
  --save_freq="${SAVE_FREQ}" \
  --wandb.enable=false \
  --seed=1000
