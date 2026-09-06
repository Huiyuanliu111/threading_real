#!/usr/bin/env bash
set -euo pipefail

# Two-card FSDP2 configuration for the free A40 GPUs on 10.157.174.249.
# Override variables on the command line, e.g. STEPS=10 ./train_full.sh.
PI05_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -d "${PI05_DIR}/../data/block_grasp_minimal_pi05_6hz" ]]; then
  PROJECT_ROOT=$(cd "${PI05_DIR}/.." && pwd)
else
  PROJECT_ROOT=$(cd "${PI05_DIR}/../.." && pwd)
fi
DATASET_ROOT=${DATASET_ROOT:-${PROJECT_ROOT}/data/block_grasp_minimal_pi05_6hz}
REPO_ID=${REPO_ID:-threading_real/block_grasp_minimal_pi05_6hz}
OUTPUT_DIR=${OUTPUT_DIR:-${PI05_DIR}/outputs/block_grasp_minimal_full}
MODEL_ID=${MODEL_ID:-lerobot/pi05_base}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
STEPS=${STEPS:-3000}
SAVE_FREQ=${SAVE_FREQ:-500}
GPU_IDS=${GPU_IDS:-4,5}
NUM_PROCESSES=${NUM_PROCESSES:-2}
FSDP_MIN_NUM_PARAMS=${FSDP_MIN_NUM_PARAMS:-10000000}

if [[ "${NUM_PROCESSES}" != "2" ]]; then
  echo "This profile is sized for exactly two A40 GPUs; got NUM_PROCESSES=${NUM_PROCESSES}." >&2
  exit 2
fi
IFS=',' read -r GPU_A GPU_B GPU_EXTRA <<<"${GPU_IDS}"
if [[ -z "${GPU_A}" || -z "${GPU_B}" || -n "${GPU_EXTRA}" || "${GPU_A}" == "${GPU_B}" ]]; then
  echo "GPU_IDS must contain two physical GPU indices, e.g. GPU_IDS=4,5." >&2
  exit 2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  for gpu_id in "${GPU_A}" "${GPU_B}"; do
    free_mib=$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if [[ ! "${free_mib}" =~ ^[0-9]+$ ]]; then
      echo "Could not read free memory for GPU ${gpu_id}." >&2
      exit 2
    fi
    if ((free_mib < 40000)); then
      echo "GPU ${gpu_id} has only ${free_mib:-unknown} MiB free; refusing to start FSDP." >&2
      exit 2
    fi
  done
fi

python - <<'PY'
try:
    from lerobot.configs.accelerator import FSDPConfig  # noqa: F401
    from lerobot.configs.parallelism import ParallelismConfig  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "Installed LeRobot lacks native FSDP2 support. Recreate the environment "
        "with bootstrap_remote.sh before training."
    ) from exc
PY

python "${PI05_DIR}/preflight.py" \
  --dataset-root "${DATASET_ROOT}" \
  --repo-id "${REPO_ID}" \
  --chunk-size 10

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  echo "Set OUTPUT_DIR to a new directory, or use the documented resume command." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

torchrun --standalone --nproc-per-node="${NUM_PROCESSES}" "$(command -v lerobot-train)" \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path="${MODEL_ID}" \
  --policy.chunk_size=10 \
  --policy.n_action_steps=2 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --parallelism.dp_shard="${NUM_PROCESSES}" \
  --accelerator.mixed_precision=bf16 \
  --accelerator.fsdp.min_num_params="${FSDP_MIN_NUM_PARAMS}" \
  --accelerator.fsdp.reshard_after_forward=true \
  --accelerator.fsdp.cpu_offload=false \
  --checkpoint_format=safetensors \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=block_grasp_minimal_pi05_full \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --log_freq=10 \
  --save_freq="${SAVE_FREQ}" \
  --wandb.enable=false \
  --seed=1000
