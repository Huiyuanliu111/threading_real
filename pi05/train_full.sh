#!/usr/bin/env bash
set -euo pipefail

# Two-card DDP configuration for the combined two-view threading dataset.
# Override variables on the command line, e.g. STEPS=10 ./train_full.sh.
PI05_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -d "${PI05_DIR}/../../data/threading_combined_pi05_15hz_sg5_nozero" ]]; then
  PROJECT_ROOT=$(cd "${PI05_DIR}/../.." && pwd)
else
  PROJECT_ROOT=$(cd "${PI05_DIR}/.." && pwd)
fi
DATASET_ROOT=${DATASET_ROOT:-${PROJECT_ROOT}/data/threading_combined_pi05_15hz_sg5_nozero}
REPO_ID=${REPO_ID:-threading_real/threading_combined_pi05_15hz_sg5_nozero}
OUTPUT_DIR=${OUTPUT_DIR:-${PI05_DIR}/outputs/threading_combined_pi05}
MODEL_ID=${MODEL_ID:-lerobot/pi05_base}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
STEPS=${STEPS:-15000}
# The training server has limited disk space. Keep three checkpoints while
# giving the 11,390-frame training split about 2.63 effective passes.
SAVE_FREQ=${SAVE_FREQ:-5000}
EVAL_FREQ=${EVAL_FREQ:-500}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-512}
EVAL_SPLIT=${EVAL_SPLIT:-0.2}
TRAIN_EXPERT_ONLY=${TRAIN_EXPERT_ONLY:-true}
WANDB_ENABLE=${WANDB_ENABLE:-true}
WANDB_PROJECT=${WANDB_PROJECT:-threading_pi05}
GPU_IDS=${GPU_IDS:-4,5}
NUM_PROCESSES=${NUM_PROCESSES:-2}

# Keep the only copy of downloaded base weights and W&B files inside this
# project. uv is invoked with --no-cache by bootstrap_remote.sh.
export HF_HOME=${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}
export WANDB_DIR=${WANDB_DIR:-${PROJECT_ROOT}/.cache/wandb}
mkdir -p "${HF_HOME}" "${WANDB_DIR}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set." >&2
  exit 2
fi
if ! python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="google/paligemma-3b-pt-224",
    filename="config.json",
)
print("Hugging Face access check passed: google/paligemma-3b-pt-224")
PY
then
  echo "HF_TOKEN is invalid or its account lacks access to google/paligemma-3b-pt-224." >&2
  exit 2
fi

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
      echo "GPU ${gpu_id} has only ${free_mib:-unknown} MiB free; refusing to start training." >&2
      exit 2
    fi
  done
fi

python - <<'PY'
try:
    from lerobot.configs.parallelism import ParallelismConfig  # noqa: F401
except ImportError as exc:
    raise SystemExit(
        "Installed LeRobot lacks the required distributed-training support. Recreate the environment "
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

exec torchrun --standalone --nproc-per-node="${NUM_PROCESSES}" "$(command -v lerobot-train)" \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --dataset.eval_split="${EVAL_SPLIT}" \
  --policy.type=pi05 \
  --policy.pretrained_path="${MODEL_ID}" \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only="${TRAIN_EXPERT_ONLY}" \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --parallelism.dp_replicate="${NUM_PROCESSES}" \
  --accelerator.mixed_precision=bf16 \
  --checkpoint_format=safetensors \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=threading_combined_pi05 \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --log_freq=10 \
  --eval_steps="${EVAL_FREQ}" \
  --max_eval_samples="${MAX_EVAL_SAMPLES}" \
  --save_freq="${SAVE_FREQ}" \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${WANDB_PROJECT}" \
  --seed=1000
