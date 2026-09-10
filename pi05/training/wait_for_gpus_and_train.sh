#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PI05_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
if [[ -d "${PI05_DIR}/../../data/threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d" ]]; then
  PROJECT_ROOT=$(cd "${PI05_DIR}/../.." && pwd)
else
  PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
fi

GPU_IDS=${GPU_IDS:-0,1,3,4,5,6}
MIN_FREE_MIB=${MIN_FREE_MIB:-40000}
MAX_UTIL_PERCENT=${MAX_UTIL_PERCENT:-5}
POLL_SECONDS=${POLL_SECONDS:-30}
STABLE_CHECKS=${STABLE_CHECKS:-3}
MIN_DISK_FREE_GIB=${MIN_DISK_FREE_GIB:-45}
ENV_DIR=${ENV_DIR:-${PROJECT_ROOT}/.venv-pi05}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/logs}

mkdir -p "${LOG_DIR}"

# This prevents this account from launching multiple watchers for the same run.
exec 9>"${PROJECT_ROOT}/.pi05_gpu_wait.lock"
if ! flock -n 9; then
  echo "Another pi05 GPU watcher or training launch already owns the lock." >&2
  exit 2
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "Missing environment: ${ENV_DIR}" >&2
  exit 2
fi

source "${ENV_DIR}/bin/activate"
export HF_HOME=${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}
export WANDB_DIR=${WANDB_DIR:-${PROJECT_ROOT}/.cache/wandb}
mkdir -p "${HF_HOME}" "${WANDB_DIR}"

# Check the exact gated dependency before waiting hours for GPUs. `hf auth
# whoami` alone is insufficient because a valid token may still lack approval
# for PaliGemma.
if ! python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="google/paligemma-3b-pt-224",
    filename="config.json",
)
print("Hugging Face access check passed: google/paligemma-3b-pt-224")
PY
then
  echo "PaliGemma access failed; log in with huggingface-cli or set an authorized HF_TOKEN." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable." >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
if ((${#GPU_ARRAY[@]} == 0)); then
  echo "GPU_IDS is empty." >&2
  exit 2
fi

stable=0
echo "Waiting for GPUs ${GPU_IDS}: free>=${MIN_FREE_MIB} MiB, util<=${MAX_UTIL_PERCENT}%, ${STABLE_CHECKS} consecutive checks."

while true; do
  mapfile -t rows < <(
    nvidia-smi --id="${GPU_IDS}" \
      --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )

  ready=true
  status=()
  if ((${#rows[@]} != ${#GPU_ARRAY[@]})); then
    ready=false
  fi
  for row in "${rows[@]}"; do
    IFS=',' read -r idx free_mib util <<<"${row}"
    idx=${idx//[[:space:]]/}
    free_mib=${free_mib//[[:space:]]/}
    util=${util//[[:space:]]/}
    status+=("gpu${idx}:free=${free_mib}MiB,util=${util}%")
    if [[ ! "${free_mib}" =~ ^[0-9]+$ || ! "${util}" =~ ^[0-9]+$ ]] || \
       ((free_mib < MIN_FREE_MIB || util > MAX_UTIL_PERCENT)); then
      ready=false
    fi
  done

  if [[ "${ready}" == true ]]; then
    ((stable += 1))
  else
    stable=0
  fi
  printf '%s ready=%s stable=%d/%d %s\n' \
    "$(date --iso-8601=seconds)" "${ready}" "${stable}" "${STABLE_CHECKS}" "${status[*]}"

  if ((stable >= STABLE_CHECKS)); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

free_kib=$(df -Pk "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')
required_kib=$((MIN_DISK_FREE_GIB * 1024 * 1024))
if [[ ! "${free_kib}" =~ ^[0-9]+$ ]] || ((free_kib < required_kib)); then
  echo "GPU is free, but disk has less than ${MIN_DISK_FREE_GIB} GiB available; refusing to train." >&2
  exit 2
fi

echo "GPUs remained available; starting pi05 training at $(date --iso-8601=seconds)."

exec env \
  DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d}" \
  REPO_ID="${REPO_ID:-threading_real/threading_combined_pi05_15hz_sg5_continuous_tcp_pose_6d}" \
  OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/threading_combined_pi05_tcp_pose_6d_continuous_v1}" \
  MODEL_ID="${MODEL_ID:-lerobot/pi05_base}" \
  STATE_REPRESENTATION="${STATE_REPRESENTATION:-tcp_pose_6d}" \
  CHUNK_SIZE="${CHUNK_SIZE:-10}" \
  N_ACTION_STEPS="${N_ACTION_STEPS:-${CHUNK_SIZE:-10}}" \
  EXPECTED_EPISODES="${EXPECTED_EPISODES:-80}" \
  EXPECTED_FPS="${EXPECTED_FPS:-15}" \
  MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-${MIN_FREE_MIB}}" \
  GPU_IDS="${GPU_IDS}" \
  NUM_PROCESSES="${NUM_PROCESSES:-6}" \
  BATCH_SIZE="${BATCH_SIZE:-2}" \
  GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}" \
  STEPS="${STEPS:-5000}" \
  SAVE_FREQ="${SAVE_FREQ:-1000}" \
  EVAL_FREQ="${EVAL_FREQ:-500}" \
  MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-512}" \
  FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}" \
  TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}" \
  PROPRIOCEPTION_DROPOUT="${PROPRIOCEPTION_DROPOUT:-0.15}" \
  IGNORE_GRIPPER_ACTION="${IGNORE_GRIPPER_ACTION:-true}" \
  FINETUNE_MODE="${FINETUNE_MODE:-visual_full_expert}" \
  VISION_LR="${VISION_LR:-2.5e-6}" \
  PROJECTOR_LR="${PROJECTOR_LR:-1e-5}" \
  EXPERT_ATTENTION_LR="${EXPERT_ATTENTION_LR:-5e-6}" \
  EXPERT_MLP_LR="${EXPERT_MLP_LR:-2.5e-6}" \
  ACTION_LR="${ACTION_LR:-1e-5}" \
  HIGH_NOISE_FRACTION="${HIGH_NOISE_FRACTION:-0.5}" \
  HIGH_NOISE_MIN_TIME="${HIGH_NOISE_MIN_TIME:-0.8}" \
  FIXED_EVAL_SEED="${FIXED_EVAL_SEED:-20260909}" \
  LOG_FREQ="${LOG_FREQ:-20}" \
  SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-250}" \
  SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-5000}" \
  bash "${SCRIPT_DIR}/train.sh"
