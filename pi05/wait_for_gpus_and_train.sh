#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -d "${SCRIPT_DIR}/../../data/threading_combined_pi05_15hz_sg5_nozero" ]]; then
  PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
else
  PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
fi

GPU_IDS=${GPU_IDS:-4,5}
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

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Read and export it before starting this watcher:" >&2
  echo "  read -r -s -p 'HF token: ' HF_TOKEN; echo; export HF_TOKEN" >&2
  exit 2
fi

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
  echo "PaliGemma access check failed. HF_TOKEN is invalid or lacks gated-repo access." >&2
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
  GPU_IDS="${GPU_IDS}" \
  BATCH_SIZE="${BATCH_SIZE:-1}" \
  STEPS="${STEPS:-15000}" \
  SAVE_FREQ="${SAVE_FREQ:-5000}" \
  EVAL_FREQ="${EVAL_FREQ:-500}" \
  MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-512}" \
  TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}" \
  bash "${SCRIPT_DIR}/train_full.sh"
