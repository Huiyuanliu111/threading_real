#!/usr/bin/env bash
set -euo pipefail

# Wait for enough free VRAM, run a two-step smoke test, then start the full run.
# Intended to be launched by the user's systemd timer.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
ENV_DIR=${ENV_DIR:-${PROJECT_ROOT}/.venv-smolvla}
GPU_ID=${GPU_ID:-0}
MIN_FREE_MIB=${MIN_FREE_MIB:-12000}
POLL_SECONDS=${POLL_SECONDS:-60}
SMOKE_OUTPUT=${SMOKE_OUTPUT:-${SCRIPT_DIR}/outputs/smoke}
FULL_OUTPUT=${FULL_OUTPUT:-${SCRIPT_DIR}/outputs/block_grasp_expert}
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/scheduled_training.log}
STATUS_FILE=${STATUS_FILE:-${LOG_DIR}/scheduled_training.status}

mkdir -p "${LOG_DIR}"
exec >>"${LOG_FILE}" 2>&1

status() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee "${STATUS_FILE}"
}

trap 'status "FAILED at line ${LINENO}; inspect ${LOG_FILE}"' ERR

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  status "FAILED: environment not found at ${ENV_DIR}"
  exit 2
fi
if [[ -e "${SMOKE_OUTPUT}" || -e "${FULL_OUTPUT}" ]]; then
  status "FAILED: smoke or full output directory already exists"
  exit 2
fi

source "${ENV_DIR}/bin/activate"
export PYTHONUNBUFFERED=1
status "Timer fired; waiting for GPU ${GPU_ID} to have ${MIN_FREE_MIB} MiB free"

while true; do
  free_mib=$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
    break
  fi
  status "GPU ${GPU_ID} still busy (${free_mib:-unknown} MiB free); checking again in ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

status "Starting two-step smoke test (${free_mib} MiB free)"
GPU_ID="${GPU_ID}" BATCH_SIZE=1 STEPS=2 SAVE_FREQ=2 \
  OUTPUT_DIR="${SMOKE_OUTPUT}" \
  bash "${SCRIPT_DIR}/train_expert.sh"

status "Smoke test passed; starting 10000-step full training"
GPU_ID="${GPU_ID}" BATCH_SIZE=1 STEPS=10000 SAVE_FREQ=1000 \
  OUTPUT_DIR="${FULL_OUTPUT}" \
  bash "${SCRIPT_DIR}/train_expert.sh"

status "COMPLETE: full training finished"
