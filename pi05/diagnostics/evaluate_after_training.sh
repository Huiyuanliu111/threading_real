#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/huiyuan/pi05}
RUN_NAME=${RUN_NAME:-threading_combined_pi05_conditioned_v8_visual_forced_probe}
TRAIN_PID=${1:?usage: evaluate_v8_after_training.sh TRAIN_PID}
OUTPUT_ROOT=${PROJECT_ROOT}/pi05/outputs/${RUN_NAME}
RESULT_ROOT=${OUTPUT_ROOT}/condition_sensitivity
LOG=${PROJECT_ROOT}/logs/${RUN_NAME}_condition_sensitivity.log
PYTHON=${PROJECT_ROOT}/.venv-pi05/bin/python

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 30
done

mkdir -p "${RESULT_ROOT}"
for step in 000500 001000 001500; do
  checkpoint=${OUTPUT_ROOT}/checkpoints/${step}/pretrained_model
  if [[ ! -f "${checkpoint}/model.safetensors" ]]; then
    echo "missing checkpoint ${checkpoint}; skipping" >>"${LOG}"
    continue
  fi
  echo "evaluating ${step} at $(date --iso-8601=seconds)" >>"${LOG}"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${PROJECT_ROOT}/pi05/diagnostics/condition_sensitivity.py" \
    --checkpoint "${checkpoint}" \
    --dataset-root "${PROJECT_ROOT}/data/threading_combined_pi05_15hz_sg5_nozero" \
    --pairs 6 \
    --seed 20260909 \
    --device cuda:0 \
    --output "${RESULT_ROOT}/${step}.json" >>"${LOG}" 2>&1
done
