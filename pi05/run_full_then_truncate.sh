#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON=${PYTHON:-${PROJECT_ROOT}/.venv-pi05/bin/python}
CHECKPOINT=${CHECKPOINT:-${SCRIPT_DIR}/outputs/threading_combined_pi05/checkpoints/last/pretrained_model}
SELECTOR=${SELECTOR:-${SCRIPT_DIR}/outputs/selector_soft_4_10}
RESULTS_DIR=${RESULTS_DIR:-${SCRIPT_DIR}/outputs/adaptive_comparison}
TRIAL_ID=full_then_truncate_$(date +%Y%m%dT%H%M%S_%N)
TRIAL_DIR=${RESULTS_DIR}/${TRIAL_ID}
TRACE=${TRIAL_DIR}/trace.jsonl
mkdir -p "${TRIAL_DIR}"

"${PYTHON}" "${PROJECT_ROOT}/threading_real/scripts/deploy_threading_real_cartesian.py" \
  "${CHECKPOINT}" \
  --policy-kind pi05 \
  --chunk-selector "${SELECTOR}" \
  --prediction-mode full_then_truncate \
  --trace-output "${TRACE}" \
  "$@"

executing=false
for argument in "$@"; do
  [[ "${argument}" == "--execute" ]] && executing=true
done

if [[ "${executing}" == true ]]; then
  mapfile -t completed_episodes < <("${PYTHON}" -c \
    'import json,sys; print(*sorted({int(json.loads(line).get("episode", 1)) for line in open(sys.argv[1]) if line.strip()}), sep="\n")' \
    "${TRACE}")
  if [[ ${#completed_episodes[@]} -eq 0 ]]; then
    echo "No completed inference cycles were recorded in ${TRACE}." >&2
    exit 1
  fi
  for episode in "${completed_episodes[@]}"; do
    read -r -p "Episode ${episode} success? [y/n] " answer
    case "${answer}" in
      y|Y|yes|YES) success=yes ;;
      n|N|no|NO) success=no ;;
      *) echo "Outcome not recorded: answer must be y or n." >&2; exit 2 ;;
    esac
    "${PYTHON}" "${SCRIPT_DIR}/record_trial_result.py" \
      --trace "${TRACE}" \
      --episode "${episode}" \
      --output "${TRIAL_DIR}/episode_${episode}/trial.json" \
      --success "${success}"
  done
else
  echo "Dry-run timing trace: ${TRACE}"
fi
