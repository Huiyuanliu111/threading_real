#!/usr/bin/env bash
set -euo pipefail

# Convert timestamp-synchronized raw RGB-D recordings into the RGB-only,
# 6 Hz Cartesian-action LeRobot v3 dataset consumed by SmolVLA.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
RAW_ROOT=${RAW_ROOT:-${PROJECT_ROOT}/data/block_grasp}
JOINT_30HZ=${JOINT_30HZ:-${PROJECT_ROOT}/data/block_grasp_lerobot_v3_joint_30hz}
CARTESIAN_30HZ=${CARTESIAN_30HZ:-${PROJECT_ROOT}/data/block_grasp_lerobot_v3_cartesian_30hz}
STRIDE5_30HZ=${STRIDE5_30HZ:-${PROJECT_ROOT}/data/block_grasp_lerobot_v3_cartesian_stride5_30hz}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/data/block_grasp_smolvla_6hz}
REPO_ID=${REPO_ID:-threading_real/block_grasp_smolvla_6hz}
VALIDATION_REPORT=${VALIDATION_REPORT:-${PROJECT_ROOT}/data/block_grasp_rgbd_validation.json}

if [[ ! -d "${RAW_ROOT}" ]]; then
  echo "Raw dataset not found: ${RAW_ROOT}" >&2
  exit 2
fi
python -c 'import scipy' || {
  echo "Missing conversion dependency: scipy" >&2
  echo "Install it with: python -m pip install scipy==1.16.3" >&2
  exit 2
}
for output in "${JOINT_30HZ}" "${CARTESIAN_30HZ}" "${STRIDE5_30HZ}" "${OUTPUT_ROOT}"; do
  if [[ -e "${output}" || -e "${output}.building" ]]; then
    echo "Refusing to overwrite existing or partial output: ${output}" >&2
    echo "Inspect it and remove it explicitly before retrying." >&2
    exit 2
  fi
done

cd "${PROJECT_ROOT}"

python validate_vla_rgbd.py "${RAW_ROOT}" --output "${VALIDATION_REPORT}"

python convert_vla_to_lerobot_v3.py \
  "${RAW_ROOT}" \
  "${JOINT_30HZ}" \
  --repo-id="${REPO_ID}_joint_30hz" \
  --task="pick up the block" \
  --fps=30 \
  --image-size=224 \
  --skip-depth

python convert_lerobot_v3_to_cartesian.py \
  "${JOINT_30HZ}" \
  "${CARTESIAN_30HZ}"

python threading_real/scripts/make_cartesian_stride_actions.py \
  "${CARTESIAN_30HZ}" \
  "${STRIDE5_30HZ}" \
  --stride=5

python threading_real/pi05/prepare_dataset.py \
  --source "${STRIDE5_30HZ}" \
  --output "${OUTPUT_ROOT}" \
  --repo-id "${REPO_ID}" \
  --stride=5

python "${SCRIPT_DIR}/preflight.py" \
  --dataset-root "${OUTPUT_ROOT}" \
  --repo-id "${REPO_ID}" \
  --chunk-size 10

echo "Dataset ready: ${OUTPUT_ROOT}"
echo "Depth was validated for stream integrity but was not copied into the LeRobot dataset."
echo "Intermediate datasets were kept for diagnosis; remove them only after inspecting the final output."
