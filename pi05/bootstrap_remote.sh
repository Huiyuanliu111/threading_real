#!/usr/bin/env bash
set -euo pipefail

# Run manually on the GPU server. This does not transfer project data.
ENV_DIR=${ENV_DIR:-$PWD/.venv-pi05}
PYTHON_BIN=${PYTHON_BIN:-python3.12}
# PyPI 0.4.4 can train pi0.5, but its checkpoint path is not FSDP-safe.  Pin
# the official LeRobot commit that provides native FSDP2 checkpoint/resume.
LEROBOT_COMMIT=${LEROBOT_COMMIT:-3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} is required by the pinned LeRobot source." >&2
  echo "Install Python 3.12, or set PYTHON_BIN to a Python >=3.12 executable." >&2
  exit 2
fi

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip
"${ENV_DIR}/bin/python" -m pip install \
  "lerobot[pi,training] @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_COMMIT}"

echo "Environment ready: source ${ENV_DIR}/bin/activate"
echo "LeRobot commit: ${LEROBOT_COMMIT}"
echo "Before training, accept google/paligemma-3b-pt-224 and run: hf auth login"
