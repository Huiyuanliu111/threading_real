#!/usr/bin/env bash
set -euo pipefail

# Reproducible single-GPU environment for the RTX 4060 Ti. CUDA 11.8 wheels
# also work on the older driver installed on the A40 server.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
ENV_DIR=${ENV_DIR:-${PROJECT_ROOT}/.venv-smolvla}
UV_BIN=${UV_BIN:-uv}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
LEROBOT_COMMIT=${LEROBOT_COMMIT:-3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu118}

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv is required. Install it first, or set UV_BIN to its path." >&2
  exit 2
fi
if [[ -e "${ENV_DIR}" ]]; then
  echo "Refusing to overwrite existing environment: ${ENV_DIR}" >&2
  echo "Remove it explicitly or set ENV_DIR to a new path." >&2
  exit 2
fi

"${UV_BIN}" python install "${PYTHON_VERSION}"
PYTHON_BIN=$("${UV_BIN}" python find "${PYTHON_VERSION}")
"${UV_BIN}" venv --python "${PYTHON_BIN}" --seed "${ENV_DIR}"

"${ENV_DIR}/bin/python" -m pip install \
  torch==2.7.1 \
  torchvision==0.22.1 \
  torchcodec==0.5 \
  --index-url "${TORCH_INDEX}"
"${ENV_DIR}/bin/python" -m pip install \
  "lerobot[smolvla,training] @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_COMMIT}"
"${ENV_DIR}/bin/python" -m pip install \
  scipy==1.16.3 \
  pin==4.1.0 \
  pyrealsense2==2.58.4.10922

"${ENV_DIR}/bin/python" - <<'PY'
import torch
import lerobot

if torch.__version__ != "2.7.1+cu118":
    raise SystemExit(f"Expected torch 2.7.1+cu118, got {torch.__version__}")
print(f"Environment ready: torch={torch.__version__}, lerobot={lerobot.__version__}")
PY
echo "Activate with: source ${ENV_DIR}/bin/activate"
echo "LeRobot commit: ${LEROBOT_COMMIT}"
