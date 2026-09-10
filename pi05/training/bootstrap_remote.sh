#!/usr/bin/env bash
set -euo pipefail

# Run from the remote project root (normally ~/pi05). This does not transfer data.
ENV_DIR=${ENV_DIR:-$PWD/.venv-pi05}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
# Pin the LeRobot revision used by the dataset, validation split, and training
# scripts so the remote run does not change when PyPI releases move.
LEROBOT_COMMIT=${LEROBOT_COMMIT:-3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 2
fi

if [[ -e "${ENV_DIR}" ]]; then
  echo "Refusing to overwrite existing environment: ${ENV_DIR}" >&2
  exit 2
fi

# uv downloads Python 3.12 when the server does not already provide it. Disable
# the package cache so the nearly-full root filesystem does not retain a second
# copy of downloaded wheels after the environment has been built.
uv venv --python "${PYTHON_VERSION}" "${ENV_DIR}"
uv pip install --python "${ENV_DIR}/bin/python" --no-cache \
  "lerobot[pi,training] @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_COMMIT}"

echo "Environment ready: source ${ENV_DIR}/bin/activate"
echo "LeRobot commit: ${LEROBOT_COMMIT}"
echo "Before training, accept google/paligemma-3b-pt-224, then set HF_TOKEN."
