#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENPI_ROOT=${OPENPI_ROOT:-${HERE}/vendor/openpi}
REV=215abfb217dbac7d5f1273282331b9b1866c0479
if [[ ! -e "${OPENPI_ROOT}" ]]; then
  git clone https://github.com/Physical-Intelligence/openpi.git "${OPENPI_ROOT}"
  git -C "${OPENPI_ROOT}" checkout "${REV}"
fi
if [[ $(git -C "${OPENPI_ROOT}" rev-parse HEAD) != "${REV}" ]]; then
  echo "OpenPI checkout must be at ${REV}; refusing to modify an existing checkout." >&2
  exit 1
fi
git -C "${OPENPI_ROOT}" submodule update --init --recursive
cd "${OPENPI_ROOT}"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
echo "Use ${OPENPI_ROOT}/.venv/bin/python for run.py and infer_one.py"
